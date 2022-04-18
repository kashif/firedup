import time

import gym
import numpy as np
import torch
import torch.nn.functional as F

from fireup.algos.dqn import core
from fireup.utils.logx import EpochLogger


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DQN agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(
            obs1=self.obs1_buf[idxs],
            obs2=self.obs2_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs],
        )


"""

Deep Q-Network

"""


def dqn(
    env_fn,
    dqnetwork=core.DQNetwork,
    ac_kwargs=dict(),
    seed=0,
    steps_per_epoch=5000,
    epochs=100,
    replay_size=int(1e6),
    gamma=0.99,
    min_replay_history=20000,
    epsilon_decay_period=250000,
    epsilon_train=0.01,
    epsilon_eval=0.001,
    lr=1e-3,
    max_ep_len=1000,
    update_period=4,
    target_update_period=8000,
    batch_size=100,
    logger_kwargs=dict(),
    save_freq=1,
):

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape[0]
    act_dim = 1  # env.action_space.shape

    # Share information about action space with policy architecture
    ac_kwargs["action_space"] = env.action_space

    # Main computation graph
    main = dqnetwork(in_features=obs_dim, **ac_kwargs)

    # Target network
    target = dqnetwork(in_features=obs_dim, **ac_kwargs)

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [main.q, main])
    print(("\nNumber of parameters: \t q: %d, \t total: %d\n") % var_counts)

    # Value train op
    value_params = main.q.parameters()
    value_optimizer = torch.optim.Adam(value_params, lr=lr)

    # Initializing targets to match main variables
    target.load_state_dict(main.state_dict())

    def get_action(o, epsilon):
        """Select an action from the set of available actions.
        Chooses an action randomly with probability epsilon otherwise
        act greedily according to the current Q-value estimates.
        """
        if np.random.random() <= epsilon:
            return env.action_space.sample()
        else:
            q_values = main(torch.Tensor(o.reshape(1, -1)))
            # return the action with highest Q-value for this observation
            return torch.argmax(q_values, dim=1).item()

    @torch.inference_mode()
    def test_agent(n=10):
        for _ in range(n):
            o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
            while not (d or (ep_len == max_ep_len)):
                # epsilon_eval used when evaluating the agent
                o, r, d, _ = test_env.step(get_action(o, epsilon_eval))
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    start_time = time.time()
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    total_steps = steps_per_epoch * epochs

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        main.eval()

        # the epsilon value used for exploration during training
        epsilon = core.linearly_decaying_epsilon(
            epsilon_decay_period, t, min_replay_history, epsilon_train
        )
        a = get_action(o, epsilon)

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len == max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        # train at the rate of update_period if enough training steps have been run
        if replay_buffer.size > min_replay_history and t % update_period == 0:
            main.train()
            batch = replay_buffer.sample_batch(batch_size)
            (obs1, obs2, acts, rews, done) = (
                torch.Tensor(batch["obs1"]),
                torch.Tensor(batch["obs2"]),
                torch.Tensor(batch["acts"]),
                torch.Tensor(batch["rews"]),
                torch.Tensor(batch["done"]),
            )
            q_pi = main(obs1).gather(1, acts.long()).squeeze()
            q_pi_targ, _ = target(obs2).max(1)

            # Bellman backup for Q function
            backup = (rews + gamma * (1 - done) * q_pi_targ).detach()

            # DQN loss
            value_loss = F.smooth_l1_loss(q_pi, backup)

            # Q-learning update
            value_optimizer.zero_grad()
            value_loss.backward()
            value_optimizer.step()
            logger.store(LossQ=value_loss.item(), QVals=q_pi.data.numpy())

        # syncs weights from online to target network
        if t % target_update_period == 0:
            target.load_state_dict(main.state_dict())

        # End of epoch wrap-up
        if replay_buffer.size > min_replay_history and t % steps_per_epoch == 0:
            epoch = t // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs - 1):
                logger.save_state({"env": env}, main, None)

            # Test the performance of the deterministic version of the agent.
            test_agent()

            # Log info about epoch
            logger.log_tabular("Epoch", epoch)
            logger.log_tabular("EpRet", with_min_and_max=True)
            logger.log_tabular("TestEpRet", with_min_and_max=True)
            logger.log_tabular("EpLen", average_only=True)
            logger.log_tabular("TestEpLen", average_only=True)
            logger.log_tabular("TotalEnvInteracts", t)
            logger.log_tabular("QVals", with_min_and_max=True)
            logger.log_tabular("LossQ", average_only=True)
            logger.log_tabular("Time", time.time() - start_time)
            logger.dump_tabular()
