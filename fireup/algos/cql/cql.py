import time

import gym
import d4rl  # Import required to register environments
import numpy as np
import torch
import torch.nn.functional as F

from fireup.algos.cql import core
from fireup.utils.logx import EpochLogger


class ReplayBuffer:
    """
    Offline RL data loader
    """

    def __init__(self, dataset):
        self.size = dataset["observations"].shape[0]
        self.obs1_buf = dataset["observations"]
        self.obs2_buf = dataset["next_observations"]
        self.acts_buf = dataset["actions"]
        self.rews_buf = dataset["rewards"]
        self.done_buf = dataset["terminals"]

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

CQL (Entropy version) with Soft Actor-Critic

(With slight variations that bring it closer to TD3)

"""


def cql(
    env_fn,
    actor_critic=core.ActorCritic,
    ac_kwargs=dict(),
    seed=0,
    steps_per_epoch=5000,
    epochs=100,
    gamma=0.99,
    polyak=0.995,
    lr=1e-3,
    alpha=0.2,
    temperature=1.0,
    num_actions=10,
    lagrangian=False,
    lagrangian_thresh=5.0,
    min_q_weight=5.0,
    optimize_alpha=True,
    batch_size=100,
    max_ep_len=1000,
    logger_kwargs=dict(),
    save_freq=1,
):
    """

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The agent's model which takes the state ``x`` and
            action, ``a`` and returns a tuple of:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       (batch, act_dim)  | Samples actions from policy given
                                           | states.
            ``mu``       (batch, act_dim)  | Computes mean actions from policy
                                           | given states.
            ``logp_pi``  (batch,)          | Gives log probability, according to
                                           | the policy, of the action sampled by
                                           | ``pi``. Critical: must be differentiable
                                           | with respect to policy parameters all
                                           | the way through action sampling.
            ``q1``       (batch,)          | Gives one estimate of Q* for
                                           | states in ``x`` and actions in
                                           | ``a``.
            ``q2``       (batch,)          | Gives another estimate of Q* for
                                           | states in ``x`` and actions in
                                           | ``a``.
            ``q1_pi``    (batch,)          | Gives the composition of ``q1`` and
                                           | ``pi`` for states in ``x_ph``:
                                           | q1(x, pi(x)).
            ``q2_pi``    (batch,)          | Gives the composition of ``q2`` and
                                           | ``pi`` for states in ``x_ph``:
                                           | q2(x, pi(x)).
            ``v``        (batch,)          | Gives the value estimate for states
                                           | in ``x``.
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the actor_critic
            class you provided to SAC.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target
            networks. Target networks are updated towards main networks
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually
            close to 1.)

        lr (float): Learning rate (used for both policy and value learning).

        alpha (float): Entropy regularization coefficient. (Equivalent to
            inverse of reward scale in the original SAC paper.)

        temperature (float): CQL loss temperature.

        num_actions (int): Number of actions to sample for CQL loss.

        lagrangian (bool): Whether to use the Lagrangian for Alpha Prime (in CQL loss).

        lagrangian_thresh (float): Threshold for Lagrangian.

        min_q_weight (float): Minimum Q weight multiplier for CQL loss.

        optimize_alpha (bool): Automatic entropy tuning flag.

        batch_size (int): Minibatch size for SGD.

        max_ep_len (int): Maximum length of trajectory / episode / rollout for testing.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_low = env.action_space.low[0]
    act_high = env.action_space.high[0]

    # Share information about action space with policy architecture
    ac_kwargs["action_space"] = env.action_space

    # Main computation graph
    main = actor_critic(in_features=obs_dim, **ac_kwargs)

    # Target value network
    target = actor_critic(in_features=obs_dim, **ac_kwargs)

    # Offline Experience buffer
    replay_buffer = ReplayBuffer(d4rl.qlearning_dataset(env))

    # Count variables
    var_counts = tuple(
        core.count_vars(module)
        for module in [main.policy, main.q1, main.q2, main.vf_mlp, main]
    )
    print(
        (
            "\nNumber of parameters: \t pi: %d, \t"
            + "q1: %d, \t q2: %d, \t v: %d, \t total: %d\n"
        )
        % var_counts
    )

    # Policy train op
    # (has to be separate from value train op, because q1_pi appears in pi_loss)
    pi_optimizer = torch.optim.Adam(main.policy.parameters(), lr=lr)

    # Value train op
    value_params = (
        list(main.vf_mlp.parameters())
        + list(main.q1.parameters())
        + list(main.q2.parameters())
    )
    value_optimizer = torch.optim.Adam(value_params, lr=lr)

    # alpha optimizer
    if optimize_alpha:
        target_entropy = -np.prod(env.action_space.shape).item()
        log_alpha = torch.zeros(1, requires_grad=True)
        alpha_optimizer = torch.optim.Adam([log_alpha], lr=lr)

    # Initializing targets to match main variables
    target.vf_mlp.load_state_dict(main.vf_mlp.state_dict())

    def get_action(o, deterministic=False):
        pi, mu, _ = main.policy(torch.Tensor(o.reshape(1, -1)))
        return mu.detach().numpy()[0] if deterministic else pi.detach().numpy()[0]

    @torch.inference_mode()
    def test_agent(n=10):
        for _ in range(n):
            o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
            while not (d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time
                o, r, d, _ = test_env.step(get_action(o, True))
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    start_time = time.time()
    total_steps = steps_per_epoch * epochs

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        """
        Train CQL by sampling batches from the offline replay buffer.
        """
        batch = replay_buffer.sample_batch(batch_size)
        (obs1, obs2, acts, rews, done) = (
            torch.Tensor(batch["obs1"]),
            torch.Tensor(batch["obs2"]),
            torch.Tensor(batch["acts"]),
            torch.Tensor(batch["rews"]),
            torch.Tensor(batch["done"]),
        )
        _, _, logp_pi, q1, q2, q1_pi, q2_pi, v = main(obs1, acts)
        v_targ = target.vf_mlp(obs2)

        # Automatic entropy tuning
        if optimize_alpha:
            alpha_loss = -(log_alpha * (logp_pi + target_entropy).detach()).mean()
            alpha_optimizer.zero_grad()
            alpha_loss.backward()
            alpha_optimizer.step()
            alpha = log_alpha.exp()
            logger.store(LossAlpha=alpha_loss.item(), Alpha=alpha.item())

        # Min Double-Q:
        min_q_pi = torch.min(q1_pi, q2_pi)

        # Targets for Q and V regression
        q_backup = (rews + gamma * (1 - done) * v_targ).detach()
        v_backup = (min_q_pi - alpha * logp_pi).detach()

        # Soft actor-critic losses
        pi_loss = (alpha * logp_pi - min_q_pi).mean()
        q1_loss = 0.5 * F.mse_loss(q1, q_backup)
        q2_loss = 0.5 * F.mse_loss(q2, q_backup)
        v_loss = 0.5 * F.mse_loss(v, v_backup)
        value_loss = q1_loss + q2_loss + v_loss

        # CQL loss
        rand_act = torch.Tensor(
            np.random.uniform(
                act_low,
                act_high,
                size=(num_actions * act_dim, env.action_space.shape[-1]),
            )
        )

        # Policy train op
        pi_optimizer.zero_grad()
        pi_loss.backward()
        pi_optimizer.step()

        # Value train op
        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()

        # Polyak averaging for target parameters
        for p_main, p_target in zip(
            main.vf_mlp.parameters(), target.vf_mlp.parameters()
        ):
            p_target.data.copy_(polyak * p_target.data + (1 - polyak) * p_main.data)

        logger.store(
            LossPi=pi_loss.item(),
            LossQ1=q1_loss.item(),
            LossQ2=q2_loss.item(),
            LossV=v_loss.item(),
            Q1Vals=q1.detach().numpy(),
            Q2Vals=q2.detach().numpy(),
            VVals=v.detach().numpy(),
            LogPi=logp_pi.detach().numpy(),
        )

        # End of epoch wrap-up
        if t > 0 and t % steps_per_epoch == 0:
            epoch = t // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs - 1):
                logger.save_state({"env": env}, main, None)

            # Test the performance of the deterministic version of the agent.
            test_agent()

            # Log info about epoch
            logger.log_tabular("Epoch", epoch)
            logger.log_tabular("TestEpRet", with_min_and_max=True)
            logger.log_tabular("TestEpLen", average_only=True)
            logger.log_tabular("TotalEnvInteracts", t)
            logger.log_tabular("Q1Vals", with_min_and_max=True)
            logger.log_tabular("Q2Vals", with_min_and_max=True)
            logger.log_tabular("VVals", with_min_and_max=True)
            logger.log_tabular("LogPi", with_min_and_max=True)
            logger.log_tabular("LossPi", average_only=True)
            logger.log_tabular("LossQ1", average_only=True)
            logger.log_tabular("LossQ2", average_only=True)
            logger.log_tabular("LossV", average_only=True)
            if optimize_alpha:
                logger.log_tabular("LossAlpha", average_only=True)
                logger.log_tabular("Alpha", average_only=True)
            logger.log_tabular("Time", time.time() - start_time)
            logger.dump_tabular()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="HalfCheetah-v2")
    parser.add_argument("--hid", type=int, default=300)
    parser.add_argument("--l", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", "-s", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--exp_name", type=str, default="sac")
    args = parser.parse_args()

    from fireup.utils.run_utils import setup_logger_kwargs

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    cql(
        lambda: gym.make(args.env),
        actor_critic=core.ActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
        gamma=args.gamma,
        seed=args.seed,
        epochs=args.epochs,
        logger_kwargs=logger_kwargs,
    )