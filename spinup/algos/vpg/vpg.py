import numpy as np
import gym
import time
import spinup.algos.vpg.core as core
from spinup.utils.logx import EpochLogger
import torch
from torch.autograd import Variable
#from spinup.utils.mpi_tf import MpiAdamOptimizer, sync_all_params
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs


class VPGBuffer:
    """
    A buffer for storing trajectories experienced by a VPG agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def _combined_shape(self, length, shape=None):
        if shape is None:
            return (length,)
        return (length, shape) if np.isscalar(shape) else (length, *shape)


    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(self._combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(self._combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        return [self.obs_buf, self.act_buf, self.adv_buf, 
                self.ret_buf, self.logp_buf]

def vpg(env_fn, 
        actor_critic=core.ActorCritic,
        ac_kwargs=dict(),
        seed=0, 
        steps_per_epoch=4000,
        epochs=50,
        gamma=0.99,
        pi_lr=3e-4,
        vf_lr=1e-3,
        train_v_iters=80,
        lam=0.97,
        max_ep_len=1000,
        logger_kwargs=dict(),
        save_freq=10):
    
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space

    actor_critic = actor_critic(in_features=obs_dim[0], **ac_kwargs)

    # Experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = VPGBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

    train_pi = torch.optim.Adam(actor_critic.policy.parameters(),
                                lr=pi_lr)
    train_v = torch.optim.Adam(actor_critic.value_function.parameters(),
                               lr=vf_lr)


    start_time = time.time()
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):

        actor_critic.eval()
        for t in range(local_steps_per_epoch):
            a, _, logp_t, v_t = actor_critic(torch.Tensor(o.reshape(1,-1)))

            # save and log
            buf.store(o, a.data.numpy(), r, v_t.item(), logp_t.data.numpy())
            logger.store(VVals=v_t)

            o, r, d, _ = env.step(a.data.numpy()[0])
            ep_ret += r
            ep_len += 1

            terminal = d or (ep_len == max_ep_len)
            if terminal or (t==local_steps_per_epoch-1):
                if not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len)
                # if trajectory didn't reach terminal state, bootstrap value target
                last_val = r if d else actor_critic.value_function(torch.Tensor(o.reshape(1,-1))).item()
                buf.finish_path(last_val)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        # Perform VPG update!
        actor_critic.train()
        train_pi.zero_grad()

        x, a, adv, ret, logp_old = [Variable(torch.Tensor(x)) for x in buf.get()]
        
        # Main outputs from computation graph
        pi, logp, logp_pi, v = actor_critic(x, a)

         # VPG objectives
        pi_loss = -torch.mean(logp * adv)
        v_loss = torch.mean((ret - v)**2)

        # Info (useful to watch during learning)
        # a sample estimate for KL-divergence, easy to compute
        approx_kl = torch.mean(logp_old - logp)
        # a sample estimate for entropy, also easy to compute
        ent = torch.mean(-logp)

        # Policy gradient step
        pi_loss.backward()
        train_pi.step()
        
        # Value function learning
        for _ in range(train_v_iters):
            train_v.zero_grad()
            v_loss.backward(retain_graph=True)
            train_v.step()

        # Log changes from update
        pi, logp, logp_pi, v = actor_critic(x, a)
        pi_l_new = -torch.mean(logp * adv)
        v_l_new = torch.mean((ret - v)**2)
        kl = torch.mean(logp_old - logp)
        logger.store(LossPi=pi_loss, LossV=v_loss, 
                     KL=kl, Entropy=ent, 
                     DeltaLossPi=(pi_l_new - pi_loss),
                     DeltaLossV=(v_l_new - v_loss))

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()