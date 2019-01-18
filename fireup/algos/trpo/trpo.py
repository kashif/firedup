import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import vector_to_parameters
import gym
from gym.spaces import Box
import time
import scipy.signal
import fireup.algos.trpo.core as core
from fireup.utils.logx import EpochLogger
from fireup.utils.mpi_torch import sync_all_params, average_gradients
from fireup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs


EPS = 1e-8

class GAEBuffer:
    """
    A buffer for storing trajectories experienced by a TRPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, info_shapes, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(self._combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(self._combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.info_bufs = {k: np.zeros([size] + list(v), dtype=np.float32) for k,v in info_shapes.items()}
        self.sorted_info_keys = core.keys_as_sorted_list(self.info_bufs)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp, info):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        for i, k in enumerate(self.sorted_info_keys):
            self.info_bufs[k][self.ptr] = info[i]
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
        self.adv_buf[path_slice] = self._discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = self._discount_cumsum(rews, self.gamma)[:-1]

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
        return [self.obs_buf, self.act_buf, self.adv_buf, self.ret_buf,
                self.logp_buf] + core.values_as_sorted_list(self.info_bufs)

    def _combined_shape(self, length, shape=None):
        if shape is None:
            return (length,)
        return (length, shape) if np.isscalar(shape) else (length, *shape)

    def _discount_cumsum(self, x, discount):
        """
        magic from rllab for computing discounted cumulative sums of vectors.

        input:
            vector x,
            [x0,
            x1,
            x2]

        output:
            [x0 + discount * x1 + discount^2 * x2,
            x1 + discount * x2,
            x2]
        """
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

"""

Trust Region Policy Optimization

(with support for Natural Policy Gradient)

"""
def trpo(env_fn, actor_critic=core.ActorCritic, ac_kwargs=dict(), seed=0,
         steps_per_epoch=4000, epochs=50, gamma=0.99, delta=0.01, vf_lr=1e-3,
         train_v_iters=80, damping_coeff=0.1, cg_iters=10, backtrack_iters=10,
         backtrack_coeff=0.8, lam=0.97, max_ep_len=1000, logger_kwargs=dict(),
         save_freq=10, algo='trpo'):
    """

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The agent's main model which for state ``x`` and
            action, ``a`` returns the following outputs:

            ============  ================  ========================================
            Symbol        Shape             Description
            ============  ================  ========================================
            ``pi``        (batch, act_dim)  | Samples actions from policy given
                                            | states.
            ``logp``      (batch,)          | Gives log probability, according to
                                            | the policy, of taking actions ``a``
                                            | in states ``x``.
            ``logp_pi``   (batch,)          | Gives log probability, according to
                                            | the policy, of the action sampled by
                                            | ``pi``.
            ``info``      N/A               | A dict of any intermediate quantities
                                            | (from calculating the policy or log
                                            | probabilities) which are needed for
                                            | analytically computing KL divergence.
                                            | (eg sufficient statistics of the
                                            | distributions)
            ``info_phs``  N/A               | A dict of placeholders for old values
                                            | of the entries in ``info``.
            ``d_kl``      ()                | The mean KL
                                            | divergence between the current policy
                                            | (``pi``) and the old policy (as
                                            | specified by the inputs to
                                            | ``info``) over the batch of
                                            | states given in ``x``.
            ``v``         (batch,)          | Gives the value estimate for states
                                            | in ``x``. (Critical: make sure
                                            | to flatten this!)
            ============  ================  ========================================

        ac_kwargs (dict): Any kwargs appropriate for the actor_critic
            function you provided to TRPO.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        delta (float): KL-divergence limit for TRPO / NPG update.
            (Should be small for stability. Values like 0.01, 0.05.)

        vf_lr (float): Learning rate for value function optimizer.

        train_v_iters (int): Number of gradient descent steps to take on
            value function per epoch.

        damping_coeff (float): Artifact for numerical stability, should be
            smallish. Adjusts Hessian-vector product calculation:

            .. math:: Hv \\rightarrow (\\alpha I + H)v

            where :math:`\\alpha` is the damping coefficient.
            Probably don't play with this hyperparameter.

        cg_iters (int): Number of iterations of conjugate gradient to perform.
            Increasing this will lead to a more accurate approximation
            to :math:`H^{-1} g`, and possibly slightly-improved performance,
            but at the cost of slowing things down.

            Also probably don't play with this hyperparameter.

        backtrack_iters (int): Maximum number of steps allowed in the
            backtracking line search. Since the line search usually doesn't
            backtrack, and usually only steps back once when it does, this
            hyperparameter doesn't often matter.

        backtrack_coeff (float): How far back to step during backtracking line
            search. (Always between 0 and 1, usually above 0.5.)

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

        algo: Either 'trpo' or 'npg': this code supports both, since they are
            almost the same.

    """

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

    # Main model
    actor_critic = actor_critic(in_features=obs_dim[0], **ac_kwargs)

    # Experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    if isinstance(env.action_space, Box):
        info_shapes = {'old_mu': [1, env.action_space.shape[-1]], 
                       'old_log_std': [env.action_space.shape[-1]]}
    else:
        info_shapes = {'old_logit': [env.action_space.n]}
    buf = GAEBuffer(obs_dim, act_dim, local_steps_per_epoch, info_shapes, gamma, lam)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in
        [actor_critic.policy, actor_critic.value_function])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    # Optimizer for value function
    train_vf = torch.optim.Adam(actor_critic.value_function.parameters(), lr=vf_lr)

    # Sync params across processes
    sync_all_params(actor_critic.parameters())
    
    def cg(Ax, b):
        """
        Conjugate gradient algorithm
        (see https://en.wikipedia.org/wiki/Conjugate_gradient_method)
        """
        x = torch.zeros_like(b)
        r = b.clone() # Note: should be 'b - Ax(x)', but for x=0, Ax(x)=0. Change if doing warm start.
        p = b.clone()
        r_dot_old = torch.dot(r,r)
        for _ in range(cg_iters):
            z = Ax(p)
            alpha = r_dot_old / (torch.dot(p, z) + EPS)
            x += alpha * p
            r -= alpha * z
            r_dot_new = torch.dot(r,r)
            p = r + (r_dot_new / r_dot_old) * p
            r_dot_old = r_dot_new
        return x

    def update():
        inputs = [torch.Tensor(x) for x in buf.get()]
        obs, act, adv, ret, logp_old = inputs[:-len(buf.sorted_info_keys)]
        policy_args = dict(zip(buf.sorted_info_keys, inputs[-len(buf.sorted_info_keys):]))

        # Main outputs from computation graph
        _, logp, _, _, d_kl, v = actor_critic(obs, act, **policy_args)

        # Prepare hessian func, gradient eval
        ratio = (logp - logp_old).exp()     # pi(a|s) / pi_old(a|s)
        pi_l_old = -(ratio * adv).mean()
        v_l_old = F.mse_loss(v, ret)

        g = core.flat_grad(pi_l_old, actor_critic.policy.parameters(), retain_graph=True).detach()
        g.data.copy_(torch.tensor(mpi_avg(g.data.numpy())))
        pi_l_old = mpi_avg(pi_l_old.item())
        
        def Hx(x):
            x, hvp = core.hessian_vector_product(d_kl, actor_critic.policy, x)
            if damping_coeff > 0:
                hvp += damping_coeff * x
            return x

        # Core calculations for TRPO or NPG
        x = cg(Hx, g)
        alpha = torch.sqrt(2*delta/(torch.dot(x, Hx(x))+EPS))

        def set_and_eval(step):
            vector_to_parameters(pi_l_old - alpha * x * step, actor_critic.policy.parameters())
            _, logp, _, _, d_kl = actor_critic.policy(obs, act, **policy_args)
            ratio = (logp - logp_old).exp()
            pi_loss = -(ratio * adv).mean()
            return mpi_avg(d_kl.item()), mpi_avg(pi_loss.item())

        if algo=='npg':
            kl, pi_l_new = set_and_eval(step=1.)
        
        elif algo=='trpo':
            for j in range(backtrack_iters):
                kl, pi_l_new = set_and_eval(step=backtrack_coeff**j)
                if kl <= delta and pi_l_new <= pi_l_old:
                    logger.log('Accepting new params at step %d of line search.'%j)
                    logger.store(BacktrackIters=j)
                    break

                if j==backtrack_iters-1:
                    logger.log('Line search failed! Keeping old params.')
                    logger.store(BacktrackIters=j)
                    kl, pi_l_new = set_and_eval(step=0.)
        
        # Value function updates
        for _ in range(train_v_iters):
            v = actor_critic.value_function(obs)
            v_loss = F.mse_loss(v, ret)

            # Value function gradient step
            train_vf.zero_grad()
            v_loss.backward()
            average_gradients(train_vf.param_groups)
            train_vf.step()
        
        v = actor_critic.value_function(obs)
        v_l_new = F.mse_loss(v, ret)
        
        # Log changes from update
        logger.store(LossPi=pi_l_old, LossV=v_l_old, KL=kl,
                     DeltaLossPi=(pi_l_new - pi_l_old),
                     DeltaLossV=(v_l_new - v_l_old))

    start_time = time.time()
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        actor_critic.eval()
        for t in range(local_steps_per_epoch):
            a, _, logp_t, info_t, _, v_t = actor_critic(torch.Tensor(o.reshape(1,-1)))

            # save and log
            buf.store(o, a.data.numpy(), r, v_t.item(), logp_t.data.numpy(),
                      core.values_as_sorted_list(info_t))
            logger.store(VVals=v_t)

            o, r, d, _ = env.step(a)
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

        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_state({'env': env}, None)

        # Perform TRPO or NPG update!
        actor_critic.train()
        update()

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
        logger.log_tabular('KL', average_only=True)
        if algo=='trpo':
            logger.log_tabular('BacktrackIters', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='trpo')
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi

    from fireup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    trpo(lambda : gym.make(args.env), actor_critic=core.ActorCritic,
         ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma,
         seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
         logger_kwargs=logger_kwargs)
