import matplotlib.pyplot as plt
import numpy as np

def np_safelog(x):
  return np.log( np.maximum(x, 1e-16) )

def softmax(x):
  e = np.exp(x - x.max())
  return e / e.sum()
    
class AInfAgentContinuous:
    """ Minimal agent performing exact inference in fully discrete POMDPs"""
    def __init__(self, 
                 model, # model of the environment
                 rng, # random number generator
                 max_f, # maximum value for uniform prior (1 if unknown polarity, 0.5 if polarity is known)
                 init_particles = None, # initial non-uniform belief over error rate
                 num_particles=1000, # number of discrete samples representing the belief distribution
                 diffusion_scale_f=0.005, # standard deviation of input error belief diffusion in each timestep
                 diffusion_scale_target=0.1, # standard deviation of user target belief diffusion in each timestep
                 planning_horizon=2, # planning horizon
                 inverse_temperature=1, # Bolzmann inverse temperature for policy sampling (has no effect when select_optimal_plan=True; higher value leads to more peaked distributions, making it more likely to sample the optimal policy).
                 use_info_gain=True, # score actions by info gain
                 weight_info_gain=1.0, # increase or decrease relative contribution of info_gain to expected free energy (useful when pragmatic value discourages sufficient exploration)
                 use_pragmatic_value=True, # score actions by pragmatic value
                 select_optimal_plan=False): # sample plan (False), select max negEFE (True).
        self.model = model
        self.rng = rng
        self.max_f = max_f
        self.init_particles = init_particles
        self.num_particles = num_particles
        self.diffusion_scale_f = diffusion_scale_f
        self.diffusion_scale_target = diffusion_scale_target
        self.planning_horizon = planning_horizon
        self.inverse_temperature = inverse_temperature
        self.use_info_gain = use_info_gain
        self.weight_info_gain = weight_info_gain
        self.use_pragmatic_value = use_pragmatic_value
        self.select_optimal_plan = select_optimal_plan
        # print(f'Enumerating {self.model.num_targets**planning_horizon:,} candidate policies of length {planning_horizon}')
        self.plans = np.stack(np.meshgrid(*[np.arange(self.model.num_targets) for _ in range(planning_horizon)])).T.reshape(-1, planning_horizon)

        # state
        self.last_action = None # last AInf agent action

    def _init_particles(self):
        q = np.zeros((self.num_particles, 2))
        
        # uniformly sample statespace within bounds
        q[:,0] = self.rng.uniform(high=self.max_f, size=self.num_particles)
        q[:,-1] = self.rng.uniform(high=self.model.num_targets, size=self.num_particles)
        w = np.ones(self.num_particles) / self.num_particles

        if self.init_particles is not None:
            w = self.init_particles['w']
            q = self.init_particles['q']
            # required because _resample only works on self.particles and has no return value
            # self.particles = {'w': w, 'q': q}
            # self._resample_particles()
            # w = self.particles['w']
            # q = self.particles['q']
            
            q[:,-1] = self.rng.uniform(high=self.model.num_targets, size=self.num_particles)
            
        return {'w': w, 'q': q}
        
    def reset(self):
        # initialize state prior
        self.particles = self._init_particles()
        plan = self._select_plan()
        self.last_action = plan[0]
        return self.last_action

    @staticmethod
    def effective_sample_size(w):
        return 1/np.sum(w**2)

    def _diffuse_particles(self):
        """Apply diffusion to particles."""
        q = self.particles['q']
        q[:,0] = np.clip(q[:,0] + self.rng.normal(scale=self.diffusion_scale_f, size=q.shape[0]), 0, 1)
        if q.shape[1] > 2:
            q[:,1] = np.clip(q[:,1] + self.rng.normal(scale=self.diffusion_scale_f, size=q.shape[0]), 0, 1)
            
        q[:,-1] = np.clip(q[:,-1] + self.rng.normal(scale=self.diffusion_scale_target, size=q.shape[0]), 0, self.model.num_targets-1e-6)
        self.particles['q'] = q

    def _resample_particles(self):
        """Resample particles using low-variance sampling."""
        n_samples = self.num_particles
        r = self.rng.uniform()/n_samples + np.arange(n_samples) / n_samples
        w_c = np.cumsum(self.particles['w'])
        indices = [np.argmax(w_c > r[i]) for i in range(n_samples)]
        self.particles['q'] = self.particles['q'][indices]
        self.particles['w'] = np.ones(n_samples) / n_samples

    def _update_particles_from_observation(self, o):
        # diffuse first (guarantees no copies after resampling)
        self._diffuse_particles()
        # update belief based on observation
        w_ = self.particles['w'] * self.model.emission_probability(user_state=self.particles['q'], app_action=self.last_action).squeeze()[int(o)]
        self.particles['w'] = w_ / w_.sum()
        if self.effective_sample_size(self.particles['w']) < 0.5 * self.num_particles:
            self._resample_particles()

    def marginal_q_target(self, particles, i=1):
        num_targets = self.model.num_targets
        return np.histogram(particles['q'][:,i], bins=num_targets, range=(0, num_targets), weights=particles['w'], density=True)[0]

    def marginal_q_flip(self, particles=None, bins=10, range=(0,1) ):
        particles =  particles if particles is not None else self.particles
        return np.histogram(particles['q'][:,0], bins=bins, range=range, weights=particles['w'], density=True)
    
    def show_particles(self, particles=None, ax=None, size=2**2):
        particles =  particles if particles is not None else self.particles
        if ax is None:
            fig, ax = plt.subplots()
            
        ax.scatter(particles['q'][:,1], particles['q'][:,0], c=particles['w'], s=size)

    def copy_particles(self):
        return {'w': self.particles['w'].copy(), 'q': self.particles['q'].copy()}
        
    def step(self, o):
        self._update_particles_from_observation(o)
        plan = self._select_plan()
        self.last_action = plan[0]
        return self.last_action
        
    def _select_plan(self, debug=False):  
      # 1. rollout all plans
      # in this scenario, all p(s1 | s0, a) are the identity mapping, so we don't need the rollout
      # i.e., our belief doesn't change as a result of taking actions
      q_ss = self.particles['w']

      # pragmatic value v2 (deprecated)
      log_p_c = np_safelog( self.marginal_q_target(self.particles) )
      self.pragmatic = log_p_c[self.plans].mean(axis=1)
        
      # state info gain
      p_o_given_s = np.asarray(self.model.emission_probability(user_state=self.particles['q'], app_action=self.plans)) # (user_input x particle x plan x timestep)
      p_o_given_s = np.einsum('osa...->a...so', p_o_given_s) # (plan x timestep x particle x user output)
      p_s_and_o = np.expand_dims(q_ss, axis=-1) * p_o_given_s # (plan x timestep x particle x user output) weighted by belief
      p_o_marginal = p_s_and_o.sum(axis=-2) # (plan x timestep x user output) prior belief over user outputs
      q_s_given_o = p_s_and_o / p_s_and_o.sum(axis=-2, keepdims=True)
      d_kl = (q_s_given_o * (np_safelog( q_s_given_o ) - np_safelog( q_ss )[...,None])).sum( axis=2 ) # KL
      info_gain = (d_kl * p_o_marginal).mean(axis=(1, 2)) # sum over o and t
      self.info_gain = info_gain

      # pragmatic value v3
      # q_s_target_given_o = np.asarray([[[self.marginal_q_target({'w':q, 'q': self.particles['q']}) for q in q_s_given_o[plan, t].T] for t in range(self.plans.shape[1])] for plan in range(self.plans.shape[0])])
      # q_s_target = self.marginal_q_target(self.particles)
      # d_kl_target = (q_s_target_given_o * (np_safelog(q_s_target_given_o) - np_safelog(q_s_target))).sum(axis=-1)
      # info_gain_target = (d_kl_target * p_o_marginal).mean(axis=(1, 2)) # sum over t and o
      # self.pragmatic = info_gain_target
        
      # expected belief after observing user input (sophisticated inference)
      # q_s_next = (q_s_given_o * p_o_marginal[:,:,None,:]).sum(axis=-1)
      # self.pragmatic = [np.mean([self.marginal_q_target(particles={
      #             'w': q_s_next[plan, t].squeeze(), 'q': self.particles['q']})[plan[t]] for t in range(q_s_next.shape[1])]) for plan in self.plans]
        
      #action selection
      nefe = self.use_pragmatic_value * (1-self.weight_info_gain) * self.pragmatic + self.use_info_gain * self.weight_info_gain * self.info_gain
      self.nefe = nefe
      p_plans = softmax(self.inverse_temperature * nefe)
      if self.select_optimal_plan:
          return self.plans[ np.argmax(nefe) ]
      else:    
          return self.plans[ np.random.choice( self.plans.shape[0], p=p_plans ) ]