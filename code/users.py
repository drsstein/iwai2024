import numpy as np

class UserSimulator():

    def __init__(self, f0, f1, target, rng):
        self.f0 = f0
        self.f1 = f1
        self.target = target
        self.rng = rng

    def step(self, o_app):
        p = self.emission_probability(o_app)
        return self.rng.choice(np.arange(2), p=p)

    def reset(self):
      pass

    def emission_probability(self, o_app):
        f0 = self.f0
        f1 = self.f1
        target = self.target
        
        is_equal = np.isclose(target, o_app)
        is_lower = target < o_app
        p = np.empty(2)
        p[0] = 0.5*is_equal + (1-is_equal)*( (1-f0) * is_lower + f1 * (1-is_lower) )
        p[1] = 0.5*is_equal + (1-is_equal)*( f0 * is_lower + (1-f1) * (1-is_lower) )
        return p

class UserModel():

  def __init__(self, num_targets):
    self.num_targets = num_targets

  def emission_probability(self, user_state, app_action):
    """
    Vectorized user emission probability.
    user_state (np.array): (num_particles, num_state_dims)
    app_action (np.array): (num_plans, num_timesteps)
    """
    f0 = user_state[:,0][:,None, None]
    f1 = f0
    target = user_state[:,-1][:,None, None]
    
    is_equal = np.isclose(target, app_action) # (particle x plan x timestep)
    is_lower = target < app_action
    p = np.empty(shape=(2,) + is_lower.shape)
    p[0] = 0.5*is_equal + (1-is_equal)*( (1-f0) * is_lower + f1 * (1-is_lower) )
    p[1] = 0.5*is_equal + (1-is_equal)*( f0 * is_lower + (1-f1) * (1-is_lower) )
    return p