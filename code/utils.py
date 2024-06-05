import matplotlib.pyplot as plt
from matplotlib.markers import CARETLEFT, CARETRIGHT, CARETUP, CARETDOWN

import numpy as np

def plot_interaction_sequence(oo_user, oo_system, user_target, ax=None):
  if ax is None:
    _, ax = plt.subplots()

  oo_user = np.asarray(oo_user)
  oo_system = np.asarray(oo_system)
  num_steps = len(oo_user)
  # plot system output
  ax.step(np.arange(num_steps+1), oo_system, where='post', label='agent output')
    
  # plot user feedback
  i_down = np.isclose(oo_user, 0)
  i_up = np.isclose(oo_user, 1)
  ax.scatter(np.arange(num_steps)[i_down]+0.5, oo_system[:-1][i_down], c='red', marker=CARETDOWN, s=8**2, label='user feedback')
  ax.scatter(np.arange(num_steps)[i_up]+0.5, oo_system[:-1][i_up], c='red', s=8**2, marker=CARETUP)
    
  # plot user target
  ax.plot([0, num_steps], [user_target]*2, 'r--', label=f'user target ({user_target})')
  ax.set_xlabel('interface step')
  ax.set_ylabel('possible target')
  ax.legend()

def plot_target_belief_distribution_sequence(beliefs, user_target, cscale='linear', ax=None):
    if ax is None:
        _, ax = plt.subplots()

    num_steps = len(beliefs)
    img = np.log(np.asarray(beliefs)).T if cscale=='log' else np.asarray(beliefs).T
    # , extent=[0, len(beliefs), 0, 1]
    ax.imshow(img, interpolation='nearest', origin='lower', aspect='auto')
    ax.plot([0, num_steps], [user_target]*2, 'r--', label=f'user target ({user_target})')
    ax.set_xlabel('interface step')
    ax.set_ylabel('possible target')
    ax.set_title('Marginal belief over target probabilities Q( target ).');
    ax.legend()

def plot_f_belief_distribution_sequence(beliefs, user_f, bins, cscale='linear', ax=None):
    if ax is None:
        _, ax = plt.subplots()
    
    num_steps = len(beliefs)
    img = np.log(np.asarray(beliefs)).T if cscale=='log' else np.asarray(beliefs).T
    # , extent=[0, len(beliefs), 0, 1]
    ax.imshow(img, interpolation='nearest', origin='lower', aspect='auto', extent=[0, num_steps, bins[0], bins[-1]])
    ax.plot([0, num_steps], [user_f]*2, 'r--', label=f'user f ({user_f})')
    ax.set_xlabel('interface step')
    ax.set_ylabel('possible error rate')
    ax.set_title('Marginal belief over flip probabilities Q( f ).')
    ax.legend()