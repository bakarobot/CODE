from create_gridworld import SimpleGridWorld
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import time

gam = 0.9

# Value Interation to find V*
# V_{k+1}(s) = max_a [ R(a,s) + gamma sum_s' P(s'|s,a]) V_k(s') ]

plt.figure(0)
plt.ion()
plt.show()

def plot_state_and_value_gridworld(v_star):
    cmap = plt.cm.get_cmap('YlOrBr')
    colors = cmap(v_star/max(v_star))
    plt.scatter(6*list(range(8)), sorted(8*list(range(6))), c=colors)
    sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max(v_star)))
    plt.gca().set_xlim((-1,8))
    plt.gca().set_ylim((-1,6))
    cbar = plt.colorbar(sm)
    plt.pause(0.001)
    plt.clf()

def value_iteration(threshold = .001):
    nums, R, P, cfvs = SimpleGridWorld()
    V_star = np.zeros(nums)
    pi_star = np.zeros(nums)
    delta = np.inf
    iteration = 0
    while delta >= threshold:
        delta = 0
        for s in range(nums):
            v = V_star[s]
            V_star[s] = max(gam * V_star @ P[:, s, :] + R[s, :])
            pi_star[s] = np.argmax(gam * V_star @ P[:, s, :] + R[s, :])
            delta = max(delta, abs(v - V_star[s]))
            plot_state_and_value_gridworld(V_star)
        iteration+=1
    return V_star, pi_star, iteration

V_star, pi_star, iteration = value_iteration()
print("V_star: ", V_star)
print("pi_star: ", pi_star)
print("Iteration run: ", iteration)