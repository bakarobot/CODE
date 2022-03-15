from create_model import Model
from evaluate_policy import policy_eval
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

def plot_state_and_value(v_star):
    cmap = plt.cm.get_cmap('YlOrBr')
    colors = cmap(v_star/11.0)
    plt.scatter(list(range(len(v_star))), np.zeros(10), c=colors)
    sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 11))
    cbar = plt.colorbar(sm)
    plt.pause(0.001)
    plt.clf()

def value_iteration(threshold = .001):
    nums, R, P, cfvs = Model()
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
        plot_state_and_value(V_star)
        time.sleep(0.2)
        iteration+=1
    return V_star, pi_star, iteration

def pi_star_from_V_star():
    nums, R, P, cfvs = Model()
    V_star, _ = value_iteration()
    pi_star = np.zeros(len(V_star))
    for s in range(nums):
        pi_star[s] = np.argmax(gam * V_star @ P[:, s, :] + R[s, :])
    return pi_star

V_star, pi_star, iteration = value_iteration()
print("V_star: ", V_star)
print("pi_star: ", pi_star)
print("Iteration run: ", iteration)