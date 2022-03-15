import numpy as np
from create_model import Model

numa = 2  # number of actions
gam = 0.9 # Discount Factor

# POLICY EVALUATION: Given a policy determine its performance
# Find Performace V^pi of a given policy pi
# V^pi(s)  = \sum_s' P(s,a,s')( r(s,a) + gamma V^pi(s') )

# Specify a policy to evaluate; length of pi = number of states

def policy_eval(policy, threshold = .001):
    nums, R, P, cfvs = Model()
    V = np.zeros(nums)
    while True:
        delta = 0
        for s in range(len(policy)):
            v = V[s]
            V[s] = np.sum(P[:, s, policy[s]] @ (R[s, policy[s]] + gam * V))
            delta = max(delta, abs(v-V[s]))
        if delta < threshold:
            return V
    return None

if __name__=="__main__":
    pi = np.random.randint(2, size = 10)
    print("Use policy: {}".format(pi))
    V = policy_eval(pi)
    print("Value function: {}".format(V))