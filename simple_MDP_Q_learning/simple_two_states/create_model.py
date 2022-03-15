import numpy as np
# Defined Model
# all models have two possible actions
gam = 0.9 # Discount Factor

def Model():
    numa = 2  # number of actions
    nums = 10 # number states
    # transition probabilities
    P = np.zeros((nums, nums, numa))
    # action = 0
    P[0, 0, 0] = 1.0
    for i in range(1,nums):
        P[i-1, i, 0] = 1.0
    # action = 1
    P[nums-1, nums-1, 1] = 1.0
    for i in range(0,nums-1):
        P[i+1,i,1] = 1.0
    # Rewards
    r = 0.1
    R = r* np.ones((nums, numa))
    R[nums-1,1] += 1.0
    cfvs = np.zeros(nums)
    cfvs[nums-1] = (1+r)/(1-gam)
    for kk in range(0,nums-1):
        cfvs[kk] = (r + gam**(nums-kk-1))/(1-gam)

    return nums, R, P, cfvs

nums, R, P, cfvs = Model()
np.set_printoptions(formatter={'float': lambda x: "{0:0.1f}".format(x)})
print('MODEL')
print('P0:\n',P[:,:,0])
print('P1:\n',P[:,:,1])
print('Rewards:\n', R)

nums, R, P, cfvs = Model()
print('Model clV* =', cfvs)