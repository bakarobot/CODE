import numpy as np
# Defined Model
# all models have two possible actions
gam = 0.9 # Discount Factor

def SimpleGridWorld():
    # action: 0: up, 1: down, 2: left, 3: right
    numa = 4
    height = 6
    width = 8
    nums = height*width # number of states
    
    # transition probabilities
    # P[new_state, current_state, action]
    P = np.zeros((nums, nums, numa))
    
    # action = 0: up
    for i in range(height-1):
        for j in range(width):
            P[(i+1)*width + j, i*width + j, 0] = 1.0
    for j in range(width):
        P[(height-1)*width + j, (height-1)*width + j, 0] = 1.0

    # action = 1: down
    for i in range(1, height):
        for j in range(width):
            P[(i-1)*width + j, (i)*width + j, 1] = 1.0
    for j in range(width):
        P[j, j, 1] = 1.0

    # action = 2: left
    for i in range(1, width):
        for j in range(height):
            P[width*j + i-1, width*j + i, 2] = 1.0
    for j in range(height):
        P[width*j, width*j, 2] = 1.0

    # action = 3: right
    for i in range(width-1):
        for j in range(height):
            P[width*j + i+1, width*j + i, 3] = 1.0
    for j in range(height):
        P[width*j + width-1, width*j + width-1, 3] = 1.0

    # Rewards
    r = 0.1
    R = r * np.ones((nums, numa))
    R[nums-1, 1] += 1.0

    return nums, R, P, None