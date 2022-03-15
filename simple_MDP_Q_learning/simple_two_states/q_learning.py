from create_model import Model
import numpy as np
import matplotlib.pyplot as plt

# Q-learning
gam = 0.9

def generative_q_learning(steps=1000, alpha=0.9):
    nums, R, P, cfvs = Model()
    numa = P.shape[2]
    Q = np.zeros((nums, numa))
    max_Q_array = []
    max_Q_per_state = []
    for i in range(steps):
        state = np.random.randint(nums, size = 1)
        action = np.argmax(Q[state, :])
        next_state = np.random.choice(range(nums), p = P[:, state, action].flatten())
        Q[state, action] = (1-alpha)*Q[state, action] + alpha*(R[state, action] + gam*max(Q[next_state, :]))
        max_Q_array.append(np.max(Q))
        max_Q_per_state.append(max(np.mean(Q, axis=1)))
    return Q, max_Q_array, max_Q_per_state

# Q-learning w/ exploration

## YOUR CODE BELOW
##---------------------------------------------------------------
def exploration_q_learning(steps=1000, alpha=0.9, eps=0.8):
    nums, R, P, cfvs = Model()
    numa = P.shape[2]
    Q = np.zeros((nums, numa))
    max_Q_array = []
    max_Q_per_state = []
    for i in range(steps):
        state = np.random.randint(nums, size = 1)
        if np.random.choice([True, False], p=[1-eps, eps]):
            action = np.argmax(Q[state, :])
        else:
            action = np.random.randint(numa)
        next_state = np.random.choice(range(nums), p = P[:, state, action].flatten())
        Q[state, action] = (1-alpha)*Q[state, action] + alpha*(R[state, action] + gam*max(Q[next_state, :]))
        max_Q_array.append(np.max(Q))
        max_Q_per_state.append(max(np.mean(Q, axis=1)))
    return Q, max_Q_array, max_Q_per_state

fig, ax = plt.subplots(1, 2, sharex = True)
fig.set_size_inches((15, 10))
# generative Q learning for each model

training_result = {}

Q, max_q_array, max_q_state = exploration_q_learning()
training_result["generative-max"] = max_q_array
training_result["generative-avg"] = max_q_state
print("Q_model:\n", Q)
ax[0].plot(max_q_array)
ax[0].set_title("Policy performance wrt iteration, max Q")
ax[0].set_ylabel("Max Q value")
ax[1].plot(max_q_state)
ax[1].set_title("Policy performance wrt iteration, average Q per state")
ax[0].set_xlabel("Iteration")
ax[1].set_xlabel("Iteration")
fig.tight_layout()
plt.show()