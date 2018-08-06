import matplotlib.pyplot as plt
import numpy as np

k = 10

# Ideally, an infinite number of time steps
timesteps = 1000
runs = 2000

# for visualization purpose...
Rs = np.zeros((runs, timesteps))

for eps in (0., 0.1):

    # learning phase
    for run in range(runs):

        # Generate a problem. defining (randomly) a mean value for each channel
        means = np.random.normal(loc=0., scale=1., size=k)

        # Initializations
        Q = np.zeros(k)  # Value function
        if eps == 0.: Q += 5
        N = np.zeros(k)  # number of times an action a has been used

        for i in range(timesteps):

            # select an action
            if np.random.rand() <= 1-eps:
                a = np.argmax(Q)
            else:
                a = np.random.randint(0, k)

            # calculating the reward
            R = np.random.normal(loc=means[a], scale=1.)

            # Update memory
            N[a] += 1
            Q[a] += 0.1 * (R - Q[a])

            Rs[run, i] = R


    if eps == 0.:
        method_used = "optimistic greedy"
    else:
        method_used = "greedy-eps "+str(eps)

    plt.plot(range(timesteps), np.mean(Rs, axis=0), label=method_used)

plt.xlabel("Time steps")
plt.ylabel("Reward average")
plt.legend()
plt.show()

