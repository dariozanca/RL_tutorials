import matplotlib.pyplot as plt
import numpy as np

###################################################################################################
# In this example we consider a non-stationary problem by sligthly changing means during the      #
# training.                                                                                       #
###################################################################################################

k = 10

# Ideally, an infinite number of time steps
timesteps = 10000
runs = 2000
change_each = 1

# for visualization purpose...
Rs = np.zeros((runs, timesteps))

eps = 0.1
alpha = 0.1
alpha_name  = ('Average', 'Exponential recency-weighted average')

for alpha_iter in range(2):

    # learning phase
    for run in range(runs):

        # Generate a problem. defining (randomly) a mean value for each channel
        means = np.random.normal(loc=0., scale=1., size=k)

        # Initializations
        Q = np.zeros(k)  # Value function
        N = np.zeros(k)  # number of times an action a has been used

        for i in range(timesteps):

            if i % change_each == 0:
                means += np.random.normal(loc=0., scale=.01, size=k)

            # select an action
            if np.random.rand() <= 1-eps:
                a = np.argmax(Q)
            else:
                a = np.random.randint(0, k)

            # calculating the reward
            R = np.random.normal(loc=means[a], scale=1.)

            # Update memory
            N[a] += 1

            if alpha_iter == 0:
                Q[a] += (1. / N[a]) * (R - Q[a])
            else:
                Q[a] += alpha * (R - Q[a])

            Rs[run, i] = R

    plt.plot(range(timesteps), np.mean(Rs, axis=0), label=alpha_name[alpha_iter])

plt.xlabel("Time steps")
plt.ylabel("Reward average")
plt.title("Comparison of greedy-epsilon 0.1 with 1/n or constant alpha")
plt.legend()
plt.show()

