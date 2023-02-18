import numpy as np
import matplotlib.pyplot as plt

class ucb_bandit:
    '''
    Upper Confidence Bound k-bandit problem

    Inputs
    ============================================
    k: number of arms (int)
    c:
    iters: number of steps (int)
    mu: set the average rewards for each of the k-arms.
        Set to "random" for the rewards to be selected from
        a normal distribution with mean = 0.
        Set to "sequence" for the means to be ordered from
        0 to k-1.
        Pass a list or array of length = k for user-defined
        values.
    '''

    def __init__(self, k, c, iters, mu='random'):
        # Number of arms
        self.k = k
        # Exploration parameter
        self.c = c
        # Number of iterations
        self.iters = iters
        # Step count
        self.n = 1
        # Step count for each arm
        self.k_n = np.ones(k)
        # Total mean reward
        self.mean_reward = 0
        self.reward = np.zeros(iters)
        # Mean reward for each arm
        self.k_reward = np.zeros(k)

        if type(mu) == list or type(mu).__module__ == np.__name__:
            # User-defined averages
            self.mu = np.array(mu)
        elif mu == 'random':
            # Draw means from probability distribution
            self.mu = np.random.normal(0, 1, k)
        elif mu == 'sequence':
            # Increase the mean for each arm by one
            self.mu = np.linspace(0, k - 1, k)

    def pull(self):
        # Select action according to UCB Criteria
        a = np.argmax(self.k_reward + self.c * np.sqrt(
            (np.log(self.n)) / self.k_n))

        reward = np.random.normal(self.mu[a], 1)

        # Update counts
        self.n += 1
        self.k_n[a] += 1

        # Update total
        self.mean_reward = self.mean_reward + (
                reward - self.mean_reward) / self.n

        # Update results for a_k
        self.k_reward[a] = self.k_reward[a] + (
                reward - self.k_reward[a]) / self.k_n[a]

    def run(self):
        for i in range(self.iters):
            self.pull()
            self.reward[i] = self.mean_reward

    def reset(self, mu=None):
        # Resets results while keeping settings
        self.n = 1
        self.k_n = np.ones(self.k)
        self.mean_reward = 0
        self.reward = np.zeros(self.iters)
        self.k_reward = np.zeros(self.k)
        if mu == 'random':
            self.mu = np.random.normal(0, 1, self.k)


class grad_bandit:

    def __init__(self, k, alpha, iters, mu='random'):
        # Number of arms
        self.k = k
        self.actions = np.arange(k)
        # Number of iterations
        self.iters = iters
        # Step count
        self.n = 1
        # Step count for each arm
        self.k_n = np.ones(k)
        # Total mean reward
        self.mean_reward = 0
        self.reward = np.zeros(iters)
        # Mean reward for each arm
        self.k_reward = np.zeros(k)
        # Initialize preferences
        self.H = np.zeros(k)
        # Learning rate
        self.alpha = alpha

        if type(mu) == list or type(mu).__module__ == np.__name__:
            # User-defined averages
            self.mu = np.array(mu)
        elif mu == 'random':
            # Draw means from probability distribution
            self.mu = np.random.normal(0, 1, k)
        elif mu == 'sequence':
            # Increase the mean for each arm by one
            self.mu = np.linspace(0, k - 1, k)

    def softmax(self):
        self.prob_action = np.exp(self.H - np.max(self.H)) \
                           / np.sum(np.exp(self.H - np.max(self.H)), axis=0)

    def pull(self):
        # Update probabilities
        self.softmax()
        # Select highest preference action
        a = np.random.choice(self.actions, p=self.prob_action)

        reward = np.random.normal(self.mu[a], 1)

        # Update counts
        self.n += 1
        self.k_n[a] += 1

        # Update total
        self.mean_reward = self.mean_reward + (
                reward - self.mean_reward) / self.n

        # Update results for a_k
        self.k_reward[a] = self.k_reward[a] + (
                reward - self.k_reward[a]) / self.k_n[a]

        # Update preferences
        self.H[a] = self.H[a] + \
                    self.alpha * (reward - self.mean_reward) * (1 -
                                                                self.prob_action[a])
        actions_not_taken = self.actions != a
        self.H[actions_not_taken] = self.H[actions_not_taken] - \
                                    self.alpha * (reward - self.mean_reward) \
                                    * self.prob_action[actions_not_taken]

    def run(self):
        for i in range(self.iters):
            self.pull()
            self.reward[i] = self.mean_reward

    def reset(self, mu=None):
        # Resets results while keeping settings
        self.n = 0
        self.k_n = np.zeros(self.k)
        self.mean_reward = 0
        self.reward = np.zeros(iters)
        self.k_reward = np.zeros(self.k)
        self.H = np.zeros(self.k)
        if mu == 'random':
            self.mu = np.random.normal(0, 1, self.k)


if __name__ == '__main__':
    k = 10
    iters = 1000
    # Initialize bandits
    grad = grad_bandit(k, 0.1, iters, mu='random')
    ucb = ucb_bandit(k, 2, iters, mu=grad.mu)
    ucb.mu = grad.mu

    ucb_rewards = np.zeros(iters)
    grad_rewards = np.zeros(iters)
    opt_grad = 0
    opt_ucb = 0

    episodes = 1000
    # Run experiments
    for i in range(episodes):
        # Reset counts and rewards
        grad.reset('random')
        ucb.reset()
        ucb.mu = grad.mu

        # Run experiments
        grad.run()
        ucb.run()

        # Update long-term averages
        grad_rewards = grad_rewards + (
                grad.reward - grad_rewards) / (i + 1)
        ucb_rewards = ucb_rewards + (
                ucb.reward - ucb_rewards) / (i + 1)

        # Count optimal actions
        opt_grad += grad.k_n[np.argmax(grad.mu)]
        opt_ucb += ucb.k_n[np.argmax(ucb.mu)]

    plt.figure(figsize=(12, 8))
    plt.plot(grad_rewards, label="Gradient", color = 'red')
    plt.plot(ucb_rewards, label="UCB", color = 'blue')
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Average Reward")
    plt.title("Average Gradient Bandit Rewards after "
              + str(episodes) + " Episodes")
    plt.show()