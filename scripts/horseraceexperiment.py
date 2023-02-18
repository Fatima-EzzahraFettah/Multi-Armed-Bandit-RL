from operator import itemgetter
from scripts.algorithms.distribution import Gaussian, Bernoulli
from scripts.algorithms.epsilon_greedy import EpsilonGreedy
from scripts.algorithms.ucb1 import UCB1
from scripts.algorithms.ts_gaussian import ThompsonSamplingGaussian
from scripts.algorithms.ts_bernoulli import ThompsonSamplingBernoulli
from scripts.algorithms.gradient_bandits import GradientBandits
from numpy import arange
import matplotlib.pyplot as plt


def plot_gaussian_algorithms() -> None:
    means_vars_data = [
        (4., 6.),
        (4.2, 10.),
        (4.5, 8.),
        (5., 9.),
        (5.5, 4.),
        (6, 10.),
        (6., 15.)]
    mu_star = max(means_vars_data, key=itemgetter(0))[0]

    steps = 500
    episodes = 500

    eps = 0.3
    eps_hl = 400

    ci = 5
    mi = mu_star * 3.

    ts_mi = 0.
    ts_si = 10.

    lr = 0.1
    lr_decay = 20.
    ucb_alpha = 4.0
    arm_distrs = [Gaussian(μ=m, σ=s) for m, s in means_vars_data]

    greedy_opt_init = EpsilonGreedy(
        arm_distributions=arm_distrs,
        time_steps=steps,
        num_episodes=episodes,
        epsilon=0.,
        epsilon_half_life=1e8,
        count_init=ci,
        mean_init=mi
    )
    eps_greedy = EpsilonGreedy(
        arm_distributions=arm_distrs,
        time_steps=steps,
        num_episodes=episodes,
        epsilon=eps,
        epsilon_half_life=1e8,
        count_init=0,
        mean_init=0.
    )
    decay_eps_greedy = EpsilonGreedy(
        arm_distributions=arm_distrs,
        time_steps=steps,
        num_episodes=episodes,
        epsilon=eps,
        epsilon_half_life=eps_hl,
        count_init=0,
        mean_init=0.
    )
    ts = ThompsonSamplingGaussian(
        arm_distributions=arm_distrs,
        time_steps=steps,
        num_episodes=episodes,
        init_mean=ts_mi,
        init_stdev=ts_si
    )
    grad_bandits = GradientBandits(
        arm_distributions=arm_distrs,
        time_steps=steps,
        num_episodes=episodes,
        learning_rate=lr,
        learning_rate_decay=lr_decay
    )
    ucb1 = UCB1(
        arm_distributions=arm_distrs,
        time_steps=steps,
        num_episodes=episodes,
        bounds_range=1.0,
        alpha=ucb_alpha
    )

    plot_colors = ['r', 'b', 'g', 'k', 'y','m']
    labels = [
        'Greedy, Optimistic Initialization',
        '$\epsilon$-Greedy',
        'Decaying $\epsilon$-Greedy',
        'UCB',
        'Thompson Sampling',
        'Gradient Bandit'
    ]

    exp_cum_regrets = [
        greedy_opt_init.get_expected_cum_regret(mu_star),
        eps_greedy.get_expected_cum_regret(mu_star),
        decay_eps_greedy.get_expected_cum_regret(mu_star),
        ucb1.get_expected_cum_regret(mu_star),
        ts.get_expected_cum_regret(mu_star),
        grad_bandits.get_expected_cum_regret(mu_star)
    ]

    x_vals = range(1, steps + 1)
    for i in range(len(exp_cum_regrets)):
        plt.plot(x_vals, exp_cum_regrets[i], plot_colors[i], label=labels[i])
    plt.xlabel("Time Steps", fontsize=10)
    plt.ylabel("Expected Total Regret", fontsize=10)
    plt.title("Total Regret Curves - Use Case 3 ", fontsize=15)
    plt.xlim(xmin=x_vals[0], xmax=x_vals[-1])
    plt.ylim(ymin=0.0)
    plt.grid(True)
    plt.legend(loc='upper left', fontsize=10)
    plt.show()

    exp_act_counts = [
        greedy_opt_init.get_expected_action_counts(),
        eps_greedy.get_expected_action_counts(),
        decay_eps_greedy.get_expected_action_counts(),
        ts.get_expected_action_counts(),
        grad_bandits.get_expected_action_counts()
    ]
    index = arange(len(means_vars_data))
    spacing = 0.4
    width = (1 - spacing) / len(exp_act_counts)

    hist_plot_colors = ['r', 'b', 'g', 'k', 'y','m']
    for i in range(len(exp_act_counts)):
        plt.bar(
            index - (1 - spacing) / 2 + (i - 1.5) * width,
            exp_act_counts[i],
            width,
            color=hist_plot_colors[i],
            label=labels[i]
        )
    plt.xlabel("Arms", fontsize=10)
    plt.ylabel("Expected Counts of Arms", fontsize=10)
    plt.title("Arms Counts Plot - Use Case 3", fontsize=15)
    plt.xticks(
        index - 0.3,
        ["$\mu$=%.1f,$\sigma$=%.1f" % (m, s) for m, s in means_vars_data]
    )
    plt.legend(loc='upper left', fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_bernoulli_algorithms() -> None:
    probs_data = [0.1, 0.2, 0.2,  0.3, 0.4,  0.45, 0.6]
    mu_star = max(probs_data)

    steps = 500
    episodes = 500

    eps = 0.1
    eps_hl = 400

    ci = 5
    mi = mu_star * 3.

    ucb_alpha = 4.0

    lr = 0.5
    lr_decay = 20.

    arm_distrs = [Bernoulli(p) for p in probs_data]

    greedy_opt_init = EpsilonGreedy(
        arm_distributions=arm_distrs,
        time_steps=steps,
        num_episodes=episodes,
        epsilon=0.,
        epsilon_half_life=1e8,
        count_init=ci,
        mean_init=mi
    )
    eps_greedy = EpsilonGreedy(
        arm_distributions=arm_distrs,
        time_steps=steps,
        num_episodes=episodes,
        epsilon=eps,
        epsilon_half_life=1e8,
        count_init=0,
        mean_init=0.
    )
    decay_eps_greedy = EpsilonGreedy(
        arm_distributions=arm_distrs,
        time_steps=steps,
        num_episodes=episodes,
        epsilon=eps,
        epsilon_half_life=eps_hl,
        count_init=0,
        mean_init=0.
    )
    ucb1 = UCB1(
        arm_distributions=arm_distrs,
        time_steps=steps,
        num_episodes=episodes,
        bounds_range=1.0,
        alpha=ucb_alpha
    )
    ts = ThompsonSamplingBernoulli(
        arm_distributions=arm_distrs,
        time_steps=steps,
        num_episodes=episodes
    )
    grad_bandits = GradientBandits(
        arm_distributions=arm_distrs,
        time_steps=steps,
        num_episodes=episodes,
        learning_rate=lr,
        learning_rate_decay=lr_decay
    )

    plot_colors = ['r', 'b', 'g', 'k', 'y','m']
    labels = [
        'Greedy, Optimistic Initialization',
        '$\epsilon$-Greedy',
        'Decaying $\epsilon$-Greedy',
        'UCB1',
        'Thompson Sampling',
        'Gradient Bandit'
    ]

    exp_cum_regrets = [
        greedy_opt_init.get_expected_cum_regret(mu_star),
        eps_greedy.get_expected_cum_regret(mu_star),
        decay_eps_greedy.get_expected_cum_regret(mu_star),
        ucb1.get_expected_cum_regret(mu_star),
        ts.get_expected_cum_regret(mu_star),
        grad_bandits.get_expected_cum_regret(mu_star)
    ]

    x_vals = range(1, steps + 1)
    for i in range(len(exp_cum_regrets)):
        plt.plot(x_vals, exp_cum_regrets[i], plot_colors[i], label=labels[i])
    plt.xlabel("Time Steps", fontsize=10)
    plt.ylabel("Expected Total Regret", fontsize=10)
    plt.title("Total Regret Curves - Use Case 5", fontsize=15)
    plt.xlim(xmin=x_vals[0], xmax=x_vals[-1])
    plt.ylim(ymin=0.0)
    plt.grid(True)
    plt.legend(loc='upper left', fontsize=10)
    plt.show()

    exp_act_counts = [
        greedy_opt_init.get_expected_action_counts(),
        eps_greedy.get_expected_action_counts(),
        decay_eps_greedy.get_expected_action_counts(),
        ucb1.get_expected_action_counts(),
        ts.get_expected_action_counts(),
        grad_bandits.get_expected_action_counts()
    ]
    index = arange(len(probs_data))
    spacing = 0.4
    width = (1 - spacing) / len(exp_act_counts)

    hist_plot_colors = ['r', 'b', 'g', 'k', 'y','m']
    for i in range(len(exp_act_counts)):
        plt.bar(
            index - (1 - spacing) / 2 + (i - 1.5) * width,
            exp_act_counts[i],
            width,
            color=hist_plot_colors[i],
            label=labels[i]
        )
    plt.xlabel("Arms", fontsize=10)
    plt.ylabel("Expected Counts of Arms", fontsize=10)
    plt.title("Arms Counts Plot - Use Case 5", fontsize=15)
    plt.xticks(
        index - 0.2,
        ["$p$=%.2f" % p for p in probs_data]
    )
    plt.legend(loc='upper left', fontsize=10)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_gaussian_algorithms()
    #plot_bernoulli_algorithms()