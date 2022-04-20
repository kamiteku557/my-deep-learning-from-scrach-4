import numpy as np
import matplotlib.pyplot as plt
from icecream import ic
import ch1


if __name__ == "__main__":
    np.random.seed(0)

    steps = 1000
    epsilon = 0.1

    bandit = ch1.Bandit()
    agent = ch1.Agent(epsilon=epsilon)
    total_reward = 0
    total_rewards = []
    rates = []

    for step in range(1, steps + 1):
        action = agent.get_action()
        reward = bandit.play(action)
        agent.update(action, reward)
        total_reward += reward

        total_rewards.append(total_reward)
        rates.append(total_reward / step)

    ic(total_reward)

    plt.ylabel("Total reward")
    plt.xlabel("Steps")
    plt.plot(total_rewards)
    plt.savefig("fig1.jpg")
    plt.clf()

    plt.ylabel("Rates")
    plt.xlabel("Steps")
    plt.plot(rates)
    plt.savefig("fig2.jpg")
    plt.clf()
