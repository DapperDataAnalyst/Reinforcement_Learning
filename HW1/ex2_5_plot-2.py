import sys
from matplotlib import pyplot as plt
from HW1.bandit import main


results = main()
sample_average = {
    'average_rs': results[0],
    'average_best_action_taken': results[1],
}
constant = {
    'average_rs': results[2],
    'average_best_action_taken': results[3],
}

assert len(sample_average['average_rs']) == len(sample_average['average_best_action_taken']) == \
    len(constant['average_rs']) == len(constant['average_best_action_taken']) == 10000

fig,axes = plt.subplots(2,1)

axes[1].set_ylim([0.,1.])

axes[0].plot(sample_average['average_rs'], label="sample average")
axes[0].plot(constant['average_rs'], label="constant step-size")
axes[0].set_ylabel("reward")
axes[0].legend()

axes[1].plot(sample_average['average_best_action_taken'], label="sample average")
axes[1].plot(constant['average_best_action_taken'], label="constant step-size")
axes[1].set_xlabel("# time steps")
axes[1].set_ylabel("best action taken")
axes[1].legend()

fig.show()

# if fig.show() doesn't work, use the following line to save the figure
# plt.savefig("example.png")

# this keeps the program alive until you press "enter"
_ = input()

