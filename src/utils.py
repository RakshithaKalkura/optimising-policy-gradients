import matplotlib.pyplot as plt

def save_training_plot(rewards, filename):
    plt.figure()
    plt.plot(rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Learning Curve")
    plt.savefig(filename)
    plt.close()
