from matplotlib import pyplot as plt


def plot_peaks(t_grid, oxide_peaks, baseline, reference):
    for i, oxide in enumerate(oxide_peaks):
        plt.plot(t_grid, oxide, label=f"oxide: {i}")
    plt.plot(t_grid, baseline, label="Baseline")
    plt.plot(t_grid, reference, label="Solution")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Excretion speed")
    plt.grid()
    plt.show()


def plot_prediction(t_grid, oxide_peaks, baseline, reference):
    for i, oxide in enumerate(oxide_peaks):
        plt.plot(t_grid, oxide, '.', label=f"oxide: {i}")
    plt.plot(t_grid, baseline, '.', label="Baseline")
    plt.plot(t_grid, reference, '.', label="Solution")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Excretion speed")
    plt.grid()
    plt.show()