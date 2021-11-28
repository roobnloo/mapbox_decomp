import numpy as np
import matplotlib.pyplot as plt
import configparser
from datetime import date, timedelta


def generate(fact_dir, rank, nfact, day_axis, figures_dir):
    time_factor = np.load(fact_dir + f"rank_{str(rank)}_factor_2.npy")
    # weights = np.load(fact_dir + f"rank_{str(rank)}_weights.npy")
    plot(day_axis, time_factor[:, 0:nfact], rank, nfact, figures_dir)


def get_day_axis():
    sdate = date(2020, 1, 1)
    days = []
    for i in range(365 + 1):
        day = sdate + timedelta(days=i)
        days.append(day)
    return days


def plot(day_axis, timefact, rank, nfact, fig_dir):
    plt.clf()
    plt.plot(day_axis, timefact)
    plt.axvline(x=date(2020, 3, 19), label="CA Stay-at-home Order", c="blue")
    plt.axvline(x=date(2020, 8, 18), label="SCU Lightning Complex fires", c="red")
    plt.axvline(x=date(2020, 10, 1), c="red")
    plt.legend()
    plt.title(f"First {nfact} factors of rank {rank} decomposition")
    fig = plt.gcf()
    fig.set_size_inches(12, 4)
    plt.savefig(fig_dir+f"timeplot_rank{rank}.png", format="png")


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('config.ini')
    dir_to_save_factors = config['PATHS']['dir_to_save_factors']
    dir_save_figures = config['PATHS']['path_to_save_figures']
    day_axis = np.array(get_day_axis())
    for r in range(5, 50, 3):
        generate(dir_to_save_factors, r, 10, day_axis, dir_save_figures)
