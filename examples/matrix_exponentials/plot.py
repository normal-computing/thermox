import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", type=str)
args = parser.parse_args()


matrix_type = args.save_dir.split("/")[-1].split("_")[1].split(".")[0]


# use latex for plots
plt.rc("text", usetex=True)
# set font
plt.rc("font", family="serif")
# set font size
plt.rcParams.update({"font.size": 10})

colors = [
    plt.cm.viridis(0.2),
    plt.cm.viridis(0.4),
    plt.cm.viridis(0.6),
    plt.cm.viridis(0.8),
]


results = pickle.load(open(args.save_dir, "rb"))
dt = results["dt"]
NT = results["ERR_abs"].shape[-1]
D = results["D"]
e0_abs = 8.0 if matrix_type == "wishart" else 19.0
ylabel_abs = (
    r"$|| \bar{C} - \exp(-A)||_F$"
    if matrix_type == "wishart"
    else r"$|| \bar{C} - \exp(-M)||_F$"
)
e0_rel = 0.9
ylabel_rel = (
    r"$\frac{|| \bar{C} - \exp(-A)||_F}{||\exp(-A)||_F}$"
    if matrix_type == "wishart"
    else r"$\frac{|| \bar{C} - \exp(-M)||_F}{||\exp(-M)||_F}$"
)
fig_label = "(A)" if matrix_type == "wishart" else "(B)"


def plot(ERR, ylabel, e0, save_path, d=False, d_squared=False, fig_label=None):
    T = np.arange(NT) * dt
    ERR_mean = ERR.mean(axis=0)

    # find time where error crosses threshold
    TC = np.zeros(len(D))
    for i in range(len(D)):
        TC[i] = np.min(T[10:][ERR_mean[i, 10:] < e0])

    plt.figure(figsize=(7, 4.5))

    if fig_label is not None:
        plt.gcf().text(0.02, 0.93, fig_label, fontsize=22)

    for i in range(len(D)):
        plt.plot(T, ERR_mean[i], color=colors[i])

    # Add error bars
    for i in range(len(D)):
        plt.fill_between(
            T,
            ERR_mean[i] - ERR[:, i].std(axis=0),
            ERR_mean[i] + ERR[:, i].std(axis=0),
            color=colors[i],
            alpha=0.3,
            zorder=0,
        )

    plt.loglog()
    plt.legend(["d = " + str(D[i]) for i in range(len(D))], loc="upper right")
    plt.xlabel(r"Time ($\mu$s)", fontsize=18)
    plt.ylabel(ylabel, fontsize=18)

    # show threshold as horizontal line
    plt.axhline(e0, color="k", linestyle="--")
    # show crossing times as vertical lines
    for i in range(len(D)):
        plt.axvline(TC[i], color=colors[i], linestyle="--")

    plt.xlim(30, T[-1])

    # inset plot showing crossing time as a function of dimension
    ax = plt.axes([0.16, 0.22, 0.3, 0.35])
    ax.tick_params(axis="y", direction="in", pad=-22)
    ax.tick_params(axis="x", direction="in", pad=-15)

    for i in range(len(D)):
        ax.scatter(D[i], TC[i], color=colors[i], zorder=10)

    ts = np.array([10, 2000])

    if d:
        plt.plot(ts, 100 * ts, color="black", linestyle="--")
        plt.text(600, 8e4, s=r"$t_C = d$", rotation=25)

    if d_squared:
        plt.plot(ts, 0.3 * ts**2, color="black", linestyle="--")
        plt.text(550, 1.7e5, s=r"$t_C = d^2$", rotation=25)

    plt.plot(D, TC, color="black", zorder=0)
    plt.xlim(20, 1500)

    plt.loglog()
    plt.xlabel(r"$d$", fontsize=15)
    plt.ylabel(r"$t_C$", fontsize=15)
    plt.minorticks_off()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


plot(
    results["ERR_abs"],
    ylabel_abs,
    e0_abs,
    f"examples/matrix_exponentials/{matrix_type}_abs.pdf",
    d_squared=True,
    fig_label=fig_label,
)


plot(
    results["ERR_rel"],
    ylabel_rel,
    e0_rel,
    f"examples/matrix_exponentials/{matrix_type}_rel.pdf",
    d=True,
    fig_label=fig_label,
)
