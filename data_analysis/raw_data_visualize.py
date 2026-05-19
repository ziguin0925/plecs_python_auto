import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import math

# CSV 로드


def data_distribution_plot(df, name_list, plot_type="box"):

    n = len(name_list)

    cols = 3
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))

    # subplot이 1개일 때 대비
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, name in enumerate(name_list):

        if plot_type == "box":
            sns.boxplot(x=df[name], ax=axes[i])

        elif plot_type == "strip":
            sns.stripplot(x=df[name], jitter=True, size=3, ax=axes[i])

        elif plot_type == "swarm":
            sns.swarmplot(x=df[name], size=3, ax=axes[i])

        axes[i].set_title(name)

    # 남는 subplot 제거
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def seaborn_pairplot(df, x_list, y_list):
    """
    x_lsit = ["L1", "L2", "C1", "fs"]
    y_list = ["L1_ESR_loss", "L2_ESR_loss", "C1_ESR_loss", "C2_ESR_loss"]
    """
    fig = sns.pairplot(
        df,
        x_vars=x_list,
        y_vars=y_list,
        aspect=1.5,
    )
    plt.show()


def missinig_value_plot(df, parameter_dict):
    # -----------------------------
    # parameter spec
    # -----------------------------
    param_specs = {
        "L1": {"min": 220e-6, "max": 1700e-6, "count": 10},
        "L2": {"min": 25e-6, "max": 205e-6, "count": 10},
        "C1": {"min": 1e-6, "max": 25e-6, "count": 10},
        "fs": {"min": 50e3, "max": 100e3, "count": 10, "round": -2},
    }

    parameter_dict = {
        key: (
            np.round(
                np.linspace(spec["min"], spec["max"], spec["count"]),
                spec.get("round", 6),
            ).tolist()
        )
        for key, spec in param_specs.items()
    }

    # -----------------------------
    # CSV load
    # -----------------------------
    csv_path = "/home/pcsl/Documents/plecs/sepic/plecs_python_auto/out/sepic_data/inner_loop/not_exist_parameters.csv"

    df = pd.read_csv(csv_path)

    # -----------------------------
    # plot
    # -----------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()

    for ax, (param_name, x_values) in zip(axes, parameter_dict.items()):

        counts = df[param_name].value_counts().reindex(x_values, fill_value=0)

        bars = ax.bar(range(len(x_values)), counts.values)

        # 숫자 표시
        ax.bar_label(
            bars, labels=[str(v) for v in counts.values], padding=3, fontsize=9
        )

        ax.set_title(f"{param_name} Distribution")
        ax.set_xlabel(param_name)
        ax.set_ylabel("Count")

        ax.set_xticks(range(len(x_values)))
        ax.set_xticklabels([f"{v:.3g}" for v in x_values], rotation=45)

        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()
