import pickle
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# import plotly.graph_objects as go
# import numpy as np
from datetime import datetime
import time
from envs.Environment import TDCVRP
import os

my_dir_loc = os.getcwd() + "/src"


def create_data_on_disk(
    graph_size, num_samples, is_save=True, filename=None, is_return=False, seed=1234
):
    """Generate validation dataset (with SEED) and save
    """

    disk_env = TDCVRP(graph_size, num_samples, seed=seed)
    depo, graphs, demand = disk_env.get_triple()

    if is_save:
        save_to_pickle(
            my_dir_loc + "/" + "output/Validation_dataset_{}.pkl".format(filename),
            (depo, graphs, demand),
        )

    if is_return:
        return (
            tf.data.Dataset.from_tensor_slices(
                (list(depo), list(graphs), list(demand))
            ),
            disk_env,
        )


def save_to_pickle(filename, item):
    """Save to pickle
    """
    with open(filename, "wb") as handle:
        pickle.dump(item, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_from_pickle(path, return_tf_data_set=True, num_samples=None):
    """Read dataset from file (pickle)
    """

    objects = []
    with (open(path, "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break
    objects = objects[0]
    if return_tf_data_set:
        depo, graphs, demand = objects
        if num_samples is not None:
            return tf.data.Dataset.from_tensor_slices(
                (list(depo), list(graphs), list(demand))
            ).take(num_samples)
        else:
            return tf.data.Dataset.from_tensor_slices(
                (list(depo), list(graphs), list(demand))
            )
    else:
        return objects


def generate_data_onfly(num_samples=10000, graph_size=20, seed=None):
    """Generate temp dataset in memory
    """

    fly_env = TDCVRP(
        graph_size, num_samples, seed=seed
    )  # TODO: add batch size to all env creations
    depo, graphs, demand = fly_env.get_triple()

    return (
        tf.data.Dataset.from_tensor_slices((list(depo), list(graphs), list(demand))),
        fly_env,
    )


def get_results(
    train_loss_results,
    train_cost_results,
    val_cost,
    save_results=True,
    filename=None,
    plots=True,
):
    my_dir = "output/"
    filename = filename
    epochs_num = len(train_loss_results)

    df_train = pd.DataFrame(
        data={
            "epochs": list(range(epochs_num)),
            "loss": train_loss_results,
            "cost": train_cost_results,
        }
    )
    df_test = pd.DataFrame(
        data={"epochs": list(range(epochs_num)), "val_сost": val_cost}
    )
    """
    if save_results:
        df_train.to_excel(
            "{}train_results_{}.xlsx".format(my_dir, filename), index=False
        )
        df_test.to_excel("{}test_results_{}.xlsx".format(my_dir, filename), index=False)
    """
    if plots:
        plt.figure(figsize=(15, 9))
        ax = sns.lineplot(
            x="epochs", y="loss", data=df_train, color="salmon", label="train loss"
        )
        ax2 = ax.twinx()
        sns.lineplot(
            x="epochs",
            y="cost",
            data=df_train,
            color="cornflowerblue",
            label="train cost",
            ax=ax2,
        )
        sns.lineplot(
            x="epochs", y="val_сost", data=df_test, palette="darkblue", label="val cost"
        ).set(ylabel="cost")
        ax.legend(loc=(0.75, 0.90), ncol=1)
        ax2.legend(loc=(0.75, 0.95), ncol=2)
        ax.grid(axis="x")
        ax2.grid(True)
        plt.savefig("{}learning_curve_plot_{}.jpg".format(my_dir, filename))
        plt.show()


def get_cur_time():
    """Returns local time as string
    """
    ts = time.time()
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def get_clean_path(arr):
    """Returns extra zeros from path.
       Dynamical model generates duplicated zeros for several graphs when obtaining partial solutions.
    """

    p1, p2 = 0, 1
    output = []

    while p2 < len(arr):

        if arr[p1] != arr[p2]:
            output.append(arr[p1])
            if p2 == len(arr) - 1:
                output.append(arr[p2])

        p1 += 1
        p2 += 1

    if output[0] != 0:
        output.insert(0, 0.0)
    if output[-1] != 0:
        output.append(0.0)

    return output

