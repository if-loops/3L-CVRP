import tensorflow as tf
from scipy.stats import ttest_rel
from tqdm import tqdm
import numpy as np

from attention_model import AttentionModel
from attention_model import set_decode_type
from generate_data import generate_data_onfly
from envs.Environment import TDCVRP
import os
import copy

my_dir_loc = os.getcwd() + "/src"


def copy_of_tf_model(model, embedding_dim=128, graph_size=20):
    """Copy model weights to new model
    """
    # https://stackoverflow.com/questions/56841736/how-to-copy-a-network-in-tensorflow-2-0
    copy_env = TDCVRP(graph_size, 1)
    depo, graphs, demand = copy_env.get_triple()
    data_random = [depo, graphs, demand]

    new_model = AttentionModel(embedding_dim)
    set_decode_type(new_model, "sampling")
    _, _ = new_model(copy_env)

    for a, b in zip(new_model.variables, model.variables):
        a.assign(b)

    return new_model


def rollout(model, dataset, rollout_env, batch_size=1000, disable_tqdm=False):
    # TODO: remove the unused dataset variable
    # Evaluate model in greedy mode
    set_decode_type(model, "greedy")
    costs_list = []
    # TODO: Batch size change
    print("Current decode for baseline is type: {}".format(model.decode_type))

    while not rollout_env.all_finished():
        cost, _ = model(rollout_env)
    costs_list.append(cost)
    # print("COST LIST", costs_list)

    return tf.concat(costs_list, axis=0)


def validate(dataset, model, env, batch_size=1000):
    """Validates model on given dataset in greedy mode
    """
    val_costs = rollout(model, dataset, env, batch_size=batch_size)
    set_decode_type(model, "sampling")  # setting back to sampling
    mean_cost = tf.reduce_mean(val_costs)
    # print(f"Validation score: {np.round(mean_cost, 4)}")
    return mean_cost


class RolloutBaseline:
    def __init__(
        self,
        model,
        filename,
        from_checkpoint=False,
        path_to_checkpoint=None,
        wp_n_epochs=1,
        epoch=0,
        num_samples=10000,
        warmup_exp_beta=0.8,
        embedding_dim=128,
        graph_size=20,
    ):
        """
        Args:
            model: current model
            filename: suffix for baseline checkpoint filename
            from_checkpoint: start from checkpoint flag
            path_to_checkpoint: path to baseline model weights
            wp_n_epochs: number of warm-up epochs
            epoch: current epoch number
            num_samples: number of samples to be generated for baseline dataset
            warmup_exp_beta: warmup mixing parameter (exp. moving average parameter)
        """

        self.num_samples = num_samples
        self.cur_epoch = epoch
        self.wp_n_epochs = wp_n_epochs
        self.beta = warmup_exp_beta

        # controls the amount of warmup
        self.alpha = 0.0

        self.running_average_cost = None

        # Checkpoint params
        self.filename = filename
        self.from_checkpoint = from_checkpoint
        self.path_to_checkpoint = path_to_checkpoint

        # Problem params
        self.embedding_dim = embedding_dim
        self.graph_size = graph_size

        # create and evaluate initial baseline
        self._update_baseline(model, epoch)

    def _update_baseline(self, model, epoch):

        # Load or copy baseline model based on self.from_checkpoint condition
        if self.from_checkpoint and self.alpha == 0:
            print("Baseline model loaded")
            self.model = load_tf_model(
                self.path_to_checkpoint,
                embedding_dim=self.embedding_dim,
                graph_size=self.graph_size,
            )
        else:
            self.model = copy.deepcopy(model)  # copy_of_tf_model(
            #    model, embedding_dim=self.embedding_dim, graph_size=self.graph_size
            # )

            # For checkpoint
            self.model.save_weights(
                my_dir_loc + "/" + "checkpoints/baseline_checkpoint.h5",
                save_format="h5",
            )

        # We generate a new dataset for baseline model on each baseline update to prevent possible overfitting
        self.dataset, self.env = generate_data_onfly(
            num_samples=self.num_samples, graph_size=self.graph_size, seed=epoch * 42
        )

        print(f"Evaluating baseline model on baseline dataset (epoch = {epoch})")
        self.bl_vals = rollout(self.model, self.dataset, self.env)
        self.mean = tf.reduce_mean(self.bl_vals)
        self.cur_epoch = epoch

        # print("Baseline: ", self.mean)

    def ema_eval(self, cost):
        """This is running average of cost through previous batches (only for warm-up epochs)
        """

        if self.running_average_cost is None:
            self.running_average_cost = tf.reduce_mean(cost)
        else:

            self.running_average_cost = self.beta * self.running_average_cost + (
                1.0 - self.beta
            ) * tf.reduce_mean(cost)

            # Workaround
            # self.running_average_cost = tf.reduce_mean(cost)

        return self.running_average_cost

    def eval(self, batch, cost):
        """Evaluates current baseline model on single training batch
        """

        if self.alpha == 0:
            return self.ema_eval(cost)

        if self.alpha < 1:
            v_ema = self.ema_eval(cost)
        else:
            v_ema = 0.0

        # batch.reset()
        print(
            "Current decode for warmup baseline is type: {}".format(
                self.model.decode_type
            )
        )
        v_b, _ = self.model(batch)

        v_b = tf.stop_gradient(v_b)
        v_ema = tf.stop_gradient(v_ema)

        # Combination of baseline cost and exp. moving average cost
        return self.alpha * v_b + (1 - self.alpha) * v_ema

    def eval_all(self, train_env):
        """Evaluates current baseline model on the whole dataset only for non warm-up epochs
        """

        if self.alpha < 1:
            return None

        val_costs = rollout(
            self.model, self.dataset, train_env, batch_size=self.num_samples
        )

        return val_costs

    def epoch_callback(self, model, epoch):
        """Compares current baseline model with the training model and updates baseline if it is improved
        """

        self.cur_epoch = epoch
        self.env.reset()
        print(
            f"Evaluating candidate model on baseline dataset (callback epoch = {self.cur_epoch})"
        )
        candidate_vals = rollout(
            model, self.dataset, self.env
        )  # costs for training model on baseline dataset
        candidate_mean = tf.reduce_mean(candidate_vals)

        diff = candidate_mean - self.mean

        print(
            f"Epoch {self.cur_epoch} candidate mean {candidate_mean}, baseline epoch {self.cur_epoch} mean {self.mean}, difference {diff}"
        )
        """
        # Saving one run by using current epoch data
         # extra args to pass in: cost_val, baseline_cost, raw_cost, bl_vals
        self.cur_epoch = epoch
        diff = cost_val - baseline_cost
        candidate_vals = raw_cost
        self.bl_vals = bl_vals
        # in loop self.bl_vals = tf.reshape(self.bl_vals, (-1))
        """

        if diff < 0:

            # statistic + p-value
            t, p = ttest_rel(candidate_vals, self.bl_vals)

            p_val = p / 2
            print(f"p-value: {p_val}")

            if p_val < 0.05:
                print("Update baseline")
                self._update_baseline(model, self.cur_epoch)

        # alpha controls the amount of warmup
        if self.alpha < 1.0:
            self.alpha = (self.cur_epoch + 1) / float(self.wp_n_epochs)
            print(f"alpha was updated to {self.alpha}")

    def rollout_full(self, rollout_env):
        model = self.model
        set_decode_type(model, "greedy")

        while not rollout_env.all_finished():
            my_cost, _ = self.model(rollout_env)

        set_decode_type(model, "sampling")

        return np.mean(my_cost)


def load_tf_model(path, embedding_dim=128, graph_size=20, n_encode_layers=3):
    """Load model weights from hd5 file
    """
    # https://stackoverflow.com/questions/51806852/cant-save-custom-subclassed-model
    loader_env = TDCVRP(graph_size, 1)

    model_loaded = AttentionModel(embedding_dim, n_encode_layers=n_encode_layers)
    set_decode_type(model_loaded, "greedy")
    _, _ = model_loaded(loader_env)

    model_loaded.load_weights(path)

    return model_loaded
