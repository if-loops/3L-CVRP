from tqdm import tqdm
import pandas as pd
import numpy as np
import tensorflow as tf
import time
import datetime
from IPython.display import clear_output
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import ttest_rel
import copy

from attention_model import set_decode_type
from reinforce_baseline import validate
from envs.Environment import TDCVRP

from generate_data import (
    generate_data_onfly,
    get_results,
    get_cur_time,
    create_data_on_disk,
)
from time import gmtime, strftime
import os

my_dir_loc = os.getcwd() + "/src"


# tf.config.threading.set_intra_op_parallelism_threads(48)


def train_model(
    optimizer,
    model_tf,
    baseline,
    validation_dataset,
    samples=1280000,
    batch=128,
    val_batch_size=1000,
    start_epoch=0,
    end_epoch=10000,
    from_checkpoint=False,
    grad_norm_clipping=1.0,
    batch_save=100,
    graph_size=15,
    filename=None,
    hwl=None,
    seed_modifier=1,
):

    if filename is None:
        filename = "NO_FILENAME_GIVEN_{}_{}".format(
            graph_size, strftime("%Y-%m-%d", gmtime())
        )
    # legacy
    def rein_loss(model, env, baseline):
        """Calculate loss for REINFORCE algorithm
        """
        if bl_vals is None:
            env2 = copy.deepcopy(env)

        # Evaluate model, get costs and log probabilities
        set_decode_type(model, "sampling")
        print("Current decode for training is type: {}".format(model.decode_type))
        cost, log_likelihood, env_carry = model(env, usage=True)
        cost = tf.stop_gradient(cost)

        # Evaluate baseline
        # For first wp_n_epochs we take the combination of baseline and ema for previous batches
        bl_val = bl_vals if bl_vals is not None else baseline.eval(env2, cost)
        bl_val = tf.stop_gradient(bl_val)

        # Calculate loss
        reinforce_loss = tf.reduce_mean((cost - bl_val) / 100 * log_likelihood)

        env_carry.bl_val = bl_val
        env_carry.cost = cost

        return (
            reinforce_loss,
            tf.reduce_mean(cost),
            tf.reduce_mean(bl_val),
            tf.reduce_mean(cost - bl_val),
            cost,
            env_carry,  # use to carry variables for logging
        )

    def grad(model, env, baseline):
        """Calculate gradients
        """
        with tf.GradientTape() as tape:
            loss, cost, baseline_cost, delta_cost, raw_cost, env_carry = rein_loss(
                model, env, baseline
            )

        return (
            loss,
            cost,  # mean
            tape.gradient(loss, model.trainable_variables),
            baseline_cost,  # mean
            delta_cost,  # mean
            raw_cost,
            env_carry,
        )

    # For plotting
    train_loss_results = []
    train_cost_results = []
    val_cost_avg = []

    # Create and save validation dataset
    data, original_val_env = create_data_on_disk(
        num_samples=val_batch_size, graph_size=graph_size, is_return=True
    )

    # "shortest distance to next" heuristic for comparison
    sp_env = copy.deepcopy(original_val_env)
    # sp_env.max_trucks = sp_env.max_trucks * 10  # turn in into near pure vrp
    # sp_env.state[:, 1:, 3:6] = 1  # shrink packages
    sp_cost = 0
    while sp_env.all_finished() is False:
        sp_dist = sp_env.state[:, :, 14] + (
            sp_env.get_mask()[:, 0, :] * 9999
        )  # weight constraint
        sp_dist[:, 0] = 999  # discourage depot trip
        sp_dists = sp_dist  #  np.concatenate((sp_dist, sp_dist), axis=1)
        sp_acts = np.zeros([sp_env.state.shape[0], sp_env.state.shape[1]])
        for batch in range(sp_env.state.shape[0]):

            sp_acts[batch] = [
                x
                for _, x in sorted(
                    zip(sp_dists[batch], np.arange(sp_env.state.shape[1]),),
                    reverse=False,
                )
            ]

        sp_env.step(sp_acts)
        sp_cost = sp_env.get_distance()

    del sp_env

    # Prevent overwriting of file
    if from_checkpoint:
        filename = "Resumed_" + filename

    # Tensorboard logging
    train_log_dir = my_dir_loc + "/" + "logs/EULER/" + filename
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    # Training loop
    for epoch in range(start_epoch, end_epoch + 1):
        env_done = False

        original_fly_env = TDCVRP(
            graph_size, batch, gendreau=None, seed=int(epoch * seed_modifier), hwl=hwl
        )
        bl_env = copy.deepcopy(original_fly_env)

        if epoch == 0:  # expensive to compute

            rand_env = copy.deepcopy(original_fly_env)
            while rand_env.all_finished() is False:
                rand_act = random_agent_actions(rand_env)
                rand_env.step(rand_act)
                rand_cost = rand_env.get_costs()

            del rand_env

        t = time.time()
        while env_done is False:
            print("\n ########## \n Starting Epoch {}".format(epoch))

            env_done = original_fly_env.all_finished()

            epoch_loss_avg = tf.keras.metrics.Mean()
            epoch_cost_avg = tf.keras.metrics.Mean()

            # Skip warm-up stage when we continue training from checkpoint
            if from_checkpoint and baseline.alpha != 1.0:
                print("Skipping warm-up mode")
                baseline.alpha = 1.0

            bl_vals = baseline.eval_all(bl_env)
            # (samples, ) or None

            bl_vals = (
                tf.reshape(bl_vals, (-1, batch)) if bl_vals is not None else None
            )  # (n_batches, batch) or None

            # -------- PPO --------
            if True:
                # run for history
                set_decode_type(model_tf, "sampling")
                print(
                    "Current decode for training is type: {}".format(
                        model_tf.decode_type
                    )
                )
                cost, log_likelihood, env_carry = model_tf(original_fly_env, usage=True)
                old_log_probs = env_carry.actions_logits_history_tf
                rewards_history = (
                    env_carry.rewards
                )  # tf.tile(env_carry.rewards, multiples=[n_steps])

                n_steps = len(env_carry.actions_history)
                cost_val = tf.reduce_mean(rewards_history)

                if bl_vals is None:
                    env2 = copy.deepcopy(env_carry)
                    env2.reset()
                bl_val = bl_vals if bl_vals is not None else baseline.eval(env2, cost)

                rewards_history = rewards_history - bl_val  # calc advantage

                rewards_history = tf.reshape(rewards_history, [-1])
                rewards_history = tf.tile(rewards_history, multiples=[n_steps])
                rewards_history = -tf.cast(rewards_history, dtype=tf.float32)

                rewards_history = (
                    rewards_history - tf.math.reduce_mean(rewards_history)
                ) / (
                    tf.math.reduce_std(rewards_history) + 1e-10
                )  # normalization

                for i in range(5):  # "minibatches"

                    with tf.GradientTape() as actor_tape:

                        actor_loss = tf.constant(0.0)
                        actions_probs = list()

                        log_probs = model_tf(env_carry, ppo_info=True)
                        log_probs = tf.gather_nd(
                            log_probs,
                            tf.cast(env_carry.actions_history_tf[:, None], tf.int32),
                            batch_dims=1,
                        )  # select chosen action log_ps

                        ratio = tf.math.exp(log_probs - old_log_probs)
                        probability = tf.math.exp(log_probs)
                        entropy = -tf.reduce_mean(
                            tf.math.multiply(probability, tf.math.log(probability))
                        )
                        # print("entropy", entropy)

                        pg_losses = tf.constant(-1.0) * ratio * rewards_history
                        pg_losses2 = (
                            tf.constant(-1.0)
                            * tf.clip_by_value(ratio, 1 - 0.2, 1 + 0.2)
                            * rewards_history
                        )

                        pg_loss = tf.math.maximum(pg_losses, pg_losses2)
                        # dirty fix for outliers
                        pg_loss = tf.math.minimum(pg_loss, pg_loss * 0 + 0.1)

                        actor_loss = tf.math.reduce_mean(pg_loss) - 0.0001 * entropy

                    actor_grads = actor_tape.gradient(
                        actor_loss, model_tf.trainable_weights
                    )
                    # print("al",actor_loss,"ag", actor_grads,"par", model_tf.trainable_weights)
                    optimizer.apply_gradients(
                        zip(actor_grads, model_tf.trainable_variables)
                    )

                loss_value = actor_loss

                baseline_cost = tf.reduce_mean(bl_val)
                delta_cost = 0
                raw_cost = cost
            else:
                # old vanilla policy gradient
                # Optimize the model
                (
                    loss_value,
                    cost_val,
                    grads,
                    baseline_cost,
                    delta_cost,
                    raw_cost,
                    env_carry,
                ) = grad(model_tf, original_fly_env, baseline)

                optimizer.apply_gradients(zip(grads, model_tf.trainable_variables))

            # Track progress
            epoch_loss_avg.update_state(loss_value)
            epoch_cost_avg.update_state(cost_val)

            # Save model weights
            if epoch % batch_save == 0:
                model_tf.save_weights(
                    my_dir_loc
                    + "/"
                    + "checkpoints/{}_model_checkpoint_epoch_{}.h5".format(
                        filename, epoch
                    ),
                    save_format="h5",
                )

            train_loss_results.append(epoch_loss_avg.result())
            train_cost_results.append(epoch_cost_avg.result())

            delta_t = np.round((time.time() - t), 1)

            delta_epoch = end_epoch - epoch
            remaining = np.round(delta_t * delta_epoch / 60 / 60, 2)  # in hours
            round_loss = np.round(epoch_loss_avg.result(), 1)
            round_cost = np.round(epoch_cost_avg.result(), 1)

            env_done = original_fly_env.all_finished()
            # Tensorboard logging
            vars = model_tf.trainable_variables
            if env_done:
                with train_summary_writer.as_default():

                    # Cost
                    tf.summary.scalar("cost_avg", epoch_cost_avg.result(), step=epoch)
                    tf.summary.scalar(
                        "distance_avg", (np.mean(env_carry.get_distance())), step=epoch,
                    )

                    tf.summary.histogram("cost", raw_cost, step=epoch)
                    tf.summary.scalar("std_cost", np.std(raw_cost), step=epoch)
                    missed = (
                        np.sum(env_carry.state[:, 1:, 16] == 0)
                        / env_carry.state.shape[0]
                    )
                    tf.summary.scalar("missed_avg", missed, step=epoch)

                    # Baseline
                    tf.summary.scalar(
                        "baseline_cost", np.mean(baseline_cost), step=epoch
                    )
                    tf.summary.scalar(
                        "advantage_avg",
                        epoch_cost_avg.result() - np.mean(baseline_cost),
                        step=epoch,
                    )

                    # Policy
                    tf.summary.histogram(
                        "Policy .log_p (without -inf since masked)",
                        env_carry.log_p,
                        step=epoch,
                    )

                    # Time
                    tf.summary.scalar("Time remaining", remaining, step=epoch)

                    # Loss
                    tf.summary.scalar("loss_avg", epoch_loss_avg.result(), step=epoch)

                    # Random baseline
                    if epoch == 0:
                        tf.summary.scalar("random cost", np.mean(rand_cost), step=epoch)
                        tf.summary.scalar(
                            "Model minus Random",
                            np.mean(raw_cost) - np.mean(rand_cost),
                            step=epoch,
                        )

                    # Time
                    tf.summary.scalar("Time for Epoch", delta_t, step=epoch)
                    # tf.summary.histogram(
                    #    "Loops for all steps", env_carry.avg_act_iters, step=epoch
                    # )
                    tf.summary.scalar(
                        "env_stepping_time", env_carry.env_stepping_time, step=epoch
                    )
                    tf.summary.scalar("model_time", env_carry.model_time, step=epoch)
                    tf.summary.scalar(
                        "relative env time",
                        env_carry.env_stepping_time / env_carry.model_time,
                        step=epoch,
                    )

                    if False:  # takes a lot of space to save
                        for var, gra in zip(vars, grads):
                            tf.summary.histogram(var.name, var[:], step=epoch)
                            tf.summary.histogram(
                                "Unclipped grad " + var.name, gra[:], step=epoch
                            )

                    if epoch % 100 == 0:
                        # Compare against a benchmark
                        print("Benchmark")
                        spec_name = "E016-03m"
                        set_decode_type(model_tf, "greedy")
                        spec_env = TDCVRP(1, 1, gendreau=1)
                        spec_cost = 0
                        while not spec_env.all_finished():
                            spec_cost, _ = model_tf(spec_env)

                        spec_cost = np.mean(spec_env.get_distance())
                        missed = np.sum(spec_env.state[0, 1:, 16] == 0)
                        tf.summary.scalar(
                            spec_name + " Distance", spec_cost, step=epoch
                        )
                        tf.summary.scalar(spec_name + " Missed", missed, step=epoch)

                        del spec_env

                        # set_decode_type(model_tf, "sampling")

                        spec_name = "E016-05m"
                        set_decode_type(model_tf, "greedy")
                        spec_env = TDCVRP(1, 1, gendreau=2)
                        spec_cost = 0
                        while not spec_env.all_finished():
                            spec_cost, _ = model_tf(spec_env)

                        spec_cost = np.mean(spec_env.get_distance())
                        missed = np.sum(spec_env.state[0, 1:, 16] == 0)
                        tf.summary.scalar(
                            spec_name + " Distance", spec_cost, step=epoch
                        )
                        tf.summary.scalar(spec_name + " Missed", missed, step=epoch)

                        del spec_env
                        set_decode_type(model_tf, "sampling")

                    if epoch % 50 == 0:
                        # Validate current model (not so often as expensive to compute)
                        val_env = copy.deepcopy(original_val_env)
                        val_cost = 0
                        while not val_env.all_finished():
                            val_cost = validate(
                                validation_dataset, model_tf, val_env, val_batch_size
                            )
                        set_decode_type(model_tf, "sampling")
                        val_cost_avg.append(val_cost)
                        tf.summary.scalar("val_cost_avg", val_cost, step=epoch)
                        tf.summary.scalar(
                            "val_SP_heuristic_avg", np.mean(sp_cost), step=epoch
                        )
                        tf.summary.scalar(
                            "val_model_minus_SP_heuristic_avg",
                            np.mean(val_env.get_distance()) - np.mean(sp_cost),
                            step=epoch,
                        )
                        missed = (
                            np.sum(val_env.state[:, 1:, 16] == 0)
                            / val_env.state.shape[0]
                        )
                        tf.summary.scalar("val_model_missed_avg", missed, step=epoch)

        print(
            get_cur_time(),
            "Epoch {}: Loss: {}: Cost: {} Time: {}sec, Remaining: {}h".format(
                epoch, round_loss, round_cost, delta_t, remaining,
            ),
        )

        if epoch % 100 == 0:
            baseline.epoch_callback(model_tf, epoch)
            set_decode_type(model_tf, "sampling")

    return model_tf


def random_agent_actions(my_env):
    av_act = np.arange(my_env.get_mask().shape[2])
    av_act[0] = 1  # dirty fix to discourage depot trips
    np.random.shuffle(av_act)
    av_act[-1] = 0
    rand_step = my_env.get_mask() * 0 + av_act

    return rand_step[:, 0, :]

