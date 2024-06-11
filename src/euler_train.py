import tensorflow as tf
from time import gmtime, strftime
import sys

from attention_model import AttentionModel, set_decode_type
from reinforce_baseline import RolloutBaseline
from train import train_model
from reinforce_baseline import load_tf_model

# from generate_data import create_data_on_disk, get_cur_time

tf.config.threading.set_intra_op_parallelism_threads(0)  # us optimal num of threads

# Params of model
# ------------------

# -------
# Example Input: bsub -W 120:00 -n 4 python src/euler_train.py 5 100 0.0001 50000 0.5 "cnn" 6 5 12
# -------

run_descriptor = (
    "EULER_"
    + str(sys.argv[6])
    + "_hwl_"
    + str(sys.argv[7])
    + str(sys.argv[8])
    + str(sys.argv[9])
    + "_seed_"
    + str(sys.argv[10])
)
GRAPH_SIZE = int(sys.argv[1])

LEARNING_RATE = float(sys.argv[3])
START_EPOCH = 0
END_EPOCH = int(sys.argv[4])

hwl = [int(sys.argv[7]), int(sys.argv[8]), int(sys.argv[9])]
BATCH = int(sys.argv[2])
VAL_BATCH_SIZE = BATCH
ROLLOUT_SAMPLES = BATCH
BATCH_SAVE = 100  # save model every n epochs
seed_modifier = int(sys.argv[10])

# TODO: Vary load level and fragility share

# Only for resuming:
NEW_TRAINING = True  # set to false
FILENAME = "Debug_2021-06-22-18-25_graph4_batch5_lr0.001"

# ------------------

SAMPLES = BATCH
embedding_dim = 128
NUMBER_OF_WP_EPOCHS = 1
GRAD_NORM_CLIPPING = float(sys.argv[5])
VALIDATE_SET_SIZE = VAL_BATCH_SIZE
SEED = 1234


if NEW_TRAINING:
    print("NEW TRAINING")
    FILENAME = "{}_{}_graph{}_batch{}_lr{}".format(
        run_descriptor,
        strftime("%Y-%m-%d-%H-%M", gmtime()),
        GRAPH_SIZE,
        BATCH,
        LEARNING_RATE,
    )

    # Initialize model
    model_tf = AttentionModel(embedding_dim)
    set_decode_type(model_tf, "sampling")
    # print(get_cur_time(), 'model initialized')

    # Create and save validation dataset
    # TODO: currently unused as we generate via seed in train
    """
    validation_dataset = create_data_on_disk(GRAPH_SIZE,
                                             VALIDATE_SET_SIZE,
                                             is_save=True,
                                             filename=FILENAME,
                                             is_return=True,
                                             seed = SEED)
    """
    validation_dataset = 0
    # print(get_cur_time(), 'validation dataset created and saved on the disk')

    # Initialize optimizer
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=LEARNING_RATE, decay_steps=10000, decay_rate=0.9
    )
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr_schedule, clipnorm=GRAD_NORM_CLIPPING, epsilon=1e-8
    )

    # Initialize baseline
    baseline = RolloutBaseline(
        model_tf,
        wp_n_epochs=NUMBER_OF_WP_EPOCHS,
        epoch=0,
        num_samples=ROLLOUT_SAMPLES,
        filename=FILENAME,
        from_checkpoint=False,
        embedding_dim=embedding_dim,
        graph_size=GRAPH_SIZE,
    )
    # print(get_cur_time(), 'baseline initialized')

    model_t = train_model(
        optimizer,
        model_tf,
        baseline,
        validation_dataset,
        samples=SAMPLES,
        batch=BATCH,
        val_batch_size=VAL_BATCH_SIZE,
        start_epoch=START_EPOCH,
        end_epoch=END_EPOCH,
        from_checkpoint=False,
        grad_norm_clipping=GRAD_NORM_CLIPPING,
        batch_save=BATCH_SAVE,
        graph_size=GRAPH_SIZE,
        filename=FILENAME,
        hwl=hwl,
        seed_modifier=seed_modifier,
    )
else:
    print("RESUMING TRAINING")

    folder = "checkpoints/"
    MODEL_PATH = folder + "model_checkpoint.h5"
    BASELINE_MODEL_PATH = folder + "baseline_checkpoint.h5"
    # VAL_SET_PATH = 'Not needed'

    # Initialize model
    model_tf = load_tf_model(
        MODEL_PATH, embedding_dim=embedding_dim, graph_size=GRAPH_SIZE
    )
    set_decode_type(model_tf, "sampling")
    # print(get_cur_time(), 'model loaded')

    # Create and save validation dataset
    validation_dataset = 0  # read_from_pickle(VAL_SET_PATH)
    # print(get_cur_time(), 'validation dataset loaded')

    # Initialize optimizer
    optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

    # Initialize baseline
    baseline = RolloutBaseline(
        model_tf,
        wp_n_epochs=NUMBER_OF_WP_EPOCHS,
        epoch=START_EPOCH,
        num_samples=ROLLOUT_SAMPLES,
        filename=FILENAME,
        from_checkpoint=True,
        embedding_dim=embedding_dim,
        graph_size=GRAPH_SIZE,
        path_to_checkpoint=BASELINE_MODEL_PATH,
    )
    print(get_cur_time(), "baseline initialized")

    train_model(
        optimizer,
        model_tf,
        baseline,
        validation_dataset,
        samples=SAMPLES,
        batch=BATCH,
        val_batch_size=VAL_BATCH_SIZE,
        start_epoch=START_EPOCH,
        end_epoch=END_EPOCH,
        from_checkpoint=True,
        grad_norm_clipping=GRAD_NORM_CLIPPING,
        batch_save=BATCH_SAVE,
        graph_size=GRAPH_SIZE,
        filename=FILENAME,
    )

