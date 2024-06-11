import sys

# sys.path.append("../src/")
# sys.path.append("../src/envs/")

import numpy as np
import random
import time
import pandas as pd


from gym import Env
from gym.spaces import Discrete, Box


from itertools import repeat
from gendreau_opt import TDCVRP

from joblib import Parallel, delayed
from itertools import repeat

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

#%%prun # for profiling
np.random.seed(42)

# random agent testing
my_batch_size = 25

env = TDCVRP(
    n_destinations=100,
    packages_perc_of_av_vol=90,
    frag_freq_prob=25,
    test_set=1,
    folder="Test_folder",
    foldername="Test_foldername",
    batch_size=my_batch_size,
)

# env.pandas_state(0).head()

