"""
Base code taken from https://github.com/ifm-mag/01_vrp/blob/master/src/vrp/envs/vrp_env.py
"""

import os
import copy
import time
import glob
from typing import Tuple, List, Dict, Union
import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import gym  # type: ignore
from gym import spaces  # type: ignore
from itertools import repeat
from joblib import Parallel, delayed
import pprofile
from numba import jit, prange
import functools
from skimage.transform import resize

import warnings

warnings.filterwarnings("ignore")


# import vrp.utils.sm_functions as sm
import matplotlib
from matplotlib.ticker import MultipleLocator  # type: ignore


class TDCVRP(gym.Env):
    """This class replicates the CVRP Environment from (Nazari et. al 2018).
        https://arxiv.org/abs/1802.04240
    Attributes:
        n_destinations (int): An integer indicating number of destinations
    """

    def __init__(
        self,
        n_destinations: int,
        batch_size: int,
        packages_perc_of_av_vol: int = 50,
        frag_freq_prob: int = 25,
        test_set: int = 1,
        folder: str = "Train",
        foldername: str = "Debug",
        gendreau=None,
        seed=None,
        # hwl=[30, 25, 60],
        hwl=[6, 5, 12],
    ):
        """Inits VRPEnv with n_destinations.
        ASSUMES THERE IS ONLY ONE DEPOT.
        """

        super(TDCVRP, self).__init__()

        self.seed = seed

        # rand_h = np.random.randint(6, 8)
        # rand_w = np.random.randint(5, 7)
        # rand_l = np.random.randint(12, 14)
        # hwl = [rand_h, rand_w, rand_l]

        # gendreau = np.random.randint(2) + 1  # TODO: remove

        # Check if we are running a predefined instance
        if gendreau is None:
            self.gendreau = None
        else:
            assert gendreau in (
                np.arange(27) + 1
            ), "Only instance numbers 1 to 27 are valid. You passed {}. Pass no value or 0 for random instances".format(
                gendreau
            )
            self.gendreau = gendreau

        self.folder = folder
        self.foldername = foldername
        self.n_destinations = n_destinations
        self.packages_perc_of_av_vol = packages_perc_of_av_vol
        self.frag_freq_prob = frag_freq_prob
        self.hwl = hwl
        self.step_size = 0

        self.test_set = test_set
        self.folder = folder
        self.foldername = foldername
        self.batch_size = batch_size
        self.homogeneous = True  # Trucks are all homogeneous i.e. the same.
        self.prev_a_delayed = None
        self.prev_a = None

        # Assetion checks
        assert 0 < min(
            n_destinations, packages_perc_of_av_vol
        ), "Input must be greater 0"
        assert (
            100 >= frag_freq_prob >= 0
        ), "Fragility frequency must be between 0 and 100"

        self.n_container = np.zeros(self.batch_size)  # current container being loaded
        self.count = 0
        self.counter = 0
        self.rewards = np.zeros(self.batch_size)

        self.travel_cost = np.zeros(self.batch_size)
        self.avg_act_iters = np.zeros(self.batch_size)
        self.env_stepping_time = 0
        self.model_time = 0
        self._generate_env()

    def _generate_env(self) -> None:
        if self.folder:

            # homogeneous tuck containers

            self.state_columns = [
                "dest",
                "x",
                "y",
                "h",
                "w",
                "l",
                "frag",
                "pos_h",  # 7 ind
                "pos_w",
                "pos_l",
                "pos_h_r",
                "pos_w_r",
                "pos_l_r",
                "rot",  # 0 for no rotation
                "d_prev",  # 14 ind
                "n_veh",
                "seq_nr",
                "l_skip",  # 17 ind
                "l_skip_r",
                "feas",  # 19 ind
                "feas_r",
                "feas_same",
                "depot_mask",  # 22 ind
                "mask_n",
                "mask_r",
                "demand",
                "cum_demand_dest",
                "feas_weight",
            ]
            self.info = {}
            self.starting = 1

            if self.seed is not None:
                np.random.seed(self.seed)
                print("Generating Env with seed ", self.seed)

            # TODO: remove; just for testing
            gend_flip = False
            # if self.gendreau is None:
            #    gend_flip = True
            #    self.gendreau = 1

            # Predefined instance from Gendreau et al.
            if self.gendreau is not None:

                # based on Gendreau
                self.max_xy = 100
                self.vehicle_h = 30
                self.vehicle_w = 25
                self.vehicle_l = 60
                self.truck_capacity = 90
                self.capacity = (
                    np.ones(self.batch_size) * self.truck_capacity * 100
                )  # scaling for int table

                # get Gendreau instance
                my_items, my_depot, my_vehicles, demand = self.get_instance(
                    self.gendreau
                )

                # Vehicle parameters
                self.max_trucks = (np.ones(self.batch_size) * my_vehicles[0]).astype(
                    int
                )
                self.vehicle_h = my_vehicles[1]
                self.vehicle_w = my_vehicles[2]
                self.vehicle_l = my_vehicles[3]

                # Fill state
                self.state = np.zeros(
                    ((self.batch_size, my_items.shape[0] + 1, len(self.state_columns)))
                )
                # Depot
                self.state[:, 0, 0:3] = my_depot
                self.state[:, 1:, 0] = my_items[:, 0]  # IDs (Node)
                self.state[:, 1:, 1:3] = my_items[:, 6:8]  # locations (xy)
                self.state[:, 1:, 3:7] = my_items[
                    :, 2:6
                ]  # package dimensions and fragility

                # Demand assignement
                for batch in range(self.batch_size):
                    cur_dest, packs = np.unique(
                        self.state[batch, :, 0], return_counts=True
                    )
                    cum_dem = demand

                    for row in range(self.state.shape[1]):
                        my_dest = int(self.state[batch, row, 0])

                        self.state[batch, row, 25] = (
                            float(cum_dem[my_dest]) / float(packs[my_dest]) * 100
                        )  # scale for integer table
                        self.state[batch, row, 26] = float(cum_dem[my_dest])

                self.n_destinations = np.max(my_items[:, 0])

                if gend_flip:  #  shuffle xy locations
                    original = np.arange(101)

                    for batch in range(self.state.shape[0]):
                        for x_orig in np.unique(self.state[batch, :, 1]):
                            new_x = np.random.choice(original, 1, replace=False)[0]
                            self.state[batch, :, 1] = np.where(
                                self.state[batch, :, 1] == x_orig,
                                new_x,
                                self.state[batch, :, 1],
                            )

                        for y_orig in np.unique(self.state[batch, :, 2]):
                            new_y = np.random.choice(original, 1, replace=False)[0]

                            self.state[batch, :, 2] = np.where(
                                self.state[batch, :, 2] == y_orig,
                                new_y,
                                self.state[batch, :, 2],
                            )
                    # self.max_trucks = self.max_trucks * 1.2

                if False:
                    self.state[:, :, 3:6] = np.round(self.state[:, :, 3:6] / 5, 0) * 5

                self.state = self.state.astype(int)

            else:
                # randomly generated instance
                self.truck_capacity = 90
                self.capacity = (
                    np.ones(self.batch_size) * self.truck_capacity * 100
                )  # scaling for int table
                self.max_xy = 100

                # if self.hwl == None:
                #    self.hwl = [6, 5, 12]  # [30, 25, 60]  # [6,5,12] #[30,25,60]

                self.vehicle_h = self.hwl[0]
                self.vehicle_w = self.hwl[1]
                self.vehicle_l = self.hwl[2]

                self.generate_random_load()

                if False:  # for speedup
                    self.state[:, :, 3:6] = np.round(self.state[:, :, 3:6] / 5, 0) * 5

                self.state = self.state.astype(int)
                # Depot location(s) removal of package info
                self.state[:, 0, 3:17] = 0

                # Demand assignement
                cum_dem_weight = np.random.randint(1, 30, self.n_destinations + 1)
                cum_dem_weight[0] = 0  # depot
                # create instances that are more focused on packing than weight
                cum_dem_packing = np.random.randint(1, 10, self.n_destinations + 1)
                cum_dem_packing[0] = 0  # depot

                for batch in range(self.batch_size):
                    cur_dest, packs = np.unique(
                        self.state[batch, :, 0], return_counts=True
                    )

                    if (
                        True
                    ):  # batch % 2 == 0:  # mix weight and routing focused instances
                        cum_dem = cum_dem_weight
                    else:
                        cum_dem = cum_dem_packing

                    for row in range(self.state.shape[1]):
                        my_dest = int(self.state[batch, row, 0])
                        self.state[batch, row, 25] = (
                            float(cum_dem[my_dest]) / float(packs[my_dest]) * 100
                        )  # scale for integer table
                        self.state[batch, row, 26] = float(cum_dem[my_dest])

                # Decide on number of trucks needed based on volume of packages
                all_pack_vol = np.sum(
                    self.state[:, 1:, 3] * self.state[:, 1:, 4] * self.state[:, 1:, 5],
                    axis=1,
                )

                # TODO: also consider weight
                cum_dem = cum_dem_weight
                req_trucks_demand = np.ceil(
                    (np.sum(cum_dem) / self.truck_capacity) * 2
                ).astype(
                    int
                )  # add safety factor

                single_truck = self.vehicle_h * self.vehicle_w * self.vehicle_l
                reduced_single_truck = single_truck * self.packages_perc_of_av_vol / 100
                req_trucks = np.ceil(all_pack_vol / reduced_single_truck).astype(int)

                self.max_trucks = np.maximum(req_trucks, req_trucks_demand)

            # Create empty truck containers needed
            self.containers = np.zeros(
                (
                    self.batch_size,
                    np.max(self.max_trucks),
                    self.vehicle_h,
                    self.vehicle_w,
                    self.vehicle_l,
                )
            )

            self.VEHICLE_CAPACITY = (
                np.ones([self.batch_size, 1])
                * self.vehicle_h
                * self.vehicle_w
                * self.vehicle_l
            )

            # everything available at depot in the beginning
            self.state[:, :, 19:25] = 1

            self.done = np.full((self.batch_size, 1), False, dtype=bool)
            self.info["n_items"] = {self.state.shape[1]}

            # Keep track of positions (max is latest)
            self.position = np.zeros(shape=(self.batch_size, self.n_destinations + 1))

            # keep track of load
            self.cur_capacity = np.zeros(self.batch_size)

            # Set depot as start
            self.position[:, 0] = 1

            # Create Mask
            self.mask = np.ones(shape=(self.batch_size, self.state.shape[1], 2))

            # Step once with action depot as start
            self.trajectory: List[
                int
            ] = []  # The agent always starts at the depot (i.e. 0)
            self.xy_trajectory: List[int] = []
            self.action_list: List[int] = []
            step_init = np.zeros([self.batch_size, self.batch_size * 2])
            self.helper = 1
            _, _, _, _ = self.step(step_init)
            self.helper = 0

            self.action_space = gym.spaces.Tuple(
                (spaces.Discrete(self.state.shape[1]), spaces.Discrete(2),)
            )

            self.reward: Union[float, None] = np.full((self.batch_size, 1), None)
            self.info: Dict = {}

            self.counter = 0

            # Save state for validation
            if self.folder == "validation":
                np.save(
                    os.path.join(self.foldername, f"test_set_{self.test_set}"),
                    self.state,
                )

            # info
            self.info["max_trucks"] = self.max_trucks
            # self.info["items_per_dest"] = items_per_dest
            self.info["state_columns"] = self.state_columns

            # TODO: add assert checks

        else:
            raise ValueError(
                f"Seed: {self.seed} and Folder: {self.folder} is not True. Check that env_instance is not 0."
            )

    # @jit(cache=True)
    def generate_random_load(self):
        # Create master array of inventory to ship
        items_per_dest = np.random.randint(1, 4, self.n_destinations + 1)
        items_per_dest[0] = 1  # depot

        self.state = np.zeros(
            ((self.batch_size, items_per_dest.sum(), len(self.state_columns)))
        )

        ind = 0

        # collect all destinations seperately for easier access
        self.destinations = np.zeros((self.batch_size, self.n_destinations + 1, 2))

        for dest in range(self.n_destinations + 1):
            # create random values shared by all packages of the same destination

            # coordinates
            dest_xy = np.random.randint(0, self.max_xy, (self.batch_size, 2))

            self.destinations[:, dest, :] = dest_xy

            for package_i in range(items_per_dest[dest]):
                # shared information
                self.state[:, ind, 0] = dest
                self.state[:, ind, 1:3] = dest_xy

                if True:
                    # Gendreau uses 0.2 and 0.6
                    instance_low = 0.2
                    instance_high = 0.6

                    # create random values per unique package
                    self.state[:, ind, 3] = np.random.randint(
                        low=int(np.ceil(self.vehicle_h * instance_low)),
                        high=int(np.ceil(self.vehicle_h * instance_high)),
                        size=(self.batch_size),
                    )
                    self.state[:, ind, 4] = np.random.randint(
                        low=int(np.ceil(self.vehicle_w * instance_low)),
                        high=int(np.ceil(self.vehicle_w * instance_high)),
                        size=(self.batch_size),
                    )
                    self.state[:, ind, 5] = np.random.randint(
                        low=int(np.ceil(self.vehicle_l * instance_low)),
                        high=int(np.ceil(self.vehicle_l * instance_high)),
                        size=(self.batch_size),
                    )
                else:
                    # create random values per unique package
                    my_step = np.random.randint(5, 8)  # SET THIS ACCORDINGLY
                    h_choice = (
                        np.arange(self.hwl[0] / (my_step + 1)) * my_step + my_step
                    )
                    w_choice = (
                        np.arange(self.hwl[1] / (my_step + 1)) * my_step + my_step
                    )
                    l_choice = (
                        np.arange(self.hwl[2] / (my_step + 1)) * my_step + my_step
                    )
                    h_choice = h_choice[h_choice >= self.hwl[0] * 0.2]
                    w_choice = w_choice[w_choice >= self.hwl[1] * 0.2]
                    l_choice = l_choice[l_choice >= self.hwl[2] * 0.2]
                    h_choice = h_choice[h_choice <= self.hwl[0] * 0.6]
                    w_choice = w_choice[w_choice <= self.hwl[1] * 0.6]
                    l_choice = l_choice[l_choice <= self.hwl[2] * 0.6]

                    h_choice = [3]
                    self.state[:, ind, 3] = np.random.choice(h_choice, self.batch_size)
                    w_choice = [2]
                    self.state[:, ind, 4] = np.random.choice(w_choice, self.batch_size)
                    l_choice = [2]
                    self.state[:, ind, 5] = np.random.choice(l_choice, self.batch_size)

                # Fragility
                p = [1 - self.frag_freq_prob / 100, self.frag_freq_prob / 100]
                self.state[:, ind, 6] = np.random.choice(2, (self.batch_size), p=p)

                ind += 1

    # @jit(cache=True)
    def step(self, action_tuple) -> Tuple[np.ndarray, float, bool, np.ndarray]:
        """Takes a step using `action`.
        IMPORTANT: Remember that row 0 is the depot. Therefore, make sure the
        mask is set accordingly. The mask essentially denotes where the agent is,
        as well as which packages have already been loaded.
        Args:
            action (int):
                States which pavkage to pack next (can also be the depot).
                Therefore, action=0 goes to depot. Action > 0 packs a package and goes tothe corresponding location.
                Second dimension decides about rotation 0-> no rot, 1-> rot
        Returns:
            next_state (np.ndarray):
                A numpy array denoting the next state.
            done (bool):
                Whether the agent has reached the terminal state (remember to
                give 0 reward to anchor Q-values).
            reward (float):
                Calculates the reward received for this given step.
            mask (np.ndarray):
                A mask that can be applied to the agent's NN logits to determine
                feasible actions.
        """
        self.prev_a_delayed = self.prev_a

        step_size_try = 25
        while self.step_size == 0:
            if (self.state[:, :, 3:6] % step_size_try).any():
                step_size_try -= 1
            else:
                self.step_size = step_size_try

        state_copy = copy.deepcopy(self.state)
        cont_copy = copy.deepcopy(self.containers)

        # determine what and how to pack
        num_actions_1d = self.state.shape[1]
        batch_rot = (action_tuple >= num_actions_1d) * 1
        batch_row = action_tuple - (num_actions_1d) * batch_rot

        self.row_chosen = np.zeros(self.batch_size)
        self.rot_chosen = np.zeros(self.batch_size)
        self.act_chosen = np.zeros(self.batch_size)

        # prof = pprofile.Profile()
        # with prof():

        # my_jobs = 4
        # my_batch_size = max(1, int(self.batch_size / my_jobs))

        """  # parallel or not
            self.par_info = Parallel(
                n_jobs=-1,
                backend="threading",
                verbose=0,
                pre_dispatch="all",
                batch_size=10,
            )(
                delayed(self.batch_pipeline)(
                    state_copy[batch_num],
                    cont_copy[batch_num],
                    int(batch_row[batch_num]),
                    int(batch_rot[batch_num]),
                    self.clean_action_list[batch_num],
                )
                for batch_num in range(self.batch_size)
            )

            # prof.print_stats()

            # update self (Parallel can't write to self; thus manually here)
            self.state = np.array([row[0] for row in self.par_info])
            self.rewards = np.array([row[1] for row in self.par_info])
            self.containers = np.array([row[2] for row in self.par_info])
            self.done = np.array([row[3] for row in self.par_info])
            self.travel_cost = np.array([row[4] for row in self.par_info])
            self.mask = self.state[:, :, [23, 24]]
        """
        self.batch_pipeline(
            state_copy, cont_copy, batch_row, batch_rot,  # self.clean_action_list,
        )

        batch_row = self.row_chosen
        action_tuple = self.act_chosen

        # Update info
        self.trajectory.append(
            self.state[np.arange(self.state.shape[0]), batch_row.astype(int), 0]
        )
        self.xy_trajectory.append(
            self.state[np.arange(self.state.shape[0]), batch_row.astype(int), 1:3]
        )

        self.clean_trajectory = np.array([np.array(xi) for xi in self.trajectory]).T
        self.clean_xy_trajectory = np.array([np.array(xi) for xi in self.xy_trajectory])

        if True:
            for batch in range(self.batch_size):
                if any(
                    self.state[batch, 1:, 23]
                ):  # TODO: add 24 back in after adding rotation
                    self.mask[batch, 0, :] = 0
                    self.state[batch, 0, [23, 24]] = 0
                else:
                    self.mask[batch, 0, :] = 1
                    self.state[batch, 0, [23, 24]] = 1

            # no depot trip when starting
            if self.starting == 1:
                self.state[:, 0, [23, 24]] = 0
                self.starting = 0
                self.mask[:, 0, :] = 0
        else:
            self.state[:, 0, [23, 24]] = 1
            self.mask[:, 0, :] = 1

        self.info["add trivia"] = "here"

        self.prev_a = np.argmax(self.state[:, :, 16], axis=1)
        curr_veh = np.max(self.state[:, :, 15], axis=1)
        # self.free_capacity = np.sum(
        #    np.reshape(
        #        (self.containers[:, curr_veh - 1]) ** 2, (self.batch_size, -1)
        #    ),  # squared == abs
        #    axis=1,
        # )[
        #    :, None
        # ]  # in current vehicle

        self.avg_act_iters = self.avg_act_iters / self.batch_size

        return self.state, self.rewards, self.done, self.info

    def last(self):
        return self.state, self.rewards, self.done, self.info

    # @jit(cache=True)
    def get_triple(self):

        # unfold the rotations
        normal_copy = copy.deepcopy(self.state)
        # rot_copy = copy.deepcopy(self.state)

        # old code from having both rotation states as input
        # rot_copy = np.delete(
        #    rot_copy,
        #    [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
        #    axis=2,
        # )
        normal_copy = np.delete(
            normal_copy,
            [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 27],
            axis=2,
        )

        # unfolded_state = np.concatenate(
        #    (normal_copy, rot_copy), axis=1
        # )  # remove initial depot entry

        # graphs = unfolded_state[:, 1:, 1:3]

        # reduced to single
        # graphs = normal_copy[:, 1:, 1:3]
        unfolded_state = normal_copy

        # scale to make different sizes possible for model
        depot = self.state[:, 0, 1:3] / self.max_xy
        unfolded_state = unfolded_state * 1.0  # int to double
        unfolded_state[:, :, 1] = unfolded_state[:, :, 1] / self.max_xy
        unfolded_state[:, :, 2] = unfolded_state[:, :, 2] / self.max_xy
        unfolded_state[:, 1:, 3] = unfolded_state[:, 1:, 3] / self.vehicle_h
        unfolded_state[:, 1:, 4] = unfolded_state[:, 1:, 4] / self.vehicle_w
        unfolded_state[:, 1:, 5] = unfolded_state[:, 1:, 5] / self.vehicle_l
        unfolded_state[:, 1:, 7] = unfolded_state[:, 1:, 7] / (
            self.truck_capacity * 100
        )
        unfolded_state[:, 1:, 8] = unfolded_state[:, 1:, 8] / self.truck_capacity

        graphs = unfolded_state[:, 1:, 1:3]

        return depot, graphs, unfolded_state[:, 1:, 3:]

    def get_context(self):
        """
        Returns the step context for the decoder (current loading situation)
        """

        # current truck
        curr_truck = np.max(self.state[:, :, 15], axis=1)
        curr_truck = np.maximum(
            curr_truck, np.ones(self.batch_size)
        )  # for initial iteration
        curr_containers = self.containers[
            np.arange(self.batch_size).astype(int), curr_truck.astype(int) - 1
        ]  # .reshape((state.batch_size, -1))

        flat_trucks = np.sum(abs(curr_containers), axis=1) * 1.0
        flat_trucks = flat_trucks / self.vehicle_h  # scale heights

        # scale the trucks
        trucks_resized = resize(
            flat_trucks, (flat_trucks.shape[0], 25, 60)
        )  # .reshape((self.batch_size, -1))

        return trucks_resized[:, :, :, None]

    def get_capacity(self):
        scaled_cap = self.cur_capacity / (self.truck_capacity * 100)  # scaling
        return scaled_cap.reshape((self.batch_size, -1))

    def get_mask(self):

        # capacity constraint
        delta = (
            self.cur_capacity[:, None] - self.state[:, :, 26] * 100
        )  # for all loation packages at once

        for batch in range(self.batch_size):
            if self.state[
                batch, :, 21
            ].all():  # check if we are currently at a location with packages remaining
                delta[batch] = delta[batch]
            else:
                delta[batch] = (
                    delta[batch] * 0 + 1
                )  # only feasibility constrain matters (this constraint is tricky, since we it spans multiple packages, thus not in state table)

        my_state = self.state[:, :, 22] * self.state[:, :, 21] * (delta > 0)

        # my_state[:, 0] = 1  # always feasible
        unfolded_mask = my_state * (-1) + 1  # flip for masking

        # for batch in range(unfolded_mask.shape[0]):
        #    if (unfolded_mask[batch, 1:]).all():
        #        unfolded_mask[:, 0] = 0
        #    else:
        #        unfolded_mask[:, 0] = 1 # no depot trips -> force delivery
        unfolded_mask[:, 0] = 0

        # for batch in range(unfolded_mask.shape[0]):
        #    if unfolded_mask[batch, 1:].all():
        #        unfolded_mask[batch, 0] = 0  # depot only option
        #    else:
        #        unfolded_mask[batch, 0] = 0  # other options available

        return unfolded_mask[
            :, None, :
        ]  # np.reshape(unfolded_mask, (self.batch_size, 1, -1))

    def all_finished(self):
        return all(self.done)

    def get_costs(self):
        return self.rewards

    def get_distance(self):
        return self.travel_cost

    def get_max_l_perc(
        self,
    ):  # returns 1 - furthest l loading location in percent of total length of container
        # current truck
        curr_truck = np.max(self.state[:, :, 15], axis=1)
        curr_truck = np.maximum(
            curr_truck, np.ones(self.batch_size)
        )  # for initial iteration
        curr_containers = self.containers[
            np.arange(self.batch_size).astype(int), curr_truck.astype(int) - 1
        ]  # .reshape((self.batch_size, -1))
        max_l_perc = (
            1
            - np.sum(np.sum(np.sum(abs(curr_containers), axis=1), axis=1) != 0, axis=1)
            / self.containers.shape[4]
        )

        return max_l_perc.reshape((self.batch_size, -1))

    def packages_remaining(self):
        return (np.sum(self.state[:, :, 22], axis=1) > 0).reshape((self.batch_size, -1))

    @jit(parallel=True, cache=True)
    def parallel_feas(
        self, all_batch_state, all_batch_containers, all_batch_row, all_batch_rot,
    ):

        my_dict = dict()
        masks = self.get_mask()[:, 0, :] * (-1) + 1  # flip --> 0 is masked now

        for batch_num in prange(self.batch_size):
            batch_row = copy.deepcopy(all_batch_row[batch_num])
            batch_rot = copy.deepcopy(all_batch_rot[batch_num])
            doable = False

            while doable == False:

                batch_state = copy.deepcopy(all_batch_state[batch_num])

                # add in the weight masking (weight constraint -> this way because it spans multiple packages)
                batch_state[1:, 27] = masks[batch_num, 1:]

                batch_containers = copy.deepcopy(all_batch_containers[batch_num])
                curr_cont = copy.deepcopy(
                    batch_containers[np.max(batch_state[:, 15]) - 1]
                )

                batch_state, chosen_info, act_iters = self.update_feasibility(
                    batch_state, curr_cont, batch_row, batch_rot,
                )

                # check for same delivery poss.
                i_row, i_count = np.unique(batch_state[:, 0], return_counts=True)
                how_many = i_count[batch_state[chosen_info[0]][0]]

                if (how_many > 1) and (batch_state[:, 21].all()):
                    same_dest_rows = np.where(
                        batch_state[:, 0] == batch_state[chosen_info[0], 0]
                    )[0].tolist()
                    all_same_dest_rows = copy.deepcopy(same_dest_rows)
                    same_dest_rows.remove(chosen_info[0])

                    # Try all permuations

                    if how_many == 2:
                        doable = self.check_and_load(
                            batch_state, curr_cont, same_dest_rows, chosen_info[0]
                        )
                    elif how_many == 3:
                        doable_1 = self.check_and_load(
                            batch_state, curr_cont, same_dest_rows, chosen_info[0]
                        )
                        same_dest_rows.reverse()
                        doable_2 = self.check_and_load(
                            batch_state, curr_cont, same_dest_rows, chosen_info[0]
                        )
                        doable = doable_1 or doable_2
                    else:
                        assert (
                            how_many <= 3
                        ), "Environment is only designed for up to three packages per destination. You have {}".format(
                            how_many
                        )

                    if doable == False:
                        # remove impossible
                        batch_row = np.delete(
                            batch_row, np.where(batch_row == chosen_info[0])[0][0]
                        )

                else:
                    doable = True

            my_dict[batch_num] = [
                batch_state,
                chosen_info,
                act_iters,
            ]  # allows for non sequential execution

        return my_dict

    def check_and_load(self, state_in, container_in, rows_in, initial_row):
        state = copy.deepcopy(state_in)
        container = copy.deepcopy(container_in)
        rows = copy.deepcopy(rows_in)

        container = self.load_single(container, state, initial_row)

        single_trip_doable = True
        for i in rows_in:
            # state[:, 19:25] = 1
            state, act, _ = self.update_feasibility(state, container, rows)

            if act[0] in rows_in:
                container = self.load_single(container, state, act[0])
                rows.remove(act[0])
            else:
                single_trip_doable = False
                break

        return single_trip_doable

    def load_single(self, container, state, row):
        row = row
        rot = state[row, 13]

        # state info for action
        pack_h = state[row, 3].astype(np.int32)
        pack_w = state[row, 4].astype(np.int32)
        pack_l = state[row, 5].astype(np.int32)
        pack_frag = state[row, 6].astype(np.int32)

        if rot == 0:
            pos_h = state[row, 7].astype(np.int32)
            pos_w = state[row, 8].astype(np.int32)
            pos_l = state[row, 9].astype(np.int32)
            state[row, 10] = -1
            state[row, 11] = -1
            state[row, 12] = -1
        else:
            pos_h = state[row, 10].astype(np.int32)
            pos_w = state[row, 11].astype(np.int32)
            pos_l = state[row, 12].astype(np.int32)
            state[row, 7] = -1
            state[row, 8] = -1
            state[row, 9] = -1

        container[
            pos_h : (pos_h + pack_h), pos_w : (pos_w + pack_w), pos_l : (pos_l + pack_l)
        ] = (1 - 2 * pack_frag)

        return container

    def batch_pipeline(
        self,
        all_batch_state,
        all_batch_containers,
        all_batch_row,
        all_batch_rot,
        # all_batch_clean_act,
    ):
        """
        Combines all necessary operations for one batch and returns the updated state and container(s)
        """
        # TODO: done skip

        feas_dict = self.parallel_feas(
            all_batch_state, all_batch_containers, all_batch_row, all_batch_rot,
        )

        for batch_num in range(self.batch_size):

            # batch_state = all_batch_state[batch_num]
            batch_containers = all_batch_containers[batch_num]
            batch_row = all_batch_row[batch_num]
            batch_rot = all_batch_rot[batch_num]
            # batch_clean_act = all_batch_clean_act[batch_num]

            batch_state, chosen_info, act_iters = feas_dict[batch_num]

            self.avg_act_iters[batch_num] += act_iters  # * (1 / self.state.shape[1])

            batch_clean_act = chosen_info[0]
            self.act_chosen[batch_num] = chosen_info[0]
            self.row_chosen[batch_num] = chosen_info[1]
            self.rot_chosen[batch_num] = chosen_info[2]

            # load
            batch_state, batch_containers, done = self.load_package(
                batch_state, batch_containers, chosen_info[1], chosen_info[2]
            )

            # add capacity (easier otuside of function to apply to all batches)
            if chosen_info[1] == 0:
                self.cur_capacity[batch_num] = self.capacity[batch_num]
            else:
                self.cur_capacity[batch_num] = (
                    self.cur_capacity[batch_num]
                    - self.state[batch_num, chosen_info[1], 25]
                )

            # print(self.pandas_state(0))

            self.state[batch_num] = batch_state
            self.containers[batch_num] = batch_containers
            self.done[batch_num] = done

        # correct to row
        self.action_list.append(self.act_chosen)
        self.clean_action_list = np.array([np.array(xi) for xi in self.action_list]).T
        self.clean_action_list = self.clean_action_list.astype(int)

        for batch_num in range(self.batch_size):
            batch_clean_act = self.clean_action_list[batch_num]
            batch_state = all_batch_state[batch_num]

            # Reward calculation (different options)
            penalty_factor = 2  # distance based penalty (2 == back and forth to depot)

            trajectory_dist = np.zeros(self.batch_size)
            my_penalty = np.zeros(self.batch_size)

            # Distance traveled
            prev = batch_state[0, 1:3]
            trajectory_dist = 0
            for act in self.clean_action_list[batch_num]:
                now = self.state[batch_num, act, 1:3]
                trajectory_dist += np.sqrt(
                    (prev[0] - now[0]) ** 2 + (prev[1] - now[1]) ** 2
                )
                prev = now

            # Max penalty (back and forth from depot to location)
            prev = batch_state[0, 1:3]
            cum_penalty = 0
            for act in range(self.state.shape[1]):
                now = self.state[batch_num, act, 1:3]
                cum_penalty += np.sqrt(
                    (prev[0] - now[0]) ** 2 + (prev[1] - now[1]) ** 2
                )

            missed = np.sum(batch_state[1:, 16] == 0)
            any_missed = (np.sum(batch_state[1:, 16] == 0) >= 1) * 1
            avg_penalty = cum_penalty / (
                self.state.shape[1] - 1
            )  # average to eliminate multiple packages per location
            max_penalty = avg_penalty * self.n_destinations * penalty_factor

            # Number of containers used (scale with penalty factor to avoid weird learned tradeoffs)
            used = 0
            available_trucks = self.containers[batch_num, :, :, :, :].shape[0]
            for i_cont in range(available_trucks):
                used += abs(self.containers[batch_num, i_cont, :, :, :]).any() * 1

            # Combine penalties
            my_penalty = (
                avg_penalty * missed * penalty_factor
                + max_penalty / available_trucks * used
            )

            # Combine costs
            travel_cost = trajectory_dist / max_penalty
            truck_cost = used / available_trucks / self.n_destinations
            missed_cost = missed / self.n_destinations

            batch_rewards = travel_cost + missed_cost  # +truck_cost * (1 - any_missed)

            # simplified reward
            # batch_rewards = (
            #    trajectory_dist / max_penalty
            #    + any_missed * 2
            #    + used / available_trucks * (1 - any_missed)
            # )

            # Packing reward
            """
            total_fill_length = np.sum(
                (
                    np.sum(
                        np.sum(abs(self.containers[batch_num, :, :, :, :]), axis=1),
                        axis=1,
                    )
                    != 0
                )
            )
            volume_max = total_fill_length * self.vehicle_h * self.vehicle_w
            volume_used = np.sum(abs(self.containers[batch_num, :, :, :, :]))

            vol_utilized = volume_used / max(1, volume_max)
            """

            self.rewards[batch_num] = batch_rewards
            self.travel_cost[batch_num] = (
                trajectory_dist + 100 * missed
            )  # for better visualization of missed packages

        # return batch_state, batch_rewards, batch_containers, done, trajectory_dist

    @jit(parallel=True, cache=True)
    def load_package(self, state, containers, row, rot):
        """
        Updates the container array with the chosen package and packing position
        If action = 0 it returns the vehicle to the depot and starts a new one
        """

        """
        if rot == 0:
            assert (
                state[row, 23] == 1
            ), "Package placement location is not feasible. You might be trying to pack the same item twice  \n{}\n{}".format(
                self.state_columns, state[row, :]
            )
        else:
            assert (
                state[row, 24] == 1
            ), "Package placement location is not feasible. You might be trying to pack the same item twice \n{}\n{}".format(
                self.state_columns, state[row, :]
            )
        """

        rot = state[row, 13]  # safety

        done = False

        # get current container number
        my_truck = np.max(state[:, 15])

        # state info for action
        pack_h = state[row, 3].astype(np.int32)
        pack_w = state[row, 4].astype(np.int32)
        pack_l = state[row, 5].astype(np.int32)
        pack_frag = state[row, 6].astype(np.int32)

        if rot == 0:
            pos_h = state[row, 7].astype(np.int32)
            pos_w = state[row, 8].astype(np.int32)
            pos_l = state[row, 9].astype(np.int32)
            state[row, 10] = -1
            state[row, 11] = -1
            state[row, 12] = -1
        else:
            pos_h = state[row, 10].astype(np.int32)
            pos_w = state[row, 11].astype(np.int32)
            pos_l = state[row, 12].astype(np.int32)
            state[row, 7] = -1
            state[row, 8] = -1
            state[row, 9] = -1

        # Mask same destination packages that were not packed after leaving destination
        if True:
            prev_row = np.argmax(state[:, 16])
            prev_dest = state[prev_row, 0]
            if (prev_dest != state[row, 0]) and (
                state[row, 16] >= 1
            ):  # at next dest and not first iteration
                state[1:, 22] = state[1:, 22] * (
                    1 - (state[1:, 0] == prev_dest) * 1
                )  # prev dest no longer available

        # Mask to only alow same destination packages if available
        same_dest_avail = (state[:, 0] == state[row, 0]) * (state[:, 16] == 0) * 1
        state[row, 22] = 0
        if np.sum(same_dest_avail[1:]) > 1:
            state[1:, 21] = same_dest_avail[1:]
        else:
            state[:, 21] = 1

        # state[:, 21] = 1 # Overrides for SDVRP

        # TODO: remove to enforce constraint
        # state[:, 21] = 1

        # update container
        if row == 0:  # depot trip
            state[0, 15] = max(1, np.max(state[:, 15]) + 1)
            my_truck = state[0, 15]
            if state[0, 15] >= containers.shape[0] + 1:
                done = True  # used all available vehicles
                state[:, 22] = 0  # no more loading possible
                my_truck = containers.shape[0]
                state[0, 15] = containers.shape[0]
            else:
                # reset l skip in feasibility check
                state[:, 17] = 0
                state[:, 18] = 0
                # in else to keep info in state for inspection
        else:
            if rot == 0:
                containers[
                    my_truck - 1,
                    pos_h : (pos_h + pack_h),
                    pos_w : (pos_w + pack_w),
                    pos_l : (pos_l + pack_l),
                ] = (
                    1 - 2 * pack_frag
                )  # 1 fo non-fragile, -1 for fragile, 0 empty
                # update state
                state[row, 7:10] = [pos_h, pos_w, pos_l]
                state[row, 10:13] = [-1, -1, -1]
            elif rot == 1:
                state[row, 13] = 1
                containers[
                    my_truck - 1,
                    pos_h : (pos_h + pack_h),
                    pos_w : (pos_w + pack_l),  # swapped l and w for rotation
                    pos_l : (pos_l + pack_w),
                ] = (
                    1 - 2 * pack_frag
                )  # 1 fo non-fragile, -1 for fragile, 0 empty
                # update state
                state[row, 7:10] = [-1, -1, -1]
                state[row, 10:13] = [pos_h, pos_w, pos_l]
            else:
                raise ValueError("{} not a valid rotation (only 0 or 1)".format(rot))

        # update state
        state[row, 15] = my_truck  # to have non-loaded packages with 0

        # update loading sequence helper column
        state[row, 16] = np.max(state[:, 16]) + 1

        # update distance
        state[:, 14] = np.sqrt(
            (state[:, 1] - state[row, 1]) ** 2 + (state[:, 2] - state[row, 2]) ** 2
        )

        return state, containers, done

    # @profile
    @jit(cache=True)
    def update_feasibility(self, state, container, rows, rots=None):
        """
        Updates state & mask with feasibility (T/F) and the min values for packing (from both rotation possibilities) 
        """
        chosen = [0, 0, 0]  # action, row, rot
        is_feas = 0
        act_counter = 0

        h_arange = abs(np.arange(container.shape[0]))[:, np.newaxis] + 100

        # are we at the depot?

        at_home = np.argmax(state[:, 16]) == 0  # True if at depot
        any_packages_left = state[
            1:, 22
        ].any()  # True if packages left to be delivered and available
        allow_home = (
            not at_home or not any_packages_left
        )  # forces loading (do not use; only debugging)
        # allow_home = not at_home  # don't skip to next truck

        # depot
        state[0, 21] = 1
        state[0, 22] = 1

        r_state = copy.deepcopy(state)
        r_state[:, 4] = state[:, 5]
        r_state[:, 5] = state[:, 4]

        # shortest l implementation

        zero_return = copy.deepcopy((state, [0, 0, 0], act_counter + 1))

        for prob_i in range(len(rows)):

            row = int(rows[prob_i])
            rot = 0  # int(rots[prob_i]) # no longer used with heuristic
            if row == 0:  # depot needs no check
                # continue
                # if state[:, 21].all() == True: # force constraint

                if state[1:, 21].all() and allow_home:
                    return (state, [0, 0, 0], act_counter + 1)
                else:
                    continue
                # else:
                #    continue
            elif (
                state[row, 22] * state[row, 21] * state[row, 27]
            ) == 0:  # not feasible
                continue
            else:

                act_counter += 1
                (
                    state[row, 19],
                    state[row, 7:10],
                    state[row, 17],
                ) = self.single_feasibility(
                    state[row, :], container, state[row, 17], h_arange
                )
                is_feas_n = state[row, 19]
                n_l = state[row, 9] + state[row, 5]
                rot = 0
                n_return = (
                    state,
                    [row, row, 0],
                    act_counter,
                )

                (
                    state[row, 20],
                    state[row, 10:13],
                    state[row, 18],
                ) = self.single_feasibility(
                    r_state[row, :], container, r_state[row, 18], h_arange
                )
                is_feas_r = state[row, 20]
                r_l = state[row, 12] + r_state[row, 5]

                r_return = (
                    state,
                    [row, row, 1],
                    act_counter,
                )

                case = None
                if is_feas_r + is_feas_n == 0:
                    continue
                elif is_feas_r * is_feas_n == 1:
                    # calc wasted space:
                    wasted_n = 0
                    wasted_r = 0
                    if True:
                        wasted_n = self.get_wasted_space(
                            container, state[row, 3:6], state[row, 7:10]
                        )
                        wasted_r = self.get_wasted_space(
                            container, r_state[row, 3:6], state[row, 10:13]
                        )

                    if wasted_n < wasted_r:
                        case = 0
                    elif wasted_n > wasted_r:
                        case = 1
                    elif n_l <= r_l:
                        case = 0
                    else:
                        case = 1
                elif is_feas_r == 1:
                    case = 1
                elif is_feas_n == 1:
                    case = 0

                # return correspondingly
                if case == 0:  # normal orientation
                    return (
                        state,
                        [row, row, 0],
                        act_counter,
                    )
                elif case == 1:  # rotatet
                    state[row, 13] = 1  # rotation yes
                    return (
                        state,
                        [row, row, 1],
                        act_counter,
                    )

        return zero_return  # (state, chosen, act_counter)  # depot

    @jit(cache=True)
    def get_wasted_space(self, container, hwl_package, pos_package):
        waste_container = -abs(container) + 1

        h_min = pos_package[0]
        h_max = h_min + hwl_package[0]
        w_min = pos_package[1]
        w_max = w_min + hwl_package[1]
        l_max = pos_package[2]

        AOI = container[h_min:h_max, w_min:w_max, 0:l_max]
        waste_container = abs(AOI)

        prev = 0
        for l_aoi in np.arange(l_max)[::-1]:  # reverse order
            waste_container[:, :, l_aoi] = waste_container[:, :, l_aoi] + prev
            prev = waste_container[:, :, l_aoi]

        waste_container = (waste_container == 0) * 1
        wasted = np.sum(np.sum(waste_container))

        return wasted

        """
        for prob_i in range(len(rows)):

            row = int(rows[prob_i])
            rot = int(rots[prob_i])
            # print(row, rot)
            # check available
            if (state[row, 22] * state[row, 21]) == 0:  # not feasible
                continue
            elif row == 0:  # depot needs no check and is not desired until full
                # continue
                return (state, [0, 0, 0], act_counter + 1)
            elif rot == 0:
                act_counter += 1
                (
                    state[row, 19],
                    state[row, 7:10],
                    state[row, 17],
                ) = self.single_feasibility(
                    state[row, :], container, state[row, 17], h_arange
                )
                is_feas = state[row, 19]

                if is_feas == 1:
                    return (
                        state,
                        [row + self.state.shape[1] * rot, row, rot],
                        act_counter,
                    )

            else:
                act_counter += 1
                r_state = copy.deepcopy(state)
                r_state[:, 4] = state[:, 5]
                r_state[:, 5] = state[:, 4]
                (
                    state[row, 20],
                    state[row, 10:13],
                    state[row, 18],
                ) = self.single_feasibility(
                    r_state[row, :], container, state[row, 18], h_arange
                )
                is_feas = state[row, 20]

                if is_feas == 1:
                    return (
                        state,
                        [row + self.state.shape[1] * rot, row, rot],
                        act_counter,
                    )
        return state, chosen, act_counter
        """

    @jit(cache=True)
    def single_feasibility(self, my_row, container, l_skip, h_arange):
        """
        Takes package dimension, rotation and current container
        Returns (T/F) if the most top back left location is feasible (# position in container)
        """

        l_skip_update = l_skip

        dim = my_row[3:6]
        frag = my_row[6]

        max_poss_h = container.shape[0] - dim[0]
        max_poss_w = container.shape[1] - dim[1]

        if dim[1] > container.shape[1]:  # can't fit
            0, [-1, -1, -1], 0

        for l in range(container.shape[2] - l_skip - dim[2] + 1):
            l = l + l_skip

            if l % self.step_size != 0 and l != 0:
                continue

            # skip filled spots
            h_check = abs(container[:, :, l]) * h_arange
            hs_for_l = np.argmin(h_check, axis=0)

            for w in range(max_poss_w + 1):

                if w % self.step_size != 0 and w != 0:
                    continue

                h_skip = hs_for_l[w]

                # filled to the achievable top
                if (container[0, w, l] != 0) & (h_skip == 0):
                    continue

                for h in range(
                    max_poss_h + 1 - h_skip
                ):  # innermost loop h allows break due to non-floating constraint

                    h = h + h_skip
                    if h % self.step_size != 0 and h != 0:
                        continue

                    """
                    The order of the constraints significantly impacts compute time (e.g. LIFO check first increases compute by x6)
                    """

                    if container[h, w, l] != 0:  # already taken
                        continue

                    # check if package is supported (min. 75% and fragility)
                    if h > 0:

                        if frag == 0:
                            # fragility constraint (only fragile allowed on fragile)
                            weakest_support = np.min(
                                container[0:h, w : (w + dim[1]), l : (l + dim[2]),]
                            )

                            if (frag == 0) and (
                                weakest_support == -1
                            ):  # no non-frag on frag
                                break

                        # minimum supporting area (strict? --> needs to be supportet all the way to the floor) #TODO: check if benchmarks do so too
                        supporting_area = np.sum(
                            np.prod(
                                (container[0:h, w : (w + dim[1]), l : (l + dim[2]),])
                                ** 2,  # **2 == abs
                                axis=0,
                            )
                        )  # abs for frag -1, non-zero if support all the way along h axis

                        package_area = dim[1] * dim[2]

                        min_support = 0.75  # TODO: move to config file
                        if supporting_area < (package_area * min_support):
                            break

                    if (
                        frag == 1
                    ):  # check that we do not move the fragile package under a non-fragile one
                        if (h + dim[0] + 1) < self.vehicle_h:
                            stuff_on_top = np.max(
                                container[
                                    (h + dim[0] - 1) : (h + dim[0] + 1),
                                    w : (w + dim[1]),
                                    l : (l + dim[2]),
                                ]
                            )
                            if stuff_on_top == 1:
                                continue

                    # check if LIFO constraint is fulfilled (unload along l axis)
                    if np.any(
                        (container[h : (h + dim[0]), w : (w + dim[1]), l:])
                        ** 2,  # **2 == abs
                    ):  # abs for frag -1
                        continue

                    my_h = h
                    my_w = w
                    my_l = l
                    return 1, [my_h, my_w, my_l], l_skip_update

            # check if this l can be skipped in the future
            free_h = max(self.vehicle_h - np.sum(container[:, :, 0], axis=1)) < dim[0]
            if free_h == 0:
                l_skip_update = l
                continue
            free_w = max(self.vehicle_w - np.sum(container[:, :, 0], axis=0)) < dim[1]
            if free_w == 0:
                l_skip_update = l

        return 0, [-1, -1, -1], l_skip_update

    def pandas_state(self, batch_nr):
        """
        For easier inspection
        """
        import pandas as pd

        pd.set_option("display.max_columns", None)

        pd_df = pd.DataFrame(self.state[batch_nr], columns=self.state_columns)
        return pd_df

    def reset(
        self, env_instance: int = None, folder: str = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        gen_mem = self.gendreau
        self.close()
        self.gendreau = gen_mem
        self._generate_env()
        return self.state

    def render_vrp(
        self, batch_nr, fig_name="unnamed", save_plt=False, show_plt=True, xy_lim=-1,
    ):
        """
        Renders the routing in xy
        """

        # quick fix in case of modification of locations afer env. creation
        self.clean_xy_trajectory[0, batch_nr] = self.state[batch_nr, 0, 1:3]

        df_items = self.pandas_state(batch_nr)

        df_only_items = df_items.copy()
        df_only_items = df_only_items.drop(0)
        depot_x = df_items.loc[0, "x"]
        depot_y = df_items.loc[0, "y"]
        df_only_items["volume"] = df_items["h"] * df_items["w"] * df_items["l"]
        df_only_items = df_only_items.filter(["dest", "volume"], axis=1)

        df_vols = df_only_items.groupby(by="dest").sum().reset_index()
        df_vols = df_items.merge(df_vols, on="dest")

        fig = plt.figure(figsize=(7, 7))

        # plot delivery locations (=customers)
        plt.scatter(
            df_vols.loc[:, "x"],
            df_vols.loc[:, "y"],
            s=(df_vols.loc[:, "volume"]) / max((df_vols.loc[:, "volume"])) * 100,
            alpha=1,
            marker="o",
        )

        for row in range(df_vols.shape[0]):
            plt.text(
                df_vols.loc[row, "x"] + 1,
                df_vols.loc[row, "y"] + 1,
                str(df_vols.loc[row, "dest"]),
                size=12,
            )

        # plot depot
        plt.scatter(depot_x, depot_y, s=200, alpha=1, marker="s")

        # add delivery paths
        if xy_lim == -1:
            location_order = self.clean_xy_trajectory[:, batch_nr]
        else:  # custom step by step
            location_order = self.clean_xy_trajectory[:, batch_nr][0:xy_lim]

        facecolors = ["blue", "green", "red", "c", "m", "y"]
        face_i = 0

        # for arrow scaling
        delta_x = (np.max(df_vols.loc[:, "x"]) - np.min(df_vols.loc[:, "x"])) / 100
        delta_y = (np.max(df_vols.loc[:, "y"]) - np.min(df_vols.loc[:, "y"])) / 100
        scale_arrow = max(delta_x, delta_y)

        if location_order is not None:
            prev = [-1, -1]
            new = True
            first = True
            for i in range(location_order.shape[0] - 1):

                # flipped as we load in reverse delivery order
                x_0 = location_order[i + 1, 0]
                dx_1 = location_order[i, 0] - x_0
                y_0 = location_order[i + 1, 1]
                dy_1 = location_order[i, 1] - y_0

                if new:
                    plt.arrow(
                        x=x_0,
                        y=y_0,
                        dx=dx_1,
                        dy=dy_1,
                        width=0.5 * scale_arrow,
                        head_width=2.5 * scale_arrow,
                        head_length=3 * scale_arrow,
                        facecolor=facecolors[face_i],
                        edgecolor="none",
                        length_includes_head=True,
                        alpha=0.5,
                        label="Vehicle " + str(face_i + 1),
                    )
                else:
                    plt.arrow(
                        x=x_0,
                        y=y_0,
                        dx=dx_1,
                        dy=dy_1,
                        width=0.5 * scale_arrow,
                        head_width=2.5 * scale_arrow,
                        head_length=3 * scale_arrow,
                        facecolor=facecolors[face_i],
                        edgecolor="none",
                        length_includes_head=True,
                        alpha=0.5,
                    )
                new = False
                prev = [location_order[i + 1, 0], location_order[i + 1, 1]]
                if (prev[0] == depot_x) and (prev[1] == depot_y) and not first:
                    face_i += 1
                    new = True
                    if face_i >= len(facecolors):  # loop through list
                        face_i = 0

                first = False

        plt.legend(loc="upper left", bbox_to_anchor=(0, 1))
        plt.gca().set_aspect("equal")
        """
        max_x = np.max(df_vols.loc[:, "x"])
        max_y = np.max(df_vols.loc[:, "y"])
        max_xy = max(max_x, max_y)
        if max_xy > 100:
            pass
        else:
            max_xy = 100

        plt.xlim(0, max_xy)
        plt.ylim(0, max_xy)
        """
        plt.xlabel("X", size=16)
        plt.ylabel("y", size=16)
        """
        plt.title(
            "[{}] \n Delivery Locations (Size eq. Order Volume | Square is Depot)".format(
                fig_name
            ),
            size=16,
        )
        """
        plt.title(
            "{} Vehicle Routes".format(fig_name), size=16,
        )

        if save_plt:
            fig.savefig("../src/instance_plots/{}_destinations.pdf".format(fig_name))

        if show_plt:
            plt.show()

        # https://ndres.me/post/matplotlib-animated-gifs-easily/
        # Used to return the plot as an image rray
        fig.canvas.draw()  # draw the canvas, cache the renderer
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        return image

    def render_container(
        self,
        batch,
        n_container,
        fig_name="unnamed",
        save_plt=False,
        show_plt=True,
        n_limit=-1,
    ):
        """
        Renders a single container of a single batch
        """

        import matplotlib.pyplot as plt

        if n_limit == -1:
            n_limit = self.pandas_state(0).shape[0] * 999  # dirty fix
        n_limit = int(n_limit)

        my_positions = []  # needs w,l,h order for next functions
        my_sizes = []  # needs w,l,h order
        coloring = []  # What to color the items by (e.g. fraility, destination, ...)
        # TODO: add legend of coloring to plot

        for row in range(self.state[batch, :].shape[0]):

            veh_packed = self.state[batch, row, 15]
            item_loc = self.state[batch, row, 0]
            seq_nr = self.state[batch, row, 16]

            # only take items from selected vehicle (and not depot)
            if (
                (veh_packed == (n_container + 1))
                and (item_loc != 0)
                and (seq_nr <= n_limit)
            ):
                my_h = self.state[batch, row, 3]
                my_w = self.state[batch, row, 4]
                my_l = self.state[batch, row, 5]

                pos_h = self.state[batch, row, 7]
                pos_w = self.state[batch, row, 8]
                pos_l = self.state[batch, row, 9]

                r_pos_h = self.state[batch, row, 10]
                r_pos_w = self.state[batch, row, 11]
                r_pos_l = self.state[batch, row, 12]

                rot = self.state[batch, row, 13]
                coloring.append(self.state[batch, row, 6])

                # consider rotation
                if rot == 0:
                    my_positions.append([pos_w, pos_l, pos_h])
                    my_sizes.append([my_w, my_l, my_h])
                elif rot == 1:
                    my_positions.append([r_pos_w, r_pos_l, r_pos_h])
                    my_sizes.append([my_l, my_w, my_h])
                else:
                    raise ValueError("Not a valid fragility value: {}".format(rot))

        # truck dimensions
        t_h = self.containers[batch, n_container].shape[0]
        t_w = self.containers[batch, n_container].shape[1]
        t_l = self.containers[batch, n_container].shape[2]

        # to plot the truck limits
        positions = [(0, 0, 0)] + my_positions
        sizes = [(t_w, t_l, t_h)] + my_sizes

        colors = ["#f6f6f6"]
        for i in coloring:
            colors += [self.color_map_color(value=i, cmap_name="bwr", vmax=1)]

        fig = plt.figure(figsize=(7, 7))

        # 3D rendering
        ax = fig.gca(projection="3d")
        ax.view_init(30, 70)  # better viewing angle for containers

        # print(positions, colors, sizes)
        pc = self.plotCubeAt2(
            positions[0], sizes[0], colors=colors[0], my_alpha=0.1, edgecolor="k"
        )
        ax.add_collection3d(pc)
        pc = self.plotCubeAt2(
            positions[1:], sizes[1:], colors=colors[1:], my_alpha=0.5, edgecolor="k"
        )
        ax.add_collection3d(pc)

        ax.set_xlim([0, t_w])
        ax.set_ylim([0, t_l])
        ax.set_zlim([0, t_h])

        ax.set_box_aspect([t_w, t_l, t_h])

        import matplotlib.patches as mpatches

        normal_patch = mpatches.Patch(
            color=(0, 0, 1, 0.5), label="Normal"
        )  # hardcoded colors
        frag_patch = mpatches.Patch(color=(1, 0, 0, 0.5), label="Fragile")

        plt.legend(
            handles=[normal_patch, frag_patch],
            loc="upper left",
            bbox_to_anchor=(0.65, 0.9),
        )

        # add capacity info

        filling_grade = np.round(
            (
                np.sum(
                    self.state[batch, self.state[batch, :, 15] == n_container + 1, 25]
                )
                / self.truck_capacity
            ),
            1,
        )

        fig_name = fig_name + str(n_container)

        plt.title(
            "Container {} | {}% Weight Cap.".format(n_container + 1, filling_grade),
            size=16,
        )

        if save_plt:
            fig.savefig(
                "../src/instance_plots/{}_container_{}.pdf".format(
                    fig_name, n_container + 1
                )
            )

        if show_plt:
            plt.show()

        # https://ndres.me/post/matplotlib-animated-gifs-easily/
        # Used to return the plot as an image rray
        fig.canvas.draw()  # draw the canvas, cache the renderer
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        return image

    def color_map_color(self, value, cmap_name="prism", vmin=0, vmax=50):
        # from GitHub
        # norm = plt.Normalize(vmin, vmax)
        import matplotlib as mpl
        import matplotlib.cm as cm

        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = cm.get_cmap(cmap_name)  # PiYG

        rgb = cmap(norm(abs(value)))[
            :3
        ]  # will return rgba, we take only first 3 so we get rgb
        color = mpl.colors.rgb2hex(rgb)
        return color

    def plotCubeAt2(self, positions, sizes=None, colors=None, my_alpha=0.5, **kwargs):
        # from GitHub
        # if not isinstance(colors,(list,np.ndarray)): colors=["C0"]*len(positions)
        # if not isinstance(sizes,(list,np.ndarray)): sizes=[(1,1,1)]*len(positions)
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        try:
            g = []
            for p, s, c in zip(positions, sizes, colors):
                g.append(self.cuboid_data2(p, size=s))

            return Poly3DCollection(
                np.concatenate(g),
                alpha=my_alpha,
                facecolors=np.repeat(colors, 6),
                **kwargs,
            )
        except:
            return Poly3DCollection(
                g, alpha=my_alpha, facecolors=np.repeat(colors, 6), **kwargs
            )

    def cuboid_data2(self, o, size=(1, 1, 1)):
        # from GitHub
        X = [
            [[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]],
            [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
            [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
            [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]],
            [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
            [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]],
        ]
        X = np.array(X).astype(float)
        for i in range(3):
            X[:, :, i] *= size[i]
        X += np.array(o)
        return X

    def disp_batch(self, batch=0, truck=0):
        self.render_vrp(
            batch,
            "{} Veh., {} Items, {} Dest.".format(
                self.max_trucks[batch], self.state.shape[1] - 1, self.n_destinations,
            ),
            False,
            True,
        )

        for n_truck in range(self.max_trucks[batch]):
            self.render_container(
                batch,
                n_truck,
                "Batch {}, Veh. {}".format(batch, truck + 1),
                False,
                True,
            )

        return self.pandas_state(batch)  # .sort_values(by=["seq_nr"])

    def disp_step_by_step(self, batch=0, n_limit=1, which_img="vrp"):
        """
        Plots vrp and loading step by step
        """

        # assert n_limit >= 1, "n_limit must be >= 1; can't pack 'nothing'"

        packed_items = self.pandas_state(batch).sort_values(by=["seq_nr"])
        packed_items = packed_items[packed_items.seq_nr != 0]

        xy_traj_limit = packed_items.iloc[
            n_limit, 16
        ]  # handles the sequence skips of going to the depot

        truck = packed_items.iloc[n_limit, 15] - 1

        image_vrp = self.render_vrp(
            batch,
            "'Step #{}' Batch {}: {} Veh., {} Items, {} Dest.".format(
                xy_traj_limit,
                batch,
                self.max_trucks[batch],
                self.state.shape[1] - 1,
                self.n_destinations,
            ),
            False,
            False,
            xy_traj_limit,
        )

        image_container = self.render_container(
            batch,
            truck,
            "'Step #{}' Batch {}, Veh. ".format(xy_traj_limit, batch),
            False,
            False,
            xy_traj_limit,
        )

        is_not_done = n_limit < packed_items.shape[0]

        if which_img == "vrp":
            return image_vrp
        else:
            return image_container

    def make_gif(self, batch=0, my_title="___", fps=1):
        """
        Save a gif of the vrp and container loading process
        To combine two gifs side by side: https://ezgif.com/combine
        """

        import imageio

        packed_items = self.pandas_state(batch).sort_values(by=["seq_nr"])
        packed_items = packed_items[packed_items.seq_nr != 0]
        # (packed_items)
        max_i = packed_items.shape[0]

        kwargs_write = {"fps": fps, "quantizer": "nq"}
        imageio.mimsave(
            "../src/instance_gifs/{}_loading.gif".format(my_title),
            [self.disp_step_by_step(batch, i, "cont") for i in np.arange(max_i)],
            fps=fps,
        )

        imageio.mimsave(
            "../src/instance_gifs/{}_vrp.gif".format(my_title),
            [self.disp_step_by_step(batch, i, "vrp") for i in np.arange(max_i)],
            fps=fps,
        )

        return "Done"

    def get_instance(self, my_instance_num):
        """
        Takes the filepath to a Gendreau Instance txt
        and returns the information readable in numpy format.
        """

        import pandas as pd

        # TODO: fix file access, currently workaround for filepath
        try:
            my_instance_path = "../Data/instances_Gendreau/3l_cvrp{}.txt".format(
                str(my_instance_num).zfill(2)
            )
            with open(my_instance_path) as f:
                scenario = f.readlines()
        except:
            my_instance_path = "./Data/instances_Gendreau/3l_cvrp{}.txt".format(
                str(my_instance_num).zfill(2)
            )
            with open(my_instance_path) as f:
                scenario = f.readlines()

        reading = 0
        loc_header = "Node - x - y - demand"
        pack_header = "Node - number of items - h - w - l - fragility for each item"
        veh_num_line = "--- number of vehicles"

        locations = list()
        items = list()
        num_vehicles = 0

        # Always the same, thus no need to read
        self.veh_h = self.vehicle_h
        self.veh_w = self.vehicle_w
        self.veh_l = self.vehicle_l

        for line in scenario:
            # select table
            if loc_header in line:
                reading = 1
            elif pack_header in line:
                reading = 2
            elif veh_num_line in line:
                num_vehicles = np.int("".join(x for x in line if x.isdigit()))

            if reading == 1:
                locations.append(
                    line.replace("Node", " Node")
                    .replace("\n", "")
                    .replace("-", " ")
                    .replace(
                        "100   ", ",100   "
                    )  # dirty fix (does not scale beyond 100 destinations=)
                    .replace(" ", ",")
                    .replace(",,", ",")
                    .replace(",,", ",")
                    .replace(",,", ",")
                    .replace(",,", ",")
                )
            elif reading == 2:
                items.append(
                    line.replace("\n", "")
                    .replace("-", " ")
                    .replace("100   ", ",100   ")
                    .replace(" ", ",")
                    .replace(",,", ",")
                    .replace(",,", ",")
                    .replace(",,", ",")
                    .replace(",,", ",")
                )

        loc = pd.DataFrame(locations)
        items = pd.DataFrame(items)

        # reformat
        loc = loc[0].str.split(",", expand=True)
        loc.columns = loc.iloc[0, :]
        loc.drop(index=0, inplace=True)
        loc.reset_index(inplace=True, drop=True)

        demand = loc.iloc[:, -1]

        loc = loc.drop(["demand"], axis=1)

        items.drop(index=0, inplace=True)
        items = items[0].str.split(",", expand=True)
        items_remaining = items
        item_locs = items.iloc[:, 1:3]

        depot = loc.iloc[0, :]

        for i in range(int(items.shape[1] / 4)):
            item_x = items_remaining.iloc[:, -4:]
            items_remaining = items_remaining.iloc[:, :-4]

            new_entry = pd.concat([item_locs, item_x], ignore_index=True, axis=1)
            new_entry.columns = ["Node", "number of items", "h", "w", "l", "fragility"]

            if i > 0:
                items_expanded = items_expanded.append(new_entry)
            else:
                items_expanded = new_entry

        items_expanded = items_expanded.reset_index(drop=True)
        items_expanded = items_expanded.sort_values("Node")
        items_expanded = items_expanded[
            items_expanded["fragility"].str.contains("None") == False
        ]

        items_df = (
            items_expanded.merge(loc, on="Node", how="left")
            .apply(pd.to_numeric)
            .dropna(axis=1)
        )  # merge to one dataset for easier handling

        return (
            items_df.to_numpy().astype(int),
            depot.to_numpy()[1:].astype(np.float).astype(int),
            [num_vehicles, self.veh_h, self.veh_w, self.veh_l],
            demand,
        )

