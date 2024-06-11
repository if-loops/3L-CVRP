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
        packages_perc_of_av_vol: int = 90,
        frag_freq_prob: int = 10,
        test_set: int = 1,
        folder: str = "Train",
        foldername: str = "Debug",
        gendreau=None,
        seed=None,
    ):
        """Inits VRPEnv with n_destinations.
        ASSUMES THERE IS ONLY ONE DEPOT.
        """

        super(TDCVRP, self).__init__()

        self.seed = seed

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

        # based on Gendreau
        self.max_xy = 100
        self.vehicle_h = 30
        self.vehicle_w = 25
        self.vehicle_l = 60

        self.test_set = test_set
        self.folder = folder
        self.foldername = foldername
        self.batch_size = batch_size
        self.homogeneous = True  # Trucks are all homogeneous i.e. the same.

        # Assetion checks
        assert 0 < min(
            n_destinations, packages_perc_of_av_vol, self.max_xy
        ), "Input must be greater 0"
        assert (
            100 >= frag_freq_prob >= 0
        ), "Fragility frequency must be between 0 and 100"

        self.n_container = np.zeros(self.batch_size)  # current container being loaded
        self.count = 0
        self.counter = 0
        self.rewards = np.zeros(self.batch_size)
        self.travel_cost = np.zeros(self.batch_size)
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
            ]
            self.info = {}
            self.starting = 1

            if self.seed is not None:
                np.random.seed(self.seed)
                # print("Generating Env with seed ", self.seed)

            # Predefined instance from Gendreau et al.
            if self.gendreau is not None:
                # get Gendreau instance
                my_items, my_depot, my_vehicles = self.get_instance(self.gendreau)

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

                self.n_destinations = np.max(my_items[:, 0])

                self.state = self.state.astype(int)

            else:
                # randomly generated instance

                # Create master array of inventory to ship (TODO: different # of packages per batch)
                items_per_dest = np.random.randint(1, 4, self.n_destinations + 1)
                items_per_dest[0] = 1  # depot

                self.state = np.zeros(
                    ((self.batch_size, items_per_dest.sum(), len(self.state_columns)))
                )

                ind = 0

                # collect all destinations seperately for easier access
                self.destinations = np.zeros(
                    (self.batch_size, self.n_destinations + 1, 2)
                )

                for dest in range(self.n_destinations + 1):
                    # create random values shared by all packages of the same destination

                    # coordinates
                    dest_xy = np.random.randint(0, self.max_xy, (self.batch_size, 2))

                    self.destinations[:, dest, :] = dest_xy

                    for package_i in range(items_per_dest[dest]):
                        # shared information
                        self.state[:, ind, 0] = dest
                        self.state[:, ind, 1:3] = dest_xy

                        # create random values per unique package
                        self.state[:, ind, 3] = np.random.randint(
                            low=int(self.vehicle_h * 0.2),
                            high=int(self.vehicle_h * 0.6),
                            size=(self.batch_size),
                        )
                        self.state[:, ind, 4] = np.random.randint(
                            low=int(self.vehicle_w * 0.2),
                            high=int(self.vehicle_w * 0.6),
                            size=(self.batch_size),
                        )
                        self.state[:, ind, 5] = np.random.randint(
                            low=int(self.vehicle_w * 0.2),
                            high=int(self.vehicle_w * 0.6),
                            size=(self.batch_size),
                        )

                        # Fragility
                        p = [1 - self.frag_freq_prob / 100, self.frag_freq_prob / 100]
                        self.state[:, ind, 6] = np.random.choice(
                            2, (self.batch_size), p=p
                        )

                        ind += 1

                self.state = self.state.astype(int)
                # Depot location(s) removal of package info
                self.state[:, 0, 3:17] = 0

                # Decide on number of trucks needed based on volume of packages
                all_pack_vol = np.sum(
                    self.state[:, 1:, 3] * self.state[:, 1:, 4] * self.state[:, 1:, 5],
                    axis=1,
                )

                single_truck = self.vehicle_h * self.vehicle_w * self.vehicle_l
                reduced_single_truck = single_truck * self.packages_perc_of_av_vol / 100
                req_trucks = np.ceil(all_pack_vol / reduced_single_truck).astype(int)
                self.max_trucks = req_trucks

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
            self.state[:, :, 19:] = 1

            self.done = np.full((self.batch_size, 1), False, dtype=bool)
            self.info["n_items"] = {self.state.shape[1]}

            # Keep track of positions (max is latest)
            self.position = np.zeros(shape=(self.batch_size, self.n_destinations + 1))

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
            step_init = np.zeros([self.batch_size])
            _, _, _, _ = self.step(step_init)

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
            # TODO: Add more info

            # TODO: add assert checks

        else:
            raise ValueError(
                f"Seed: {self.seed} and Folder: {self.folder} is not True. Check that env_instance is not 0."
            )

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
        state_copy = copy.deepcopy(self.state)
        cont_copy = copy.deepcopy(self.containers)

        # determine what and how to pack
        num_actions_1d = self.state.shape[1]
        batch_rot = (action_tuple >= num_actions_1d) * 1
        batch_row = action_tuple - (num_actions_1d) * batch_rot

        self.trajectory.append(
            self.state[np.arange(self.state.shape[0]), batch_row.astype(int), 0]
        )
        self.xy_trajectory.append(
            self.state[np.arange(self.state.shape[0]), batch_row.astype(int), 1:3]
        )
        self.action_list.append(action_tuple)

        self.clean_trajectory = np.array([np.array(xi) for xi in self.trajectory]).T
        self.clean_xy_trajectory = np.array([np.array(xi) for xi in self.xy_trajectory])
        self.clean_action_list = np.array([np.array(xi) for xi in self.action_list]).T
        self.clean_action_list = self.clean_action_list.astype(int)
        # prof = pprofile.Profile()
        # with prof():

        my_jobs = 4
        # my_batch_size = max(1, int(self.batch_size / my_jobs))

        if True:  # parallel or not
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
        else:
            for batch_num in range(self.batch_size):
                (
                    self.state[batch_num],
                    self.rewards[batch_num],
                    self.containers[batch_num],
                    self.done[batch_num],
                    self.travel_cost[batch_num],
                ) = self.batch_pipeline(
                    state_copy[batch_num],
                    cont_copy[batch_num],
                    int(batch_row[batch_num]),
                    int(batch_rot[batch_num]),
                    self.clean_action_list[batch_num],
                )

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

        return self.state, self.rewards, self.done, self.info

    # A few short functions for the Kool code:
    def last(self):
        return self.state, self.rewards, self.done, self.info

    def get_triple(self):
        depot = self.state[:, 0, 1:3]

        # unfold the rotations
        normal_copy = copy.deepcopy(self.state)
        rot_copy = copy.deepcopy(self.state)

        rot_copy = np.delete(
            rot_copy, [7, 8, 9, 13, 15, 16, 17, 19, 20, 21, 22, 23, 24], axis=2
        )
        normal_copy = np.delete(
            normal_copy, [10, 11, 12, 13, 15, 16, 18, 19, 20, 21, 22, 23, 24], axis=2,
        )

        unfolded_state = np.concatenate(
            (normal_copy, rot_copy), axis=1
        )  # remove initial depot entry

        graphs = unfolded_state[:, 1:, 1:3]

        return depot, graphs, unfolded_state[:, 1:, 3:]

    def get_mask(self):
        # flip 1 and 0 for reused code
        # return (
        #    np.reshape(self.state[:, :, 23], (self.batch_size, 1, -1)) * (-1) + 1
        # )  # TODO: extend to include both rots
        unfolded_mask = (
            np.concatenate((self.state[:, :, 23], self.state[:, :, 24]), axis=1) * (-1)
            + 1
        )
        return np.reshape(unfolded_mask, (self.batch_size, 1, -1))

    def all_finished(self):
        return all(self.done)

    def get_costs(self):
        return self.rewards

    def batch_pipeline(
        self, batch_state, batch_containers, batch_row, batch_rot, batch_clean_act
    ):
        """
        Combines all necessary operations for one batch and returns the updated state and container(s)
        """
        # TODO: done skip

        # load
        batch_state, batch_containers, done = self.load_package(
            batch_state, batch_containers, batch_row, batch_rot
        )

        batch_state = self.update_feasibility(
            batch_state, batch_containers[np.max(batch_state[:, 15]) - 1]
        )

        # Reward calculation (different options)
        penalty_factor = 2

        # Sparse (only end of episode currently)
        # Penalizes all undelivered packages with depot to item and back distance
        max_penalty = 0
        for row_i in range(self.state.shape[1]):
            max_penalty += (-penalty_factor) * np.sqrt(
                np.sum((batch_state[row_i, 1:3] - batch_state[0, 1:3]) ** 2)
            )

        trajectory_dist = 0
        delivered = 0

        # correct to row
        num_actions_1d = self.state.shape[1]
        batch_rot = (batch_clean_act >= num_actions_1d) * 1
        batch_row = batch_clean_act - (num_actions_1d) * batch_rot

        for row_i in batch_row:
            delivered += (penalty_factor) * np.sqrt(
                np.sum((batch_state[row_i, 1:3] - batch_state[0, 1:3]) ** 2)
            )

        xys = batch_state[batch_row, 1:3]
        prev_row = xys[0]  # start with depot
        for row_i in range(xys.shape[0]):
            row = xys[row_i]
            trajectory_dist -= np.sqrt(np.sum((row - prev_row) ** 2))
            prev_row = row

        batch_rewards = max_penalty + delivered + trajectory_dist
        batch_rewards = -batch_rewards

        # assert batch_rewards <= 0, "Reward cannot be positive."

        return batch_state, batch_rewards, batch_containers, done, trajectory_dist

    def load_package(self, state, containers, row, rot):
        """
        Updates the container array with the chosen package and packing position
        If action = 0 it returns the vehicle to the depot and starts a new one
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
        prev_row = np.argmax(state[:, 16])
        prev_dest = state[prev_row, 0]
        if (prev_dest != state[row, 0]) and (
            state[row, 16] >= 1
        ):  # at next dest and not first iteration
            state[1:, 22] = state[1:, 22] * (
                1 - (state[1:, 0] == prev_dest) * 1
            )  # prev dest no longer available

        # Nask to only alow same destination packages if available
        same_dest_avail = (state[:, 0] == state[row, 0]) * (state[:, 16] == 0) * 1
        state[row, 22] = 0
        if np.sum(same_dest_avail[1:]) > 1:
            state[1:, 21] = same_dest_avail[1:]
        else:
            state[:, 21] = 1

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
    def update_feasibility(self, state, container):
        """
        Updates state & mask with feasibility (T/F) and the min values for packing (from both rotation possibilities) 
        """

        # only run theoretically feasible rows
        to_do_rows = np.unique(state[:, 19] * state[:, 22] * np.arange(state.shape[0]))
        to_do_rows_r = np.unique(
            state[:, 20] * state[:, 22] * np.arange(state.shape[0])
        )

        # set to 0/-1 for the skipped ones
        state[:, 19:21] = 0
        state[state[:, 22] == 1, 7:13] = -1

        # Swap w and l for 90deg rotation
        r_state = copy.deepcopy(state)
        r_state[:, 4] = state[:, 5]
        r_state[:, 5] = state[:, 4]

        for row in to_do_rows:

            if row == 0:
                continue

            # normal orientation
            state[row, 19], state[row, 7:10], state[row, 17] = self.single_feasibility(
                state[row, :], container, state[row, 17],
            )

        for row in to_do_rows_r:

            if row == 0:
                continue

            # rotatet
            state[row, 20], state[row, 10:13], state[row, 18] = self.single_feasibility(
                r_state[row, :], container, state[row, 18],
            )

        # Combined masks
        state[:, 23] = state[:, 19] * state[:, 21] * state[:, 22]
        state[:, 24] = state[:, 20] * state[:, 21] * state[:, 22]
        # Depot always feasible
        # state[0, 19:] = 1

        return state

    def single_feasibility(self, my_row, container, l_skip):
        """
        Takes package dimension, rotation and current container
        Returns (T/F) if the most top back left location is feasible (# position in container)
        """

        l_skip_update = l_skip

        dim = my_row[3:6]
        frag = my_row[6]

        for l in range(container.shape[2] - l_skip - dim[2] + 1):

            l = l + l_skip

            for w in range(container.shape[1] - dim[1] + 1):

                for h in range(
                    container.shape[0] - dim[0] + 1
                ):  # innermost loop h allows break due to non-floating constraint

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
                                container[
                                    (h - 1) : h, w : (w + dim[1]), l : (l + dim[2])
                                ]
                            )

                            if (frag == 0) and (
                                weakest_support == -1
                            ):  # no non-frag on frag
                                break

                        # minimum supporting area (strict? --> needs to be supportet all the way to the floor) #TODO: check if benchmarks do so too
                        supporting_area = np.sum(
                            np.prod(
                                (
                                    container[
                                        0 : (h - 1), w : (w + dim[1]), l : (l + dim[2]),
                                    ]
                                )
                                ** 2,  # **2 == abs
                                axis=0,
                            )
                        )  # abs for frag -1, non-zero if support all the way along h axis

                        package_area = dim[1] * dim[2]

                        min_support = 0.75  # TODO: move to config file
                        if supporting_area < (package_area * min_support):
                            break

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

        # TODO: Legend with vehicle number (+ env desc) and num packages etc.

        df_items = self.pandas_state(batch_nr)

        df_only_items = df_items.copy()
        df_only_items = df_only_items.drop(0)
        depot_x = df_items.loc[0, "x"]
        depot_y = df_items.loc[0, "y"]
        df_only_items["volume"] = df_items["h"] * df_items["w"] * df_items["l"]
        df_only_items = df_only_items.filter(["dest", "volume"], axis=1)

        df_vols = df_only_items.groupby(by="dest").sum().reset_index()
        df_vols = df_items.merge(df_vols, on="dest")

        fig = plt.figure(figsize=(10, 10))

        # plot delivery locations (=customers)
        plt.scatter(
            df_vols.loc[:, "x"],
            df_vols.loc[:, "y"],
            s=(df_vols.loc[:, "volume"]) / max((df_vols.loc[:, "volume"])) * 100,
            alpha=1,
            marker="o",
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
            for i in range(location_order.shape[0] - 1):
                # flipped as we load in reverse delivery order
                x_0 = location_order[i + 1, 0]
                dx_1 = location_order[i, 0] - x_0
                y_0 = location_order[i + 1, 1]
                dy_1 = location_order[i, 1] - y_0

                plt.arrow(
                    x=x_0,
                    y=y_0,
                    dx=dx_1,
                    dy=dy_1,
                    width=0.5 * scale_arrow,
                    facecolor=facecolors[face_i],
                    edgecolor="none",
                    length_includes_head=True,
                    alpha=0.5,
                )

                if (x_0 == depot_x) and (y_0 == depot_y):
                    face_i += 1
                    if face_i >= len(facecolors):  # loop through list
                        face_i = 0

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
        plt.title(
            "[{}] \n Delivery Locations (Size eq. Order Volume | Square is Depot)".format(
                fig_name
            ),
            size=16,
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
            n_limit = self.pandas_state(0).shape[0]
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

        fig = plt.figure(figsize=(10, 10))

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

        plt.title(
            "[{}] \n Container {} with fragile items in red".format(
                fig_name, n_container + 1
            ),
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
            "Batch {}: {} Veh., {} Items, {} Dest.".format(
                batch,
                self.max_trucks[batch],
                self.state.shape[1] - 1,
                self.n_destinations,
            ),
            False,
            True,
        )
        self.render_container(
            batch, truck, "Batch {}, Veh. {}".format(batch, truck + 1), False, True
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
            "'Step #{}' Batch {}, Veh. {}".format(xy_traj_limit, batch, truck + 1),
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
        print(packed_items)
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
        veh_h = 30
        veh_w = 25
        veh_l = 60

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
            [num_vehicles, veh_h, veh_w, veh_l],
        )

