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
from functools import lru_cache


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
        packages_perc_of_av_vol: int,
        frag_freq_prob: int,
        test_set: int,
        folder: str,
        foldername: str,
        batch_size: int,
    ):
        """Inits VRPEnv with n_destinations.
        ASSUMES THERE IS ONLY ONE DEPOT.
        """

        super(TDCVRP, self).__init__()

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

        self._generate_env()

    def _generate_env(self) -> None:
        if self.folder:

            # homogeneous tuck containers

            # Create master array of inventory to ship (TODO: different # of packages per batch)
            items_per_dest = np.random.randint(1, 4, self.n_destinations + 1)
            items_per_dest[0] = 1  # depot

            # TODO: change table to have feasible for both rots and locs -> also change feasibility check
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
                "feas",
                "feas_r",
            ]
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
                    self.state[:, ind, 6] = np.random.choice(2, (self.batch_size), p=p)

                    ind += 1

            self.state = self.state.astype(int)
            # Depot location(s) removal of package info
            self.state[:, 0, 3:] = 0
            self.done = np.full((self.batch_size, 1), False, dtype=bool)

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

            # Keep track of positions (max is latest)
            self.position = np.zeros(shape=(self.batch_size, self.n_destinations + 1))

            # Set depot as start
            self.position[:, 0] = 1

            # Create Mask(s)
            # We use multiple masks to keep track of what is feasible and what has already been loaded

            # Tracks current feasibility (1 = feasible to load)
            self.feasibility_mask = np.ones(
                shape=(self.batch_size, 2, self.state.shape[1],)
            )  # two dimensions for rotations 0 = original orientation

            self.feasibility_mask_either = np.ones(
                shape=(self.batch_size, self.state.shape[1],)
            )

            # Tack if packages at same destination are available
            self.same_dest_mask = np.ones(shape=(self.batch_size, self.state.shape[1],))

            # Tracks which packages are not yet loaded and still at the depot (1 = available)
            self.depot_mask = np.ones(shape=(self.batch_size, self.state.shape[1],))
            # Combines the above masks (#TODO: reduce to a single mask; currently easier with multiple for debugging)
            self.mask = np.ones(shape=(self.batch_size, self.state.shape[1],))

            # Update Masks
            self.update_feasibility()

            # Update distances to depot in state
            self.update_distance()

            # The agent cannot take an action to go to the depot as it is already there.
            self.mask[:, 0] = 0

            # Spaces
            self.observation_space = spaces.Box(
                low=0, high=9, shape=(self.n_destinations + 1, 3), dtype=np.float64
            )
            # second dim for rotation
            self.action_space = gym.spaces.Tuple(
                (spaces.Discrete(self.state.shape[1]), spaces.Discrete(2),)
            )

            self.reward: Union[float, None] = np.full((self.batch_size, 1), None)
            self.info: Dict = {}

            self.counter = 0
            self.trajectory: List[
                int
            ] = []  # The agent always starts at the depot (i.e. 0)
            self.xy_trajectory: List[int] = []
            self.trajectory.append(
                self.state[np.arange(self.batch_size), 0, 0]
            )  # add start
            self.xy_trajectory.append(
                self.state[np.arange(self.batch_size), 0, 1:3]
            )  # add start

            # Save state for validation
            if self.folder == "validation":
                np.save(
                    os.path.join(self.foldername, f"test_set_{self.test_set}"),
                    self.state,
                )

            # info
            self.info["max_trucks"] = self.max_trucks
            self.info["items_per_dest"] = items_per_dest
            self.info["state_columns"] = self.state_columns

            assert (
                np.sum(self.position) == self.batch_size
            ), f"The agent can only be in one place, got locations: {np.sum(self.position)} instead."
            assert (
                np.count_nonzero(self.mask == 0) == self.batch_size
            ), f"The agent can only not go to the depot at the beginning. Got: {np.count_nonzero(self.mask)}. \n{self.mask}"

        else:
            raise ValueError(
                f"Seed: {self.seed} and Folder: {self.folder} is not True. Check that env_instance is not 0."
            )

    def single_feasibility(
        self,
        my_row,
        curr_container,
        pack_available,
        prev_feasible,
        rotating,
        item_row,
        batch_num,
        same_skip,
        is_same,
    ):
        """
        Takes package dimension (and current container)
        Returns (T/F) if the most top back left location is feasible (# position in container)
        """

        # skip to an l where width is feasible
        # skip to an l where width is feasible
        if curr_container[0, 0, 0] == 0:  # empty container
            return True, 0, 0, 0, 0
        elif (pack_available == 0) or (same_skip and (is_same == 0)):
            # Use state and mask info to skip already packed parcels and depot
            # only allow packages from same location (if available)
            if rotating == 0:
                return (
                    False,
                    my_row[7],
                    my_row[8],
                    my_row[9],
                    my_row[17],
                )  # keep the packed position
            elif rotating == 1:
                return False, my_row[10], my_row[11], my_row[12], my_row[18]
        elif my_row[0] == 0:  # depot skip
            return True, -1, -1, -1, 0
        elif rotating == 0:
            l_skip = my_row[17]
            l_skip_update = my_row[17]
        elif rotating == 1:
            l_skip = my_row[18]
            l_skip_update = my_row[18]

        any_lifo = 1

        dim = my_row[3:6]
        frag = my_row[6]

        # select current container
        container = curr_container

        for l in range(container.shape[2] - l_skip - dim[2] + 1):

            l = l + l_skip
            # check if it can fit in the remaining search space
            # if (container.shape[2] - l) < dim[2]:
            #   break

            # determine upper bound for search space reduction (in case of package > half height)
            # if container.shape[0] <= 2 * dim[0]:
            #    upper_bound = dim[0]
            # else:
            #    upper_bound = -dim[0]

            if any_lifo == 0:
                l_skip_update = l
                # self.state[
                #    batch_num, item_row, 17
                # ] = l  # for skips (never start searching earlier than this)
            any_lifo = 0  # reset for new l
            """
            # takes longer than without (tested with %prof)
            width_available = any(
                list(
                    map(
                        self.search_sequence_numpy,
                        container[(dim[0] - 1) : upper_bound, :, l],
                        repeat(np.zeros(dim[1])),
                    )
                )
            )
            
            # check if it can fit in the remaining search space
            if width_available == False:
                continue
            """

            for w in range(container.shape[1] - dim[1] + 1):

                for h in range(
                    container.shape[0] - dim[0] + 1
                ):  # innermost loop h allows break due to non-floating constraint

                    """
                    The order of the constraints significantly impacts compute time (e.g. LIFO check first increases compute by x6)
                    """

                    # check if the package stays within container walls
                    # inside = (
                    #    ((h + dim[0]) < container.shape[0])
                    #    * ((w + dim[1]) < container.shape[1])
                    #    * ((l + dim[2]) < container.shape[2])
                    # )

                    # if inside != 1:
                    #    my_h = np.nan
                    #    my_w = np.nan
                    #    my_l = np.nan
                    #    break
                    if (
                        np.sum(
                            container[h : h + dim[0], w : w + dim[1], l : l + dim[1]]
                        )
                        != 0
                    ):
                        continue  # corner not available

                    # placemnet behing h>0 is quicker. Thus only run it for l_skip until lifo valid found for current l
                    if any_lifo == 0:
                        # check if LIFO constraint is fulfilled (unload along l axis)
                        if (
                            abs(container[h : (h + dim[0]), w : (w + dim[1]), l:]).sum()
                            != 0
                        ):  # abs for frag -1
                            any_lifo += 1  # indicates that at least one possible future lifo exists
                            continue

                    # check if package is supported (min. 75% and fragility)
                    if h > 0:

                        # fragility constraint (only fragile allowed on fragile)
                        weakest_support = np.amin(
                            container[(h - 1), w : (w + dim[1]), l : (l + dim[2])]
                        )

                        if (frag == 0) and (
                            weakest_support == -1
                        ):  # no non-frag on frag
                            break

                        # minimum supporting area (strict --> needs to be supportet all the way to the floor) #TODO: check if benchmarks do so too
                        supporting_area = np.sum(
                            np.prod(
                                abs(
                                    container[0:(h), w : (w + dim[1]), l : (l + dim[2])]
                                ),
                                axis=0,
                            )
                        )  # abs for frag -1, non-zero if support all the way along h axis

                        package_area = dim[1] * dim[2]

                        min_support = 0.75  # TODO: move to config file
                        if supporting_area < (package_area * min_support):
                            break

                    # check if LIFO constraint is fulfilled (unload along l axis)
                    if (
                        abs(container[h : (h + dim[0]), w : (w + dim[1]), l:]).sum()
                        != 0
                    ):  # abs for frag -1
                        continue

                    my_h = h
                    my_w = w
                    my_l = l
                    return True, my_h, my_w, my_l, l_skip_update

        return False, -1, -1, -1, l_skip_update

    def search_sequence_numpy(self, arr, seq):
        """ Find sequence in an array using NumPy only.
        https://stackoverflow.com/questions/36522220/searching-a-sequence-in-a-numpy-array

        Parameters
        ----------    
        arr    : input 1D array
        seq    : input 1D array

        Output
        ------    
        Output : 1D Array of indices in the input array that satisfy the 
        matching of input sequence in the input array.
        In case of no match, an empty list is returned.
        """

        # Store sizes of input array and sequence
        Na, Nseq = arr.size, seq.size

        # Range of sequence
        r_seq = np.arange(Nseq)

        # Create a 2D array of sliding indices across the entire length of input array.
        # Match up with the input sequence & get the matching starting indices.

        M = (arr[np.arange(Na - Nseq + 1)[:, None] + r_seq] == seq).all(1)

        # Get the range of those indices as final output
        if M.any() > 0:
            return True  # np.where(np.convolve(M,np.ones((Nseq),dtype=int))>0)[0]
        else:
            return False  # []         # No match found

    def get_rot_feas(self, batch, action):
        """ Returns the rotation feasibility mask for a specific action """
        return self.feasibility_mask[batch, :, int(action)]

    def update_feasibility(self):
        """
        Updates state & mask with feasibility (T/F) and the min values for packing (from both rotation possibilities) 
        """

        # Swap w and l for 90deg rotation
        r_state = copy.deepcopy(self.state)
        r_state[:, :, 4] = self.state[:, :, 5]
        r_state[:, :, 5] = self.state[:, :, 4]
        self.r_state = r_state

        # Iterate over batches
        # print(self.n_container[0])

        self.par_info = Parallel(n_jobs=-1)(
            delayed(self.parallel_feas)(batch_num)
            for batch_num in range(self.batch_size)
        )

        # since joblib can't write to self, we need to manually redo it
        self.feas_to_self(self.par_info)

        # Combine masks
        self.mask = self.depot_mask * self.feasibility_mask_either * self.same_dest_mask
        self.mask[np.arange(self.batch_size), 0] = 1  # depot always feasible

        # restrict depot to only one rotation possibility
        self.feasibility_mask[:, :, 0] = 1
        self.feasibility_mask[:, 1, 0] = 0

        pass

    def parallel_feas(self, i):
        # for i in range(self.batch_size):

        # only allow loading of packages of the same location (if available)
        # locate current dest
        curr_row = np.argmax(self.state[i, :, 16])
        curr_dest = self.state[i, curr_row, 0]

        # do this outside of map to only call once per batch
        av_same_mask = (self.state[i, :, 0] == curr_dest) * self.depot_mask[i]
        av_same_dest = np.sum(av_same_mask)

        same_skip = True
        if (curr_dest == 0) or (av_same_dest == 0):
            same_skip = False

        # get feasibility information
        my_info = np.array(
            list(
                map(
                    self.single_feasibility,
                    self.state[i],
                    repeat(self.containers[i, int(self.n_container[i])]),
                    self.depot_mask[i],
                    self.feasibility_mask[i, 0, :],
                    repeat(0),  # rotating
                    np.arange(self.state[i].shape[0]),  # item row
                    repeat(i),  # batch number
                    repeat(same_skip),
                    av_same_mask,
                )
            ),
            dtype=object,
        )
        """
        my_info_par = Parallel(n_jobs=-1)(
            delayed(self.single_feasibility)(
                self.state,
                self.containers[:, self.n_container[:].astype(int)],
                self.depot_mask,
                self.feasibility_mask[:, 0, :],
                0,  # rotating
                np.arange(self.state.shape[0]),  # item row
                i,  # batch number
                same_skip,
                av_same_mask,
            )
            for i in range(self.batch_size)
        )
        """
        my_info_r = np.array(
            list(
                map(
                    self.single_feasibility,
                    self.r_state[i],
                    repeat(self.containers[i, int(self.n_container[i])]),
                    self.depot_mask[i],
                    self.feasibility_mask[i, 1, :],
                    repeat(1),  # rotating
                    np.arange(self.state[i].shape[0]),  # item row
                    repeat(i),  # batch number
                    repeat(same_skip),
                    av_same_mask,
                )
            ),
            dtype=object,
        )

        return [my_info, my_info_r]

    def feas_to_self(self, my_info):

        for i in range(self.batch_size):

            # Update masks and state
            self.state[i, :, 7:10] = my_info[i][0][
                :, 1:4
            ]  # index: batch, normal or rot, row, col
            self.state[i, :, 10:13] = my_info[i][1][:, 1:4]
            self.state[i, :, 17] = my_info[i][0][:, 4]
            self.state[i, :, 18] = my_info[i][1][:, 4]
            self.feasibility_mask[i, 0, :] = my_info[i][0][:, 0] * 1
            self.feasibility_mask[i, 1, :] = my_info[i][1][:, 0] * 1

            # Masking to enforce that all packages from the same location need to be packed together
            curr_pos = np.argmax(self.position[i])
            self.same_dest_mask[i] = (curr_pos == self.state[i, :, 0].T).T * 1

            # check if at least one rotation is feasible
            self.feasibility_mask_either[i, :] = (
                self.feasibility_mask[i, 0, :] + self.feasibility_mask[i, 1, :] * 1
            ) >= 1

            # check if possible (available and placeable)
            same_comb_mask = (
                self.same_dest_mask * self.depot_mask * self.feasibility_mask_either
            )
            av_same = np.sum(same_comb_mask[:, 1:], axis=1)
            same_feasible = (av_same >= 1) * 1

            for i_batch in range(self.batch_size):
                self.same_dest_mask[i_batch, :] = (
                    self.same_dest_mask[i_batch, :] * same_feasible[i_batch]
                    + 1
                    - same_feasible[i_batch]
                )

        pass

    def update_distance(self):
        """
        Updates state with distance of all packages to last delivery location
        """
        for batch in range(self.batch_size):

            curr_pos = np.argmax(self.position, axis=1)[batch]

            for ind in range(self.state.shape[1]):
                # distance to previous (= depot for start)
                self.state[batch, ind, 14] = (
                    np.sqrt(
                        (self.state[batch, curr_pos, 1] - self.state[batch, ind, 1])
                        ** 2
                        + (self.state[batch, curr_pos, 2] - self.state[batch, ind, 2])
                        ** 2
                    )
                    * 10
                )  # *10 to add extra percision to integer

    def pandas_state(self, batch_nr):
        """
        For easier inspection
        """
        import pandas as pd

        pd_df = pd.DataFrame(self.state[batch_nr], columns=self.state_columns)
        return pd_df

    def load_package(self, action_rows, rotation):
        """
        Updates the container array with the chosen package and packing position
        If action = 0 it returns the vehicle to the depot and starts a new one
        """

        for my_batch in range(self.batch_size):

            if self.done[my_batch]:
                continue

            # pick the row corresponding to the chosen action per batch
            row = action_rows[my_batch]
            rot = rotation[my_batch]

            # truck
            my_truck = self.n_container[my_batch].astype(np.int32)

            # package dimensions and fragility
            pack_h = self.state[my_batch, row, 3].astype(np.int32)
            pack_w = self.state[my_batch, row, 4].astype(np.int32)
            pack_l = self.state[my_batch, row, 5].astype(np.int32)
            pack_frag = self.state[my_batch, row, 6].astype(np.int32)

            # location in truck to be packed
            pos_h = self.state[my_batch, row, 7].astype(np.int32)
            pos_w = self.state[my_batch, row, 8].astype(np.int32)
            pos_l = self.state[my_batch, row, 9].astype(np.int32)

            r_pos_h = self.state[my_batch, row, 10].astype(np.int32)
            r_pos_w = self.state[my_batch, row, 11].astype(np.int32)
            r_pos_l = self.state[my_batch, row, 12].astype(np.int32)

            # Mask same destination packages that were not packed after leaving destination
            if self.trajectory[-1][my_batch] != self.state[my_batch, row, 0]:
                used_dest = self.state[my_batch, :, 0][
                    self.depot_mask[my_batch].astype(int) == 0
                ]
                same_constraint = np.ones(self.state.shape[1])
                for dest_i in used_dest:
                    same_constraint *= self.state[my_batch, :, 0] != dest_i

                self.depot_mask[my_batch] *= same_constraint

            # update container
            if row == 0:
                self.n_container[my_batch] += 1

                if self.n_container[my_batch] >= self.max_trucks[my_batch]:
                    self.done[my_batch] = True  # used all available vehicles
                    self.n_container[my_batch] -= 1  # otherwise overflow
                    self.depot_mask[my_batch, :] = 0  # no more loading possible
                else:
                    # reset width check skip in feasibility check
                    self.state[my_batch, :, 17] = 0
                    # in else to keep info in state for inspection
            else:

                if rot == 0:
                    self.containers[
                        my_batch,
                        my_truck,
                        pos_h : (pos_h + pack_h),
                        pos_w : (pos_w + pack_w),
                        pos_l : (pos_l + pack_l),
                    ] = (
                        1 - 2 * pack_frag
                    )  # 1 fo non-fragile, -1 for fragile, 0 empty
                elif rot == 1:
                    self.state[my_batch, row, 13] = 1
                    self.containers[
                        my_batch,
                        my_truck,
                        r_pos_h : (r_pos_h + pack_h),
                        r_pos_w : (r_pos_w + pack_l),  # swapped l and w for rotation
                        r_pos_l : (r_pos_l + pack_w),
                    ] = (
                        1 - 2 * pack_frag
                    )  # 1 fo non-fragile, -1 for fragile, 0 empty
                else:
                    raise ValueError(
                        "{} not a valid rotation (only 0 or 1)".format(rot)
                    )

            # update state
            self.state[my_batch, row, 7:10] = [pos_h, pos_w, pos_l]
            self.state[my_batch, row, 15] = (
                my_truck + 1
            )  # +1 to have non-loaded packages with 0
            # update loading sequence helper column
            self.state[my_batch, row, 16] = np.max(self.state[my_batch, :, 16]) + 1

            # update masking (what has been loaded so far)
            self.depot_mask[np.arange(self.batch_size), action_rows] = 0
            self.depot_mask[np.arange(self.batch_size), 0] = 1  # depot always feasible

            # Update masks to reflect unpackable (set positions to -1)
            for row in range(self.state[my_batch].shape[0] - 1):
                not_in_veh = self.state[my_batch, row + 1, 15] == 0
                if not_in_veh:
                    # self.state[my_batch, row + 1, 7:13] = -1
                    pass

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
        actions = action_tuple[:, 0]
        rots = action_tuple[:, 1]

        # load package into truck
        self.load_package(actions, rots)

        # update position
        self.position = np.zeros(shape=(self.batch_size, self.n_destinations + 1))
        self.position[np.arange(self.batch_size), self.state[:, actions, 0]] = 1

        # update mask
        self.update_feasibility()

        # Update package distances to current location in state
        self.update_distance()

        # record route
        self.trajectory.append(self.state[np.arange(self.batch_size), actions, 0])
        self.xy_trajectory.append(self.state[np.arange(self.batch_size), actions, 1:3])
        # maybe add prev location and curr location for easier calc of reward
        position = self.position.astype(np.bool)

        self.counter += 1

        # Rewards
        self.rewards = np.full((self.batch_size,), None)
        for batch in range(self.batch_size):

            # departure and arrival location distance (neg) is reward
            dep_locs = np.zeros([len(self.trajectory) - 1, 2])
            arr_locs = np.zeros([len(self.trajectory) - 1, 2])

            i = 0
            for trip in range(len(self.trajectory) - 1):
                dep_locs[i, :] = self.destinations[batch, self.trajectory[trip][batch]]
                arr_locs[i, :] = self.destinations[
                    batch, self.trajectory[trip + 1][batch]
                ]
                i += 1

            self.rewards[batch] = np.linalg.norm(
                dep_locs[-2:, :] - arr_locs[-2:, :]
            )  # only last step

            # check if unpacked remaining after done
            if (self.done[batch] == True) and (np.min(self.state[batch, :, -2]) == 0):
                # penalize undelivered packages (average of not packed)
                # TODO: Change to something better
                self.rewards[batch] = -(
                    np.sum(self.state[batch, self.state[batch, :, -2] == 0, 14]) * 2
                ) / np.sum(
                    self.state[batch, :, -2] == 0
                )  # average of not packed with way back and forth

        self.info["mask"] = self.mask
        self.clean_trajectory = np.array([np.array(xi) for xi in self.trajectory]).T
        self.clean_xy_trajectory = np.array(
            [(np.array(xi)) for xi in self.xy_trajectory]
        )

        return self.state, self.rewards, self.done, self.info

    def reset(
        self, env_instance: int = None, folder: str = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        self.close()
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
                    width=0.5,
                    facecolor=facecolors[face_i],
                    edgecolor="none",
                    length_includes_head=True,
                )

                if (x_0 == depot_x) and (y_0 == depot_y):
                    face_i += 1
                    if face_i >= len(facecolors):  # loop through list
                        face_i = 0

        plt.xlim(0, 100)
        plt.ylim(0, 100)
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
            positions[0:2], sizes[0:2], colors=colors[0:2], my_alpha=0.1, edgecolor="k"
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

        g = []
        for p, s, c in zip(positions, sizes, colors):
            g.append(self.cuboid_data2(p, size=s))

        return Poly3DCollection(
            np.concatenate(g), alpha=my_alpha, facecolors=np.repeat(colors, 6), **kwargs
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

        packed_items = self.pandas_state(batch).sort_values(by=["seq_nr"])[
            self.pandas_state(0).seq_nr != 0
        ]

        xy_traj_limit = packed_items.iloc[
            n_limit, 16
        ]  # handles the sequence skips of goin to the depot

        truck = packed_items.iloc[n_limit, 15] - 1

        image_vrp = self.render_vrp(
            batch,
            "'Step #{}' Batch {}: {} Veh., {} Items, {} Dest.".format(
                xy_traj_limit - 1,
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

    def make_gif(self, batch=0, my_title="___"):
        """
        Save a gif of the vrp and container loading process
        To combine two gifs side by side: https://ezgif.com/combine
        """

        import imageio

        max_i = (
            self.pandas_state(batch)
            .sort_values(by=["seq_nr"])[self.pandas_state(batch).seq_nr != 0]
            .shape[0]
        )

        # workaround to have equal gif length (no package to load at last depot trip)
        i_range_load = np.arange(max_i + 1)
        i_range_load_vrp = np.arange(max_i + 1)
        i_range_load[max_i] = max_i - 1
        i_range_load_vrp[max_i] = 0

        kwargs_write = {"fps": 1.0, "quantizer": "nq"}
        imageio.mimsave(
            "../src/instance_gifs/{}_loading.gif".format(my_title),
            [self.disp_step_by_step(batch, i, "cont") for i in i_range_load],
            fps=1,
        )

        imageio.mimsave(
            "../src/instance_gifs/{}_vrp.gif".format(my_title),
            [self.disp_step_by_step(batch, i, "vrp") for i in i_range_load_vrp],
            fps=1,
        )

        return "Done"
