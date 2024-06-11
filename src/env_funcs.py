# Contains all functions used for the 3L-CVRP environment

import numpy as np
from my_helper_functions import hf
from itertools import repeat


class Vehicle:
    def __init__(
        self,
        min_dim=5,
        max_dim=30,
        customers=3,
        packages=6,
        max_xy=100,
        frag_freq_perc=10,
    ):

        """
        Generates a randomly generated instance based on the Gendreau et al. specifications
        """

        # Truck/Container size
        t_w = 25
        t_l = 60
        t_h = 30

        self.t_w = 25
        self.t_l = 60
        self.t_h = 30
        self.container = np.zeros((t_h, t_w, t_l))
        # TODO: enable multiple containers

        # for visualization
        self.my_sizes = list()
        self.my_positions = list()
        self.my_locations = list()
        self.my_gps = list()

        # Assetion checks
        assert customers <= packages, "Can't have more customers than packages"
        assert min_dim < max_dim, "Min must be smaller max. dimension"
        assert 0 < min(min_dim, customers, packages, max_xy), "Input must be greater 0"
        assert (
            100 >= frag_freq_perc >= 0
        ), "Fragility frequency must be between 0 and 100"

        # Ensure that every customer is represented with at least one package
        customer_ids_unique = (np.arange(customers) + 1).tolist()
        customer_ids = list(customer_ids_unique)

        for i in range(packages - len(customer_ids_unique)):
            customer_ids.append(np.random.choice(customer_ids_unique))
        np.random.shuffle(customer_ids)

        # add depot location
        self.depot = np.array(
            [
                0,
                np.random.randint(0, max_xy, dtype=int),
                np.random.randint(0, max_xy, dtype=int),
            ]
        )
        self.my_gps.append(self.depot[1:])

        # add customer locations and package count
        ids, counts = np.unique(customer_ids, return_counts=True)
        count_dict = {}
        x_dict = {}
        y_dict = {}

        for my_id, my_count in zip(ids, counts):
            count_dict[my_id] = my_count
            x_dict[my_id] = np.random.randint(0, max_xy)
            y_dict[my_id] = np.random.randint(0, max_xy)

        # add package dimensions and fragility
        h = np.random.randint(min_dim, max_dim, packages, dtype=int)
        w = np.random.randint(min_dim, max_dim, packages, dtype=int)
        l = np.random.randint(min_dim, max_dim, packages, dtype=int)
        frag = np.random.randint(0, 100, packages, dtype=int) >= (100 - frag_freq_perc)

        # Match to customer id
        customer_package_count = [count_dict[key] for key in customer_ids]
        customer_x = [x_dict[key] for key in customer_ids]
        customer_y = [y_dict[key] for key in customer_ids]

        self.items = np.array(
            [
                customer_ids,
                customer_package_count,
                h,
                w,
                l,
                frag,
                customer_x,
                customer_y,
            ]
        ).T

        self.items = self.items[self.items[:, 0].argsort()]  # sort by customer

    def feasible(self):
        """
        returns the T/F feasibility of the furthest back, right, lowest location
        """

        my_info = np.array(
            list(map(hf.get_feasible, self.items, repeat(self.container)))
        )

        feasible = my_info[:, 0]
        package_pos = my_info[:, 1]

        return feasible, package_pos

    def load_package(self, dim, frag, position, gps):
        """
        Adds the given package dimensions to the container position
        """

        h = position[0]
        w = position[1]
        leng = position[2]

        if frag == 1:
            self.container[
                h : (h + dim[0]), w : (w + dim[1]), leng : (leng + dim[2])
            ] = -1
        else:
            self.container[
                h : (h + dim[0]), w : (w + dim[1]), leng : (leng + dim[2])
            ] = 1

        # for visualization
        self.my_sizes.append(dim)
        self.my_positions.append(position)
        self.my_locations.append(frag)
        self.my_gps.append(gps)

        pass

    def show_loading(self):

        hf.fill_truck(self.my_sizes, self.my_positions, self.my_locations)

        hf.show_destinations(
            self.items,
            self.depot,
            "TEST NAME",
            location_order=np.array(self.my_gps),
            show_plt=False,
            save_plt=False,
        )
        pass
