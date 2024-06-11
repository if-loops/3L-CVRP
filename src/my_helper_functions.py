import pandas as pd
import numpy as np
import matplotlib.cm as cm
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from itertools import repeat


class hf:
    def get_instance(my_instance_path):
        """
        Takes the filepath to a Gendreau Instance
        and returns a numpy array with node locations and items
        --> Node, number of items,	h,	w,	l,	fragility,	x,	y

        locs, items_expanded = get_instance(my_instance_path)
        """

        with open(my_instance_path) as f:
            scenario = f.readlines()

        reading = 0
        loc_header = "Node - x - y - demand"
        pack_header = "Node - number of items - h - w - l - fragility for each item"

        locations = list()
        items = list()

        for line in scenario:
            # select table
            if loc_header in line:
                reading = 1
            elif pack_header in line:
                reading = 2

            if reading == 1:
                locations.append(
                    line.replace("Node", " Node")
                    .replace("\n", "")
                    .replace("-", " ")
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
        )

    def cuboid_data2(o, size=(1, 1, 1)):
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

    def plotCubeAt2(positions, sizes=None, colors=None, **kwargs):
        # from GitHub
        # if not isinstance(colors,(list,np.ndarray)): colors=["C0"]*len(positions)
        # if not isinstance(sizes,(list,np.ndarray)): sizes=[(1,1,1)]*len(positions)
        g = []
        for p, s, c in zip(positions, sizes, colors):
            g.append(hf.cuboid_data2(p, size=s))

        return Poly3DCollection(
            np.concatenate(g), alpha=0.5, facecolors=np.repeat(colors, 6), **kwargs
        )

    def color_map_color(value, cmap_name="prism", vmin=0, vmax=50):
        # from GitHub
        # norm = plt.Normalize(vmin, vmax)
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = cm.get_cmap(cmap_name)  # PiYG

        rgb = cmap(norm(abs(value)))[
            :3
        ]  # will return rgba, we take only first 3 so we get rgb
        color = mpl.colors.rgb2hex(rgb)
        return color

    def fill_truck(
        my_sizes, my_positions, my_locations
    ):  # locations determines the coloring (can be used for fragility highlighting too)
        """
        Plots packages within a truck sized container, color coded by delivery location
        """
        # TODO: fragility highlighting

        my_sizes = hf.reorder(my_sizes)
        my_positions = hf.reorder(my_positions)
        # truck size
        t_w = 25
        t_l = 60
        t_h = 30

        positions = [(0, 0, 0)] + my_positions
        sizes = [(t_w, t_l, t_h)] + my_sizes

        colors = ["#FFFFFF00"]
        for i in my_locations:
            colors += [hf.color_map_color(value=i, cmap_name="bwr", vmax=1)]

        fig = plt.figure(figsize=(10, 10))
        ax = fig.gca(projection="3d")

        pc = hf.plotCubeAt2(positions, sizes, colors=colors, edgecolor="k")
        ax.add_collection3d(pc)

        ax.set_xlim([0, t_w])
        ax.set_ylim([0, t_l])
        ax.set_zlim([0, t_h])

        ax.set_box_aspect([t_w, t_l, t_h])

        plt.show()

    def show_packages(df, fig_name, show_plt=False, save_plt=False):
        """
        Plots all packages next to each other color coded by location
        """

        """
        Old pandas implementation
        sizes = df.filter(["h", "w", "l"], axis=1).values.tolist()
        locations = df.filter(["Node"], axis=1).T.values.tolist()[0]
        frags = df.filter(["fragility"], axis=1).values.tolist()
        """

        sizes = df[:, 2:5].tolist()
        locations = df[:, 0].tolist()
        frags = df[:, 5].tolist()

        sizes = hf.reorder(sizes)
        colors = []
        for i in locations:
            colors += [
                hf.color_map_color(
                    value=i, cmap_name="prism", vmax=max(25, len(locations))
                )
            ]

        positions = []
        rolling = 0

        for size, frag in zip(sizes, frags):
            max_plot_width = 400
            row_num = int(rolling / max_plot_width % max_plot_width)
            deduct = row_num * max_plot_width

            positions.append(
                [rolling - deduct, (60 - size[1]) + 120 * row_num, min(frag * 30, 30)]
            )
            rolling += size[0] + 1

        fig = plt.figure(figsize=(10, 10))
        ax = fig.gca(projection="3d")

        pc = hf.plotCubeAt2(positions, sizes, colors=colors, edgecolor="k")
        ax.add_collection3d(pc)

        ax.set_xlim([0, min(max_plot_width, rolling)])
        ax.set_ylim([0, 60 + 120 * row_num])
        ax.set_zlim([0, 60])

        ax.set_box_aspect([min(max_plot_width, rolling), 60 + 120 * row_num, 60])
        plt.title(
            "[{}] All packages colored by destination (fragile lifted)".format(fig_name)
        )

        if save_plt:
            fig.savefig("../Data/instance_plots/{}_packages.pdf".format(fig_name))

        if show_plt:
            plt.show()

        pass

    def show_destinations(
        items, depot, fig_name, location_order=None, show_plt=False, save_plt=False
    ):
        df_items = pd.DataFrame(
            items, columns=["Node", "items", "h", "w", "l", "frag", "x", "y"]
        )
        df_only_items = df_items.copy()
        df_only_items["volume"] = df_items["h"] * df_items["w"] * df_items["l"]
        df_only_items = df_only_items.filter(["Node", "volume"], axis=1)

        df_vols = df_only_items.groupby(by="Node").sum().reset_index()
        df_vols = df_items.merge(df_vols, on="Node")

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
        plt.scatter(depot[1], depot[2], s=200, alpha=1, marker="s")

        # add delivery paths
        if location_order is not None:
            for i in range(location_order.shape[0] - 1):
                x_0 = location_order[i, 0]
                dx_1 = location_order[i + 1, 0] - x_0
                y_0 = location_order[i, 1]
                dy_1 = location_order[i + 1, 1] - y_0
                plt.arrow(
                    x=x_0,
                    y=y_0,
                    dx=dx_1,
                    dy=dy_1,
                    width=0.25,
                    facecolor="blue",
                    edgecolor="none",
                    length_includes_head=True,
                )

        plt.xlabel("X", size=16)
        plt.ylabel("y", size=16)
        plt.title(
            "[{}] Delivery Locations (Size eq. Order Volume | Square is Depot)".format(
                fig_name
            ),
            size=16,
        )

        if save_plt:
            fig.savefig("../Data/instance_plots/{}_destinations.pdf".format(fig_name))

        if show_plt:
            plt.show()

        pass

    def reorder(my_sizes):
        """
        Reorder h, w, l to w,l,h (x,y,z)
        """
        new_size = []
        for indv_size in my_sizes:
            new_size.append([indv_size[1], indv_size[2], indv_size[0]])
        return new_size

    def get_item_info(array):

        # TODO: fix unclean code
        try:
            dimensions = array[:, 2:5].tolist()
            locations = array[:, 0].tolist()
            fragilities = array[:, 5].tolist()
            coordinates = array[:, 6:8].tolist()
        except:
            dimensions = array[2:5].tolist()
            locations = array[0].tolist()
            fragilities = array[5].tolist()
            coordinates = array[6:8].tolist()

        return dimensions, locations, fragilities, coordinates

    def get_feasible(my_item, container):
        """
        Takes package dimenstion and current container status
        Returns (T/F) if the most top back left location is feasible
        """

        dim, locs, frag, coo = hf.get_item_info(my_item)

        for l in range(container.shape[2]):
            # check if it can fit in the remaining search space
            if (container.shape[2] - l) < dim[2]:
                break

            # determine upper bound for search space reduction (in case of package > half height)
            if container.shape[0] <= 2 * dim[0]:
                upper_bound = dim[0]
            else:
                upper_bound = -dim[0]

            width_available = any(
                list(
                    map(
                        hf.search_sequence_numpy,
                        container[(dim[0] - 1) : upper_bound, :, l],
                        repeat(np.zeros(dim[1])),
                    )
                )
            )

            # check if it can fit in the remaining search space
            if width_available == False:
                continue

            for w in range(container.shape[1]):

                for h in range(
                    container.shape[0]
                ):  # innermost loop h allows break due to non-floating constraint

                    """
                    The order of the constraints significantly impacts compute time (e.g. LIFO check first increases compute by x6)
                    """

                    # check if the package stays within container walls
                    inside = (
                        ((h + dim[0]) < container.shape[0])
                        * ((w + dim[1]) < container.shape[1])
                        * ((l + dim[2]) < container.shape[2])
                    )

                    if inside != 1:
                        my_h = np.nan
                        my_w = np.nan
                        my_l = np.nan
                        break
                    elif container[h, w, l] != 0:
                        continue  # corner not available

                    # check if LIFO constraint is fulfilled (unload along l axis)
                    if (
                        abs(container[h : (h + dim[0]), w : (w + dim[1]), l:]).sum()
                        != 0
                    ):  # abs for frag -1
                        break

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
                                    container[
                                        0 : (h - 1), w : (w + dim[1]), l : (l + dim[2])
                                    ]
                                ),
                                axis=0,
                            )
                        )  # abs for frag -1, non-zero if support all the way along h axis

                        package_area = dim[1] * dim[2]

                        min_support = 0.75  # TODO: move to config file
                        if supporting_area < (package_area * min_support):
                            break

                    my_h = h
                    my_w = w
                    my_l = l
                    return True, [my_h, my_w, my_l]

        return False, [np.nan, np.nan, np.nan]

    def load_package(dim, frag, position, container):
        """
        Adds the given package dimensions to the container position
        (Places via furthest back corner position)
        """
        h = position[0]
        w = position[1]
        l = position[2]

        if frag == 1:
            container[h : (h + dim[0]), w : (w + dim[1]), l : (l + dim[2])] = -1
        else:
            container[h : (h + dim[0]), w : (w + dim[1]), l : (l + dim[2])] = 1

        return container

    def search_sequence_numpy(arr, seq):
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

    def generate_instance(
        min_dim=5, max_dim=30, customers=3, packages=6, max_xy=100, frag_freq_perc=10
    ):

        """
        Generates a randomly generated instance based on the Gendreau et al. specifications
        """

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
        depot = np.array(
            [
                0,
                np.random.randint(0, max_xy, dtype=int),
                np.random.randint(0, max_xy, dtype=int),
            ]
        )

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

        my_items = np.array(
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

        my_items = my_items[my_items[:, 0].argsort()]  # sort by customer

        return my_items, depot
