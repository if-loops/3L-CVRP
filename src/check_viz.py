for l in range(Truck length - package length):
            
            is l > l_skip ? (skip)

            get h contour along w(l)

            for w in range(Truck width - package width):

                h already full? (skip)

                for h in range(min. contour height(w) to (Truck height - package height)): 

                    Is this corner taken? (skip)

                    if h > 0:

                        Is the fragility constraint fullfilled? (break)

                        Is the minimum supp. area constraint fullfilled? (break)

                    Is LIFO fullfilled? (skip)

                    return Found location, coordinates

            Can we skip this l next time?

        return No location found, l_skip update




        max_poss_h = container.shape[0] - dim[0]
        max_poss_w = container.shape[1] - dim[1]
        h_arange = abs(np.arange(container.shape[0]))[:, np.newaxis] + 100

        for l in prange(container.shape[2] - l_skip - dim[2] + 1):
            # skip already checked l values
            l = l + l_skip

            # skip filled spots
            h_check = abs(container[:max_poss_h, :max_poss_w, l]) * h_arange
            hs_for_l = np.argmin(h_check, axis=0)

            for w in range(max_poss_w + 1):

                h_skip = hs_for_l[w]

                # filled to the achievable top
                # if (container[0, w, l] != 0) & (h_skip == 0):
                #    continue

                for h in range(
                    max_poss_h + 1
                ):  # innermost loop h allows break due to non-floating constraint

                    """
                    The order of the constraints significantly impacts compute time (e.g. LIFO check first increases compute by x6)
                    """

                    if container[h, w, l] != 0:  # already taken
                        continue