from cda import Ellipse
import numpy as np
import itertools
import math


class Catalog:
    def __init__(self, catalog_file, catalog_height, catalog_fov, catalog_size):
        self.catalog_size = catalog_size
        self.catalog_file = catalog_file
        self.catalog_height = catalog_height
        self.catalog_fov = catalog_fov
        self.catalog = None

    def _pixel_dimensions(self):
        """
        Determine the size of one pixel

        :return: pixel size in meters
        """
        dimensions = 2 * self.catalog_height * math.tan(math.radians(self.catalog_fov / 2))
        return dimensions / self.catalog_size

    def initialize_catalog(self):
        """
        Initialize the crater catalog by generation Ellipse objects

        :return: crater catalog
        """
        catalog = []
        for c_id, x, y, r in self.catalog_file:
            ellipse = Ellipse(int(c_id), [x, y], r / 2, r / 2)
            catalog.append(ellipse)

        catalog = np.array(catalog)
        self.catalog = catalog
        return catalog

    def localize(self, crater_matches):
        """
        Perform a localization for a collection of crater matches
        TODO: Input sanitation

        :param crater_matches: collection of crater matches between input image and catalog
        :return: altitude in m and x,y translation in m
        """
        # Determine scale change
        res = []
        for match in crater_matches:
            sc_1 = match[0].h_r / match[1].h_r
            sc_2 = match[0].v_r / match[1].v_r

            res.append((sc_1 + sc_2) / 2)
        d_s = np.mean(res)

        # Determine translation
        res_d = []
        for match in crater_matches:
            d_x = match[1].center[0] - match[0].center[0] / d_s
            d_y = match[1].center[1] - match[0].center[1] / d_s
            res_d.append([d_x, d_y])
        d_t = np.mean(res_d, axis=0)
        alt = self.catalog_height / d_s
        tr = d_t * self._pixel_dimensions()
        return alt, tr


def invariants(crater_list):
    """
    Determine the invariants for each combination of two craters in a collection of craters ellipses

    :param crater_list: collection of crater ellipses
    :return: collection of crater invariants, each row contains the craters that were matched and their invariants
    """
    pairs = list(itertools.combinations(crater_list, 2))
    t = []
    for pair in pairs:
        invs = pair[0].invariant(pair[1])
        new_item = [(pair[0], pair[1]), invs]
        t.append(new_item)
    return np.array(t)


def match(crater_catalog, fitted_ellipses):
    """
    Perform a matching between a collection of ellipses and a catalog of craters (which is also a collection of ellipses)

    :param crater_catalog:
    :param fitted_ellipses:
    :return: for each crater in fitted_ellipses, the best match in the catalog
    """
    e_l, e_h = 0.99, 1.01
    # 1 - Get invariants
    T_catalog = invariants(crater_catalog)  # T_a: m(m-1)/2
    T_fitted = invariants(fitted_ellipses)  # T_b: n(n-1)/2

    # 2 - Preallocate Mapping matrix
    mapping = np.zeros((fitted_ellipses.shape[0], crater_catalog.shape[0]))  # (m, n)

    # 3 - Compare every catalog crater pair with every pair in fitted_ellipses
    # TODO Come up with a nice numpy idiom for this monstrosity
    count = 0
    for T_f in T_fitted:
        for T_c in T_catalog:
            inv_f = T_f[1]
            inv_c = T_c[1]
            e_f1 = T_f[0][0]
            e_f2 = T_f[0][1]
            e_c1 = T_c[0][0]
            e_c2 = T_c[0][1]

            c_1 = e_l < inv_f[0] / inv_c[0] < e_h
            c_2 = e_l < inv_f[1] / inv_c[1] < e_h
            c_3 = e_l < inv_f[0] / inv_c[1] < e_h
            c_4 = e_l < inv_f[1] / inv_c[0] < e_h

            if c_1 and c_2:
                count += 1
                mapping[e_f1.c_id, e_c1.c_id] += inv_f[0] / inv_c[0]
                mapping[e_f2.c_id, e_c2.c_id] += inv_f[1] / inv_c[1]

            if c_3 and c_4:
                count += 1
                mapping[e_f1.c_id, e_c2.c_id] += inv_f[0] / inv_c[1]
                mapping[e_f2.c_id, e_c1.c_id] += inv_f[1] / inv_c[0]

    res = []
    ind = 0
    for mp in mapping:
        res.append((fitted_ellipses[ind], crater_catalog[np.argmax(mp)]))
        ind += 1
    return res
