import numpy as np
import math
from scipy.spatial import distance

import cv2
import sys

from motus import config


class Ellipse:
    def __init__(self, c_id, center, h_r, v_r, angle=0.0):
        self.c_id = c_id
        self.center = center
        self.h_r = h_r
        self.v_r = v_r
        self.angle = angle
        self.area = 2 * math.pi * h_r * v_r

    def eccentricity(self):
        a = max(self.h_r, self.v_r)
        b = min(self.h_r, self.v_r)
        return math.sqrt(1 - math.pow(b / a, 2))

    def quadratic_form(self):
        """
        Calculate the quadratic form. Currently suspected broken and not used

        :return: 6 coefficients
        """
        e_a = max(self.h_r, self.v_r)  # semimajor axis
        e_b = min(self.h_r, self.v_r)  # semiminor axis
        t = self.angle  # theta
        x_c = self.center[0]  # xcenter
        y_c = self.center[1]  # ycenter

        a = (e_a ** 2) * (math.sin(t)) ** 2 + (e_b ** 2) * (math.cos(t) ** 2)
        b = 2 * (e_b ** 2 - e_a ** 2) * math.sin(t) * math.cos(t)
        c = (e_a ** 2) * (math.cos(t) ** 2) + (e_b ** 2) * (math.sin(t) ** 2)
        d = -2 * a * x_c - b * y_c
        e = -b * x_c - 2 * c * y_c
        f = a * (x_c ** 2) + b * x_c * y_c + c * (y_c ** 2) - (e_a ** 2) * (e_b ** 2)
        return [a, b, c, d, e, f]

    def coefficient_matrix(self):
        a, b, c, d, e, f = self.quadratic_form()
        return np.array([[a, b / 2, d / 2], [b / 2, c, e / 2], [d / 2, e / 2, f]])

    def invariant(self, ellipse):
        """
        Calculate angle invariants. Note: only works for images that are not rotated nor have any perspective changes

        :param ellipse:
        :return: invariants
        """
        c_xa = self.center[0] - ellipse.center[0]
        c_ya = self.center[1] - ellipse.center[1]
        c_xb = ellipse.center[0] - self.center[0]
        c_yb = ellipse.center[1] - self.center[1]

        ang_ab = np.arctan2(c_ya, c_xa)
        ang_ba = np.arctan2(c_yb, c_xb)

        return ang_ab, ang_ba


def merit(c_a, c_b, size_a, size_b):
    """
    Compute merit score between two craters A and B

    :param c_a:
    :param c_b:
    :param size_a:
    :param size_b:
    :return: merit score
    """
    # M_A angle illumination direction & centroids connecting line
    c_n = np.subtract(c_b, c_a)

    # Deviation from illumination angle
    # TODO: Improve angle comparison
    ang = np.arctan2(c_n[1], c_n[0])
    if ang < 0:
        ang += 2 * np.pi
    m_a = 1 - (ang - config.ILLUMINATION_ANGLE) / (2 * np.pi)

    # Relative distance
    m_d = math.pow(distance.euclidean(c_a, c_b), 2)

    # Relative size
    m_s = 1 - float(min(size_a, size_b)) / float(max(size_a, size_b))
    return m_a * m_d * m_s


def mser_cv(img):
    """
    Find lit and dark areas using CV2 MSER

    :param img: grayscale input image
    :return: lit, dark areas
    """
    # Configure CV2.MSER
    mser = cv2.MSER_create(config.MSER_DELTA, config.MSER_MIN_AREA, config.MSER_MAX_AREA, config.MSER_MAX_VARIATION,
                           config.MSER_MIN_DIVERSITY)
    # Only process one area at a time to be able to distinguish between lit and dark areas
    mser.setPass2Only(True)

    # Detect the lit and dark areas separately
    mser_areas_light, _ = mser.detectRegions(img)
    mser_areas_dark, _ = mser.detectRegions(255 - img)

    # For some reason the MSER here returns not just the largest extremal
    # region, but smaller results as well. Therefore just gather all pixels
    # that have a value > 0 and apply the union-find strategy to find the
    # connected sets again.
    overlap_light = np.zeros(img.size).reshape(img.shape[:2])
    overlap_dark = overlap_light.copy()

    for area in mser_areas_light:
        xs, ys = zip(*area)
        overlap_light[ys, xs] = 255

    for area in mser_areas_dark:
        xs, ys = zip(*area)
        overlap_dark[ys, xs] = 255

    return overlap_light, overlap_dark


def rois(a, b):
    """
    Find Regions of Interest (ROIs) by applying the merit function to combinations of lit and dark crater areas

    :param a:
    :param b:
    :return: ROIs
    """
    regions = []
    for region_a in a:
        min_merit = sys.maxsize
        for region_b in b:
            # Apply the merit function to find a merit value.
            fitted = merit(region_a.centroid, region_b.centroid, region_a.area, region_b.area)
            if fitted < min_merit:
                min_merit = fitted
                region_b_match = region_b

        # TODO No hardcoded merit cut-off
        if min_merit < 500:
            # Matches with a value that is too low are discarded.
            regions.append([region_a, region_b_match])
    return regions


def fit(roi_collection):
    """
    Fit ellipses to a collection of ROIs

    :param roi_collection: a collection of ROIs. Each ROI is a collection of coordinates belonging to that ROI
    :return: collection of fitted ellipses
    """
    # For each ROI, fit an ellipse
    ellipses = []
    ind = 0
    for r_a, r_b in roi_collection:
        roi = np.concatenate((r_a.coords, r_b.coords), axis=0)
        ellipse = fit_ellipse(roi, ind)
        if ellipse.eccentricity() < 0.9:
            ellipses.append(ellipse)
            ind += 1

    return np.array(ellipses)


def fit_ellipse(roi, c_id):
    """
    Perform an ellipse fit on a collection of coordinates

    @param roi: coordinates
    @param c_id: ellipse id
    @return: Ellipse object
    """
    # Reformat data a little bit
    roi[:, [0, 1]] = roi[:, [1, 0]]

    # Normalize data
    mu = np.asarray(roi).mean(axis=0)
    roi = roi - mu

    # Perform singular value decomposition
    eigenvectors, eigenvalues, V = np.linalg.svd(roi.T, full_matrices=False)

    # Project the data along the x- and y-axes
    projected_data = np.dot(roi, eigenvectors)
    xns, yns = zip(*projected_data)

    # Find the outer edges of the data to determine width&height
    xmin, xmax, ymin, ymax = np.amin(xns), np.amax(xns), np.amin(yns), np.amax(yns)
    hradius = abs(xmax - xmin) / 2
    vradius = abs(ymax - ymin) / 2

    # Rotation angle
    theta = np.arccos(eigenvectors[0, 1])

    return Ellipse(c_id, mu, hradius, vradius, theta)
