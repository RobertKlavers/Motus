import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from motus import config, trn
import os
import cda
import pandas as pd
import cv2
import math

from skimage import measure


def plot_ellipses(input_image, ellipses, filename):
    """
    Plot a collection of ellipses on top of a background image and save it.

    :param input_image: a grayscale image which is the image used to fit the ellipses
    :param ellipses: a collection of Ellipse objects
    :param filename: filename of the outoput file
    """
    fig, ax = plt.subplots()

    plt.imshow(input_image, cmap='gray')

    for ellipse in ellipses:
        patch = mpatches.Ellipse(xy=ellipse.center, width=ellipse.v_r * 2, height=ellipse.h_r * 2,
                                 angle=(ellipse.angle / math.pi * 180), fill=False, edgecolor="red")
        ax.add_patch(patch)
        ax.text(ellipse.center[0], ellipse.center[1], '{}'.format(ellipse.c_id), color='white')

    ax.set_ylim([input_image.shape[0], 0])
    ax.set_xlim([0, input_image.shape[1]])
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("{}plots/{}_ellipses.png".format(config.OUTPUT_FOLDER, filename))
    plt.close()


def run_mser(scene_file, plot_result):
    """
    Runs the CV2 MSER based cda-trn algorithm

    :param scene_file: specified the scene to match
    :param plot_result: boolean, plot the result ellipse if true
    """
    image_gray_cv2 = cv2.imread(scene_file, 0)

    # Retrieve the lit and unlit extremal regions
    mser_light, mser_dark = cda.mser_cv(image_gray_cv2)
    labeled_sun = measure.label(mser_light)
    labeled_shadow = measure.label(mser_dark)

    # Identify ROIs
    rois = cda.rois(measure.regionprops(labeled_shadow), measure.regionprops(labeled_sun))

    # Fit Ellipses to the identified ROIs
    fitted_ellipses = cda.fit(rois)

    # Read catalog data
    catalog_data = pd.read_csv('{}catalog.csv'.format(config.INPUT_FOLDER)).as_matrix()

    # Set up catalog utility object
    catalog = trn.Catalog(catalog_data, config.CATALOG_ALTITUDE, config.CATALOG_FOV, config.CATALOG_SIZE)
    catalog_ellipses = catalog.initialize_catalog()

    # Plot the fitted and ellipses
    if plot_result:
        plot_ellipses(image_gray_cv2, fitted_ellipses, os.path.splitext(os.path.basename(scene_file))[0])
        plot_ellipses(image_gray_cv2, catalog_ellipses, "Catalog")

    # Perform crater matching
    crater_matches = trn.match(catalog_ellipses, fitted_ellipses)

    # Perform localization
    cam_altitude, cam_translation = catalog.localize(crater_matches)

    print("altitude: {0:.2f}, position: <x: {1:.2f} m, y: {2:.2f} m>".format(cam_altitude, cam_translation[0],
                                                                             cam_translation[1]))
