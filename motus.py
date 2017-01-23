#!/usr/bin/env python

import argparse
from motus import main, config

parser = argparse.ArgumentParser()
parser.add_argument("filename", help="The filename of the inputfile. This should be in the input folder directory")
parser.add_argument("-p", "--plot", help="Also plot the fitted and catalog ellipses.", action="store_true")
args = parser.parse_args()

main.run_mser('input/{}'.format(args.filename), args.plot)