# Motus
Motus is a python implementation for a Terrain Relative Navigation system using a Crater Detection Algorithm based on the work published by B. Maass http://ieeexplore.ieee.org/abstract/document/6046676/. This application was developed in the context of the Spaceflight Minor at Delft University of Technology. 

## Prerequisites
Either install the packages as defined in `requirements.txt` or just create a VM from the VagrantFile which will bootstrap the environment. I have only tested it under linux, you'll probably have to do some tinkering to get it running under windows. I'd just recommend using the VagrantFile:[Installation Guide](https://www.vagrantup.com/docs/installation/)

Motus uses scikit-image, opencv-python, numpy/scipy and matplotplib.

## Usage
Edit the config.py to specify a few specific properties. Such as the folders to use for input and output as well as MSER detector properties. Details on what the various properties entail can be found in the opencv docs: http://docs.opencv.org/master/d3/d28/classcv_1_1MSER.html
 
The app can be run via the commandline:

```
   python motus.py <filename> -p
```

if the optional flag `-p` is set, the fitted results will be saved to the output folder.

## Project structure
module | description 
--- | ---
`motus.main`  | is the main algorithm execution chain and states each step in the chain.
`motus.cda`   | contains the CDA specific code, no crater catalog is necessary for this module.
`modus.trn`   | contains TRN specific code, needs an initialized crater catalog.csv
`modus.config`| contains Some configuration

## Miscellaneous

### Crater catalog
The app expects a catalog.csv (configurable in `config.py`) with the columns c_id, c_x, c_y, c_r. Specifying an crater id, x- and y-coordinates and radius.

### CDA Ellipse
The CDA module includes a Ellipse class with tests and methods intended to be used to allow coplanar conic invariant pair matching. However, my linear algebra is a bit rusty and I suspect there is an error in there somewhere. Sadly I have not been able to find it yet, thus the conic invariant approach cannot be used yet.