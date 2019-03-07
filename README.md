# Cortical Parcellation tool

This program quickly labels the human brain cortical regions, from a pre-segmented mask input.

The input can be a probabilistic (e.g. SPM c1/wc1 images) or a boolean mask (e.g. FSL and other).

The regions follow one of the Desikan-Killiany (DK), Destrieux 2009, or PALS_Brodman definition, originally defined on the surface.


![out](https://user-images.githubusercontent.com/590921/52026767-609aa300-254c-11e9-8787-eb67f74f2f89.gif)


This is based on the methods described in the manuscript *Learning a cortical parcellation of the
brain robust to the MRI segmentation with convolutional neural networks* (under review). It contains background information and more.


## Installation
The code uses numpy and Theano, and relies on ANTs.
No GPU is required.

To setup a ANTs environment, get it from http://stnava.github.io/ANTs/ (or alternatively, from a docker container such as http://www.mindboggle.info/ ). The 2.1.0 binaries are known to work.

The simplest way to install the rest from scratch is to use a Anaconda environment, then
* install scipy and Theano >=0.9.0 (`conda install theano`) (no need for CUDA/GPU-related packages if asked)
* nibabel is available on pip (`pip install nibabel`)
* Lasagne (version >=0.2 If still not available, it should be probably pulled from the github repo `pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip`)


## Usage:
After download, you can run

`./first_run.sh` in the source directory to ensure the environment is ok and to pre-compile the models.

Then, to use the program, simply run:

`./parcel_seg.sh -a aseg example/example_segmentation_t1.nii.gz`.

For more flexibility, the following options are available:

```
Options: -c N : extract label N and use that for input, useful if the input is a label map.
         -a X : use atlas X, where X is one of: aseg a2009 apals. Default to all
         -n   : skip the internal registration step, useful if the input is already aligned to MNI space
         -t X : threshold the input segmentation at X instead of 0.5 when creating the final cortical label map.
         -d   : debug - keep all temporary files
```


