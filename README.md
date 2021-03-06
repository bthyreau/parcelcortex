# Cortical Parcellation tool

This program quickly labels the human brain cortical regions, from a pre-segmented mask input.

The input can be a probabilistic (e.g. SPM c1/wc1 images) or a boolean mask (e.g. FSL and other).

The regions follow one of the Desikan-Killiany (DK), Destrieux 2009, or PALS_Brodman definition, originally defined on the surface.


![out](https://user-images.githubusercontent.com/590921/52026767-609aa300-254c-11e9-8787-eb67f74f2f89.gif)


This is based on the methods described in the manuscript *Learning a cortical parcellation of the
brain robust to the MRI segmentation with convolutional neural networks* ( https://doi.org/10.1016/j.media.2020.101639 ). It contains background information and more.

This version is a direct PyTorch port of the model originally developed and trained on the (now unmaintained) Theano framework. The theano-based inference code is still available in the "theano" git branch of this repository. Both version use the same parameters and should give identical results.

## Installation
This code uses pytorch, and relies on ANTs.
No GPU is required.

To setup a ANTs environment, get it from http://stnava.github.io/ANTs/ (or alternatively, from a docker container). The 2.1.0 binaries are known to work.

The simplest way to install the rest from scratch is to use a Anaconda or virtualenv environment using Python >3.5, then
* install numpy, scipy, if not installed (`conda install scipy` or `pip install scipy`)
* install nibabel, available on pip (`pip install nibabel`)
* install pytorch ( > 1.0.0, tested until 1.4.0 ) following the instruction of https://pytorch.org (CPU only is ok, no need for CUDA/GPU-related dependencies)



## Usage:
After download, you can run

`./test_run.sh` in the source directory to ensure the environment is ok.

To use the program, simply run:

`./parcel_seg.sh -a aseg example/example_segmentation_t1.nii.gz`.

For more flexibility, the following options are available:

```
Options: -c N : extract label N and use that for input, useful if the input is a label map.
         -a X : use atlas X, where X is one of: aseg a2009 apals. Default to all
         -n   : skip the internal registration step, useful if the input is already aligned to MNI space
         -t X : threshold the input segmentation at X instead of 0.5 when creating the final cortical label map.
         -d   : debug - keep all temporary files
```


