from __future__ import print_function

from lasagne.layers import get_output, InputLayer, DenseLayer, ReshapeLayer, NonlinearityLayer
from lasagne.nonlinearities import rectify, leaky_rectify
import sys, os, time

import nibabel
import numpy as np
import theano
import theano.tensor as T

import lasagne

# Note that Conv3DLayer and .Conv3DLayer have opposite filter-fliping defaults
from lasagne.layers import Conv3DLayer, MaxPool3DLayer
from lasagne.layers import Upscale3DLayer

from lasagne.layers import *

import pickle
import theano.misc.pkl_utils

cachefile = os.path.dirname(os.path.realpath(__file__)) + "/model_cached.pkl"

if not os.path.exists(cachefile):
    l_input = InputLayer(shape = (None, 1, 48, 96, 96), name="input")
    l_input_atlas = InputLayer(shape = (None, 3), name="input_atlas")
    l_input_side = InputLayer(shape = (None, 2), name="input_side")

    l_atlas_hint = Upscale3DLayer(ReshapeLayer(l_input_atlas, ([0], [1], 1,1,1)), scale_factor = (3,6,6), name="upscale")
    l_side_hint = Upscale3DLayer(ReshapeLayer(l_input_side, ([0], [1], 1,1,1)), scale_factor = (3,6,6), name="upscale")

    l = l_input
    l = Conv3DLayer(l, num_filters = 16, filter_size = (1,1,3), pad = 'same', name="conv")
    l = Conv3DLayer(l, num_filters = 16, filter_size = (1,3,1), pad = 'same', name="conv")
    l = Conv3DLayer(l, num_filters = 16, filter_size = (3,1,1), pad = 'same', name="conv")
    l = batch_norm(l)
    li0 = l 

    l = MaxPool3DLayer(l, pool_size = 2, name ='maxpool')
    l = Conv3DLayer(l, num_filters = 32, filter_size = (3,3,3), pad = 'same', name="conv")
    l = batch_norm(l)
    li1 = l
    l = Conv3DLayer(l, num_filters = 96, filter_size = 1, pad = 'same', name="conv")

    l = MaxPool3DLayer(l, pool_size = 2, name ='maxpool')
    l = Conv3DLayer(l, num_filters = 96, filter_size = (3,3,3), pad = 'same', name="conv")
    l = batch_norm(l)
    li2 = l 
    l = Conv3DLayer(l, num_filters = 128, filter_size = 1, pad = 'same', name ='conv')

    l = MaxPool3DLayer(l, pool_size = 2, name ='maxpool')
    l = Conv3DLayer(l, num_filters = 96, filter_size = (3,3,3), pad = 'same', name="conv")
    l = Conv3DLayer(l, num_filters = 128, filter_size = 1, pad = 'same', name ='conv')
    l = batch_norm(l)

    l = MaxPool3DLayer(l, pool_size = 2, name ='maxpool')
    l = ConcatLayer([l, l_atlas_hint, l_side_hint])
    l = Conv3DLayer(l, num_filters = 128, filter_size = (3,3,3), pad = 'same', name="conv")
    l = Conv3DLayer(l, num_filters = 128, filter_size = 1, pad = 'same', name ='conv')
    l = batch_norm(l)
    l_middle = l
    l = Upscale3DLayer(l, scale_factor = 2, name="upscale")
    l = Conv3DLayer(l, num_filters = 128, filter_size = (3,3,3), pad = 'same', name="conv")
    l = Conv3DLayer(l, num_filters = 128, filter_size = 1, pad = 'same', name ='conv')
    l = batch_norm(l)

    l = Upscale3DLayer(l, scale_factor = 2, name="upscale")
    l = Conv3DLayer(l, num_filters = 96, filter_size = (3,3,3), pad = 'same', name="conv")
    l = ConcatLayer([l, li2])
    l = Conv3DLayer(l, num_filters = 96, filter_size = 1, pad = 'same', name ='conv')
    l = batch_norm(l)

    l = Upscale3DLayer(l, scale_factor = 2, name="upscale")
    l = Conv3DLayer(l, num_filters = 96, filter_size = (3,3,3), pad = 'same', name="conv")
    l = ConcatLayer([l, li1])
    l = Conv3DLayer(l, num_filters = 96, filter_size = 1, pad = 'same', name ='conv')
    l = batch_norm(l)

    l = Upscale3DLayer(l, scale_factor = 2, name="upscale")
    l = Conv3DLayer(l, num_filters = 96, filter_size = (1,1,3), pad = 'same', name="conv")
    l = Conv3DLayer(l, num_filters = 96, filter_size = (1,3,1), pad = 'same', name="conv")
    l = Conv3DLayer(l, num_filters = 96, filter_size = (3,1,1), pad = 'same', name="conv")
    l = ConcatLayer([l, li0])

    l = Conv3DLayer(l, num_filters = 75, filter_size = 1, pad = "same", name="conv1x", nonlinearity = lasagne.nonlinearities.sigmoid )
    lastl = l
    network = l

    def reload_fn(fn):
        with np.load(fn) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(lastl, param_values)

    reload_fn(os.path.dirname(os.path.realpath(__file__)) + "/params_parcel.npz")

    print("Compiling")

    getout = theano.function([l_input.input_var, l_input_atlas.input_var, l_input_side.input_var], lasagne.layers.get_output(lastl, deterministic=True))

    print("Pickling")
    pickle.dump(getout, open(cachefile, "wb"))

else:
    print("Loading from cache")
    getout = pickle.load(open(cachefile,"rb"))

atlas_codes = {"a2009": ([1,0,0], 75), "aseg": ([0, 1, 0], 35), "pals": ([0, 0, 1], 48)}
hemi_template_file = os.path.dirname(os.path.realpath(__file__)) + "/templates/dil_ig_ribbon_ig_b96_box128_lout_T1_thr.nii.gz"
roi = nibabel.load(hemi_template_file).get_data() > .5

if len(sys.argv) > 1:
    fnamel = sys.argv[1]
    assert("b96_box128_lout" in fnamel)

    if len(sys.argv) >= 3:
        atlas_list = sys.argv[2:]
        assert all([atlas in ["a2009", "aseg", "pals"] for atlas in atlas_list])
    else:
        atlas_list = ["a2009", "aseg", "pals"]

    T = time.time()
    for atlas in atlas_list:
        print("Using atlas %s" % atlas)
        for fname in [fnamel, fnamel.replace("_lout_", "_rout_")]:
            img = nibabel.load(fname)

            d = img.get_data().astype(np.float32)
            d_orr = d
            side_hint = [1, 0]

            if "_rout_" in fname:
                d_orr = d_orr[::-1]
                side_hint = side_hint[::-1]

            print("Starting inference on %s using atlas %s" % (fname, atlas))
            atlas_code, nb_roi = atlas_codes[atlas]
            d_orr[~roi] = 0
            out1 = getout(d_orr[None,None], [atlas_code], [side_hint])
            print("Inference " + str(time.time() - T))

            a=np.argmax(out1[:,:nb_roi], axis=1) + 1
            a[out1[:,:nb_roi].max(axis=1) < .001] = 0 # mostly for debug
            outt = a[0].astype(np.uint8)
            outt[~roi] = 0 # no need to fill too far

            if "_rout_" in fname:
                outt = outt[::-1]

            nibabel.Nifti1Image(outt, img.affine).to_filename( fname.replace(".nii.gz", "_outlab_%s_filled.nii.gz" % atlas))
