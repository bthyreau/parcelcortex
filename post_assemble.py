import nibabel
import numpy as np
import sys

segfile_fn, Lout_fn, Rout_fn, out_fn, seg_threshold = sys.argv[1:]

img = nibabel.load(segfile_fn)
mask = img.get_data() > float(seg_threshold) # adjust for each segmentation (or related MR contrast), default .5

mL = nibabel.load(Lout_fn).get_data()
mR = nibabel.load(Rout_fn).get_data()

out = np.zeros(img.shape, np.uint8)

out[mask] = mL[mask]
# XXX: need to setup proper Left-Right separation
mask[mR == 0] = 0
out[mask] = mR[mask] + 100

nibabel.Nifti1Image(out, img.affine, img.header).to_filename(out_fn)
