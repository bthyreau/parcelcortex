import nibabel
import numpy as np
import sys
import scipy.ndimage

segfile_fn, Lout_fn, Rout_fn, out_fn, atlas, seg_threshold = sys.argv[1:]

img = nibabel.load(segfile_fn)
mask = img.get_data() > float(seg_threshold) # adjust for each segmentation (or related MR contrast), default .5

mL = nibabel.load(Lout_fn).get_data().astype(np.uint8)
mR = nibabel.load(Rout_fn).get_data().astype(np.uint8)

out = np.zeros(img.shape, np.uint8)

out[mask] = mL[mask]


mask[mR == 0] = 0
# XXX: need to setup proper Left-Right separation
# but here i lost track of the separation plane, so quickly approximate it
bL = np.array(scipy.ndimage.center_of_mass(mL[::2,::2,::2])) * 2
bR = np.array(scipy.ndimage.center_of_mass(mR[::2,::2,::2])) * 2
ijk = np.array(np.where(mR & mL))
sideL = np.dot( (ijk.T - ((bR+bL)/2.)) , (bR-bL) ) < 0
ijk = [x[sideL] for x in ijk]
mask[ijk] = 0


out[mask] = mR[mask]
nibabel.Nifti1Image(out, img.affine, img.header).to_filename(out_fn)





#out[mask] = mR[mask] + 100
