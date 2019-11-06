import nibabel
import numpy as np
import sys
import scipy.ndimage

segfile_fn_L, segfile_fn_R, Lout_fn, Rout_fn, out_fn, atlas, seg_threshold = sys.argv[1:]



img = nibabel.load(segfile_fn_L)
voxvol = np.abs(np.linalg.det(img.affine))

if img.get_data().max() < 2:
    print("WARNING: This script is intended for hemi encoded as 0-255")

labout = np.zeros(img.shape, np.uint8)

mask = img.get_data() > float(seg_threshold)
m = nibabel.load(Lout_fn).get_data().astype(np.uint8)
labout[mask] = m[mask]
outimg = labout.copy()
label_sumL = scipy.ndimage.sum(np.ones(labout.shape), labels = labout, index = range(100)) * voxvol
label_weightedsumL = scipy.ndimage.sum(img.get_data(), labels = labout, index = range(100)) * voxvol / 255.
thicknessL = nibabel.load(segfile_fn_L.replace("_ribbon", "_thickness")).get_data()
labout[thicknessL == 0] = 0 # do not include voxels where there is no thickness data
label_thickmeanL = np.nan_to_num(scipy.ndimage.mean(thicknessL, labels = labout, index = range(100)))
label_thickvarL = np.nan_to_num(scipy.ndimage.variance(thicknessL, labels = labout, index = range(100)))



img = nibabel.load(segfile_fn_R)

labout = np.zeros(img.shape, np.uint8)

mask = img.get_data() > float(seg_threshold)
m = nibabel.load(Rout_fn).get_data().astype(np.uint8)
labout[mask] = m[mask]
outimg[mask] = m[mask]
label_sumR = scipy.ndimage.sum(np.ones(labout.shape), labels = labout, index = range(100)) * voxvol
label_weightedsumR = scipy.ndimage.sum(img.get_data(), labels = labout, index = range(100)) * voxvol / 255.
thicknessR = nibabel.load(segfile_fn_R.replace("_ribbon", "_thickness")).get_data()
labout[thicknessR == 0] = 0 # do not include voxels where there is no thickness data
label_thickmeanR = np.nan_to_num(scipy.ndimage.mean(thicknessR, labels = labout, index = range(100)))
label_thickvarR = np.nan_to_num(scipy.ndimage.variance(thicknessR, labels = labout, index = range(100)))

nibabel.Nifti1Image(outimg, img.affine, img.header).to_filename(out_fn)
del outimg


if atlas == "aseg":
    name_d = {1: 'bankssts', 2: 'caudalanteriorcingulate', 3: 'caudalmiddlefrontal', 4: 'corpuscallosum', 5: 'cuneus', 6: 'entorhinal', 7: 'fusiform', 8: 'inferiorparietal', 9: 'inferiortemporal', 10: 'isthmuscingulate', 11: 'lateraloccipital', 12: 'lateralorbitofrontal', 13: 'lingual', 14: 'medialorbitofrontal', 15: 'middletemporal', 16: 'parahippocampal', 17: 'paracentral', 18: 'parsopercularis', 19: 'parsorbitalis', 20: 'parstriangularis', 21: 'pericalcarine', 22: 'postcentral', 23: 'posteriorcingulate', 24: 'precentral', 25: 'precuneus', 26: 'rostralanteriorcingulate', 27: 'rostralmiddlefrontal', 28: 'superiorfrontal', 29: 'superiorparietal', 30: 'superiortemporal', 31: 'supramarginal', 32: 'frontalpole', 33: 'temporalpole', 34: 'transversetemporal', 35: 'insula'}

elif atlas == "a2009":
    name_d = {1: 'G_and_S_frontomargin', 2: 'G_and_S_occipital_inf', 3: 'G_and_S_paracentral', 4: 'G_and_S_subcentral', 5: 'G_and_S_transv_frontopol', 6: 'G_and_S_cingul-Ant', 7: 'G_and_S_cingul-Mid-Ant', 8: 'G_and_S_cingul-Mid-Post', 9: 'G_cingul-Post-dorsal', 10: 'G_cingul-Post-ventral', 11: 'G_cuneus', 12: 'G_front_inf-Opercular', 13: 'G_front_inf-Orbital', 14: 'G_front_inf-Triangul', 15: 'G_front_middle', 16: 'G_front_sup', 17: 'G_Ins_lg_and_S_cent_ins', 18: 'G_insular_short', 19: 'G_occipital_middle', 20: 'G_occipital_sup', 21: 'G_oc-temp_lat-fusifor', 22: 'G_oc-temp_med-Lingual', 23: 'G_oc-temp_med-Parahip', 24: 'G_orbital', 25: 'G_pariet_inf-Angular', 26: 'G_pariet_inf-Supramar', 27: 'G_parietal_sup', 28: 'G_postcentral', 29: 'G_precentral', 30: 'G_precuneus', 31: 'G_rectus', 32: 'G_subcallosal', 33: 'G_temp_sup-G_T_transv', 34: 'G_temp_sup-Lateral', 35: 'G_temp_sup-Plan_polar', 36: 'G_temp_sup-Plan_tempo', 37: 'G_temporal_inf', 38: 'G_temporal_middle', 39: 'Lat_Fis-ant-Horizont', 40: 'Lat_Fis-ant-Vertical', 41: 'Lat_Fis-post', 42: 'Medial_wall', 43: 'Pole_occipital', 44: 'Pole_temporal', 45: 'S_calcarine', 46: 'S_central', 47: 'S_cingul-Marginalis', 48: 'S_circular_insula_ant', 49: 'S_circular_insula_inf', 50: 'S_circular_insula_sup', 51: 'S_collat_transv_ant', 52: 'S_collat_transv_post', 53: 'S_front_inf', 54: 'S_front_middle', 55: 'S_front_sup', 56: 'S_interm_prim-Jensen', 57: 'S_intrapariet_and_P_trans', 58: 'S_oc_middle_and_Lunatus', 59: 'S_oc_sup_and_transversal', 60: 'S_occipital_ant', 61: 'S_oc-temp_lat', 62: 'S_oc-temp_med_and_Lingual', 63: 'S_orbital_lateral', 64: 'S_orbital_med-olfact', 65: 'S_orbital-H_Shaped', 66: 'S_parieto_occipital', 67: 'S_pericallosal', 68: 'S_postcentral', 69: 'S_precentral-inf-part', 70: 'S_precentral-sup-part', 71: 'S_suborbital', 72: 'S_subparietal', 73: 'S_temporal_inf', 74: 'S_temporal_sup', 75: 'S_temporal_transverse'}

elif atlas == "pals":
    name_d = {1: 'Brodmann.1', 2: 'Brodmann.2', 3: 'Brodmann.3', 4: 'Brodmann.4', 5: 'Brodmann.5', 6: 'Brodmann.6', 7: 'Brodmann.7', 8: 'Brodmann.8', 9: 'Brodmann.9', 10: 'Brodmann.10', 11: 'Brodmann.11', 17: 'Brodmann.17', 18: 'Brodmann.18', 19: 'Brodmann.19', 20: 'Brodmann.20', 21: 'Brodmann.21', 22: 'Brodmann.22', 23: 'Brodmann.23', 24: 'Brodmann.24', 25: 'Brodmann.25', 26: 'Brodmann.26', 27: 'Brodmann.27', 28: 'Brodmann.28', 29: 'Brodmann.29', 30: 'Brodmann.30', 31: 'Brodmann.31', 32: 'Brodmann.32', 33: 'Brodmann.33', 35: 'Brodmann.35', 36: 'Brodmann.36', 37: 'Brodmann.37', 38: 'Brodmann.38', 39: 'Brodmann.39', 40: 'Brodmann.40', 41: 'Brodmann.41', 42: 'Brodmann.42', 43: 'Brodmann.43', 44: 'Brodmann.44', 45: 'Brodmann.45', 46: 'Brodmann.46', 47: 'Brodmann.47'}


# vol is the region volume; weighted_vol is the region volume weighted by the tissue volume
# (useful if the segmentation is made of continuous [0-1] values)

with open(out_fn.replace(".nii.gz", "_metrics.txt"), "w") as h:
    h.write("label,side,region_name,vol,weighted_vol,thickness_mean,thickness_var\n")
    for idx in sorted(name_d.keys()):
        h.write("%d,L,%s,%.2f,%.2f,%.3f,%.3f\n" % (idx, name_d[idx], label_sumL[idx], label_weightedsumL[idx], label_thickmeanL[idx], label_thickvarL[idx]))
    for idx in sorted(name_d.keys()):
        h.write("%d,R,%s,%.2f,%.2f,%.3f,%.3f\n" % (idx, name_d[idx], label_sumR[idx], label_weightedsumR[idx], label_thickmeanR[idx], label_thickvarR[idx]))
