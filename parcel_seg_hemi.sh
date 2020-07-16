#!/bin/bash
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1
export OPENBLAS_NUM_THREADS=1

scriptpath=$(dirname $0); [ "${0:0:1}" != '/' ] && scriptpath="$PWD/$scriptpath"

while (( "$#" )); do
        case $1 in
        -a) shift; ATLAS=$1;;
        -t) shift; seg_threshold=$1;;
        -LR) SHIFT_RIGHT_LABELS="1";;
        -h) echo "Usage  : $0 [ -a X ] [ -t X ] input_segmentation

 where input_segmentation is a prefix such that:
  input_segmentation_ribbon{R,L}.nii.gz exist containing a single hemi each,
  and input_segmentation_mni0Affine.mat (or .txt) an ANTs-compatible matrix to MNI space.

Options:
 -a X : use atlas X, where X is one of: aseg a2009 apals. Default to all
 -t X : threshold the input segmentation at X instead of 92 when creating the
        final cortical label map and counting region volumes. (0 to 255)
 -LR  : Separate Left and Right in the final labelmap (Right codes shifted +100)"
        exit;;
        -*) echo "unexpected option $1"; exit;;
         *) if [ "$filename" != "" ] ; then echo "unexpected argument $1"; exit; fi; filename=$1;;
        esac
        shift
done

which antsApplyTransforms > /dev/null
if [ $? -eq "1" ]; then echo "ANTs scripts not in path"; exit; fi

if [ "$filename" == "" ]; then echo "Use -h option for usage"; exit; fi

a=$(basename $filename)
pth=$(dirname $filename)
cd $pth


if [ ! -f ${a}_ribbonR.nii.gz ]; then
	echo "Ribbon file not found (should be named ${a}_ribbonL.nii.gz, resp. R)"
    exit 1;
fi

# In this script, matrix filenames are hardcoded .txt files, but it should also work
# with mat files. Unfortunately, ANTs use the filename suffix to detect the format,
# so a trivial rename of ${a}0GenericAffine.mat to ${a}_mni0Affine.txt will not work
suffix_mat="txt"
if [ ! -f ${a}_mni0Affine.txt ]; then
    if [ -f ${a}_mni0Affine.mat ]; then
	    echo "ANTs-compatible MNI Affine transformation file found as .mat file (${a}_mni0Affine.mat). Using it."
        suffix_mat="mat"
    else
	    echo "ANTs-compatible MNI Affine transformation file not found (should be named ${a}_mni0Affine.txt)"
        exit 1;
    fi
fi

atlas_list=${ATLAS:-"a2009 aseg pals"}




antsApplyTransforms -i ${a}_ribbonR.nii.gz -t ${a}_mni0Affine.$suffix_mat -r ${scriptpath}/templates/dil_ig_ribbon_ig_b96_box128_lout_T1_thr.nii.gz -o b96_box128_lout_${a}.nii.gz --float -n Gaussian

antsApplyTransforms -i ${a}_ribbonL.nii.gz -t ${a}_mni0Affine.$suffix_mat -r ${scriptpath}/templates/dil_ig_ribbon_ig_b96_box128_rout_T1_thr.nii.gz -o b96_box128_rout_${a}.nii.gz --float -n Gaussian

python $scriptpath/model_apply_parcel.py b96_box128_lout_${a}.nii.gz $atlas_list

for atlas in $atlas_list; do
    antsApplyTransforms -i b96_box128_rout_${a}_outlab_${atlas}_filled.nii.gz -r ${a}_ribbonL.nii.gz -o Lout_${a}_${atlas}_filled.nii -t [ ${a}_mni0Affine.$suffix_mat,1] -n NearestNeighbor --float
    antsApplyTransforms -i b96_box128_lout_${a}_outlab_${atlas}_filled.nii.gz -r ${a}_ribbonR.nii.gz -o Rout_${a}_${atlas}_filled.nii -t [ ${a}_mni0Affine.$suffix_mat,1] -n NearestNeighbor --float
    python $scriptpath/post_assemble_hemi.py ${a}_ribbonL.nii.gz ${a}_ribbonR.nii.gz Lout_${a}_${atlas}_filled.nii Rout_${a}_${atlas}_filled.nii ${a}_labelled_${atlas}.nii.gz $atlas ${seg_threshold:-"92"} ${SHIFT_RIGHT_LABELS:-"0"}
    #python $scriptpath/post_assemble_hemi_with_thickness.py ${a}_ribbonL.nii.gz ${a}_ribbonR.nii.gz Lout_${a}_${atlas}_filled.nii Rout_${a}_${atlas}_filled.nii ${a}_labelled_${atlas}.nii.gz $atlas ${seg_threshold:-"92"} ${SHIFT_RIGHT_LABELS:-"0"}
done


rm b96_box128_[rl]out_${a}*.nii.gz
# Those are the non-masked labels:
#rm Lout_${a}_*_filled.nii Rout_${a}_*_filled.nii
gzip -f Lout_${a}_*_filled.nii ; gzip -f Rout_${a}_*_filled.nii
