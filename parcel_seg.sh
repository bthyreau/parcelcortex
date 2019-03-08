#!/bin/bash
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1
export OPENBLAS_NUM_THREADS=1

scriptpath=$(dirname $0); [ "${0:0:1}" != '/' ] && scriptpath="$PWD/$scriptpath"

while (( "$#" )); do
        case $1 in
        -n) NOREG=1;;
        -c) shift; CLASS=$1;;
        -a) shift; ATLAS=$1;;
        -t) shift; seg_threshold=$1;;
        -d) DEBUG=1;;
        -h) echo "Usage  : $0 [ -n ] [ -a X ] [ -c N ] [ -t X ] input_segmentation.nii

Options:
 -c N : extract label N and use it for input, useful if the input is a labelmap.
 -a X : use atlas X, where X is one of: aseg a2009 apals. Default to all
 -n   : skip the internal registration step, useful if the input is already
        aligned to MNI space.
 -t X : threshold the input segmentation at X instead of 0.5 when creating the
        final cortical label map and counting region volumes.
 -d   : debug - keep all temporary files."
        exit;;
        -*) echo "unexpected option $1"; exit;;
         *) if [ "$filename" != "" ] ; then echo "unexpected argument $1"; exit; fi; filename=$1;;
        esac
        shift
done

which antsApplyTransforms > /dev/null
if [ $? -eq "1" ]; then echo "ANTs scripts not in path"; exit; fi

if [ "`echo ?`" != '?' ]; then
    echo "*** The following file(s) were found in the current directory ($PWD): `echo ?`"
    echo "The presence of files named with a single character may cause failures in some ANTs version."
    echo "Aborting for safety."
    exit 1;
fi

if [ ! -f "$filename" ]; then echo "input file not found $filename"; exit; fi

# try to drop a few differents suffix names
a=$(basename $filename)
a1=$a
for suffix in gz nii img hdr mgz mgh; do a=$(basename $a .$suffix); done
pth=$(dirname $filename)
cd $pth

atlas_list=${ATLAS:-"a2009 aseg pals"}

#echo "filename=$filename NOREG=$NOREG CLASS=$CLASS atlas_list=$atlas_list st=$seg_threshold"

if [ $CLASS ]; then
        echo "Extracting component $CLASS from $a1"
        ThresholdImage 3 $a1 class_${a}.nii $CLASS $CLASS 1 0
        if [ $NOREG ]; then echo "Smoothing"; ImageMath 3 class_${a}.nii G class_${a}.nii 1; fi
        a1=class_${a}.nii
fi

if [ $NOREG ]; then
    cp $scriptpath/affine.mat aff_${a}0GenericAffine.mat
else
    ResampleImage 3 ${a1} res64_${a}.nii 64x64x64 1 0
    echo "Realignment"
    echo antsRegistrationSyNQuick.sh -m res64_${a}.nii -f ${scriptpath}/templates/TPM_c1.nii.gz -t a -o aff_${a}

    antsRegistrationSyNQuick.sh -m res64_${a}.nii -f ${scriptpath}/templates/TPM_c1.nii.gz -t a -o aff_${a} -n 2 > /dev/null
fi

antsApplyTransforms -i ${a1} -t aff_${a}0GenericAffine.mat -r ${scriptpath}/templates/dil_ig_ribbon_ig_b96_box128_lout_T1_thr.nii.gz -o b96_box128_lout_${a}.nii.gz --float -n Gaussian

antsApplyTransforms -i ${a1} -t aff_${a}0GenericAffine.mat -r ${scriptpath}/templates/dil_ig_ribbon_ig_b96_box128_rout_T1_thr.nii.gz -o b96_box128_rout_${a}.nii.gz --float -n Gaussian

THEANO_FLAGS="device=cpu,floatX=float32,compile.wait=1" python $scriptpath/model_apply_parcel.py b96_box128_lout_${a}.nii.gz $atlas_list

for atlas in $atlas_list; do
    antsApplyTransforms -i b96_box128_rout_${a}_outlab_${atlas}_filled.nii.gz -r ${a1} -o Lout_${a}_${atlas}_filled.nii -t [aff_${a}0GenericAffine.mat,1] -n NearestNeighbor --float
    antsApplyTransforms -i b96_box128_lout_${a}_outlab_${atlas}_filled.nii.gz -r ${a1} -o Rout_${a}_${atlas}_filled.nii -t [aff_${a}0GenericAffine.mat,1] -n NearestNeighbor --float
    python $scriptpath/post_assemble.py ${a1} Lout_${a}_${atlas}_filled.nii Rout_${a}_${atlas}_filled.nii ${a}_labelled_${atlas}.nii.gz $atlas ${seg_threshold:-".5"}
done

if [ $DEBUG ]; then
    gzip -f -3 res64_${a}.nii
else
    rm -f class_${a}.nii
    rm b96_box128_[rl]out_${a}*.nii.gz
    rm aff_${a}0GenericAffine.mat
    # Those are the non-masked labels:
    rm Lout_${a}_*_filled.nii Rout_${a}_*_filled.nii
    rm res64_${a}.nii
    rm aff_${a}*Warped.nii.gz
fi
