# Cortical Parcellation tool

This program quickly labels the human brain cortical regions, from a pre-segmented mask input.

See the main branch for more background information and instruction.

This branch is intended to skip any preprocessing and directly work on clean pre-segmented hemisphere masks, such as those produced by FreeSurfer.
It needs a registration matrix to MNI too.

```
Usage  : ./parcel_seg_hemi.sh [ -a X ] [ -t X ] input_segmentation

 where input_segmentation is a prefix such that:
  input_segmentation_ribbon{R,L}.nii.gz exist containing a single hemi each,
  and input_segmentation_mni0Affine.mat (or .txt) an ANTs-compatible matrix to MNI space.

Options:
 -a X : use atlas X, where X is one of: aseg a2009 apals. Default to all
 -t X : threshold the input segmentation at X instead of 92 when creating the
        final cortical label map and counting region volumes. (0 to 255)
 -LR  : Separate Left and Right in the final labelmap (Right codes shifted +100)
```
