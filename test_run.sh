# Check for ANTs
which antsApplyTransforms > /dev/null
if [ $? -eq "1" ]; then echo "ANTs scripts not in path"; exit; fi
# Check deps
cd `dirname $0`
python model_apply_parcel.py
