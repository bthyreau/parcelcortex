# Check for ANTs
which antsApplyTransforms > /dev/null
if [ $? -eq "1" ]; then echo "ANTs scripts not in path"; exit; fi
# Check deps and precompile the inference model
THEANO_FLAGS="device=cpu,floatX=float32" python model_apply_parcel.py
