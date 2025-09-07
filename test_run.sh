r0=$(realpath $0)
PATH=$(dirname $r0)/.venv/bin/:$(dirname $r0)/venv/bin/:$PATH # for uv-based installation
# Check for ANTs
which antsApplyTransforms > /dev/null
if [ $? -eq "1" ]; then echo "ANTs scripts not in path"; exit; fi
# Check deps
cd `dirname $0`
python3 model_apply_parcel.py
