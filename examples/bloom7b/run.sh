SCRIPT_DIR=$(cd $(dirname $0); pwd)
cd $SCRIPT_DIR

# Replace modeling_bloom.py
# linear
cd ./transformers_patch
SCRIPT_DIR=$(cd $(dirname $0); pwd)
cp $SCRIPT_DIR/modeling_bloom.py /root/anaconda3/envs/bloom7b/lib/python3.7/site-packages/transformers/models/bloom/modeling_bloom.py
cd ../
python3 run_bloom_npu.py