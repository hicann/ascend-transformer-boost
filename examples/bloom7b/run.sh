SCRIPT_DIR=$(cd $(dirname $0); pwd)
if [ -z $1 ];then
    SCRIPT_PATH=$SCRIPT_DIR/modeling_bloom.py
else
    SCRIPT_PATH=$(cd $(dirname $1); pwd)/$(basename $1)
fi
TRANSFORMER_PACKAGE_PATH=$(python3 -c 'import transformers; import os; print(os.path.dirname(transformers.__file__))')/models/bloom

if [ ! -f "$SCRIPT_DIR/flax_model-00001-of-00002.msgpack" ];then
    ln -s $ACLTRANSFORMER_TESTDATA/weights/bloom7b/flax_model-00001-of-00002.msgpack $SCRIPT_DIR/flax_model-00001-of-00002.msgpack
    ln -s $ACLTRANSFORMER_TESTDATA/weights/bloom7b/flax_model-00002-of-00002.msgpack $SCRIPT_DIR/flax_model-00002-of-00002.msgpack
    ln -s $ACLTRANSFORMER_TESTDATA/weights/bloom7b/pytorch_model-00001-of-00002.bin $SCRIPT_DIR/pytorch_model-00001-of-00002.bin
    ln -s $ACLTRANSFORMER_TESTDATA/weights/bloom7b/pytorch_model-00002-of-00002.bin $SCRIPT_DIR/pytorch_model-00002-of-00002.bin
    ln -s $ACLTRANSFORMER_TESTDATA/weights/bloom7b/tokenizer.json $SCRIPT_DIR/tokenizer.json
fi

if [ -f "$TRANSFORMER_PACKAGE_PATH/modeling_bloom_origin.py" ];then
    rm -rf $TRANSFORMER_PACKAGE_PATH/modeling_bloom_origin.py
fi

if [ -f "$TRANSFORMER_PACKAGE_PATH/modeling_bloom.py" ];then
    mv $TRANSFORMER_PACKAGE_PATH/modeling_bloom.py $TRANSFORMER_PACKAGE_PATH/modeling_bloom.py.bak
fi

if [ ! -f $SCRIPT_PATH ];then
    echo "cannot find the file to be tested"
    exit 1
fi

cp $SCRIPT_DIR/modeling_bloom.py $TRANSFORMER_PACKAGE_PATH/modeling_bloom_origin.py
cp $SCRIPT_PATH $TRANSFORMER_PACKAGE_PATH/modeling_bloom.py

python3 run_bloom_npu.py

rm -f $TRANSFORMER_PACKAGE_PATH/modeling_bloom_origin.py
if [ -f "$TRANSFORMER_PACKAGE_PATH/modeling_bloom.py.bak" ];then
    mv $TRANSFORMER_PACKAGE_PATH/modeling_bloom.py.bak $TRANSFORMER_PACKAGE_PATH/modeling_bloom.py
fi