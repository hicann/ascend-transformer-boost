# for asd log
#export ASDOPS_LOG_TO_STDOUT=1
#export ASDOPS_LOG_LEVEL=INFO
#export ACLTRANSFORMER_SAVE_TENSOR=1

export ACLTRANSFORMER_CONVERT_NCHW_TO_ND=1

rm -rf test_log.log
python3 main_manually.py

