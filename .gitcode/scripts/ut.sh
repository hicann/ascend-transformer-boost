#!/bin/bash

set -ex

source ${WORKSPACE}/.gitcode/scripts/common.sh
check_docs_changes ${WORKSPACE}/pr_filelist.txt

cd ${WORKSPACE}

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${ASCEND_HOME_PATH}/$(uname -i)-linux/devlib
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
export PATH=$PATH:/opt/buildtools/python-3.10.2/bin
export LD_LIBRARY_PATH=/opt/buildtools/python-3.10.2/lib/python3.10/site-packages/torch_npu/lib:/opt/buildtools/python-3.10.2/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=/usr/local/Ascend/cann/$(arch)-linux/lib64:${LD_LIBRARY_PATH}
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export ATB_BUILD_DEPENDENCY_PATH=${ATB_HOME_PATH}
unset ASCEND_RT_VISIBLE_DEVICES

pip3 uninstall torch-atb -y

python ${WORKSPACE}/atb_scheduler.py --branch ${TARGET_BRANCH}

ret=$?
if [ $ret -ne 0 ]; then
    echo "run ut fail"
    exit 1
fi
exit 0