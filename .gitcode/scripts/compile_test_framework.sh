#!/bin/bash

set -ex

source ${WORKSPACE}/common.sh
check_docs_changes ${WORKSPACE}/pr_filelist.txt

cann_version="${cann_version:-9.1.0}"

cd ${WORKSPACE}
mkdir 3rdparty && cd 3rdparty
wget https://ascend-cann-open.obs.cn-north-4.myhuaweicloud.com/ascend-cann/3rdparty/$(arch)/v1.13.0.tar.gz
tar -zxf v1.13.0.tar.gz
mv google-googletest-b796f7d googletest
 
wget https://ascend-cann-open.obs.cn-north-4.myhuaweicloud.com/ascend-cann/3rdparty/$(arch)/doxygen-1.9.3.src.tar.gz
tar -zxf doxygen-1.9.3.src.tar.gz 
mv doxygen-1.9.3 doxygen

wget https://ascend-cann-open.obs.cn-north-4.myhuaweicloud.com/ascend-cann/3rdparty/$(arch)/nlohmann-json-v3.11.3.zip
unzip -q nlohmann-json-v3.11.3.zip 
mv nlohmann-json-v3.11.3 nlohmannJson

wget https://ascend-cann-open.obs.cn-north-4.myhuaweicloud.com/ascend-cann/3rdparty/$(arch)/release-2.4.2.tar.gz
tar -zxf release-2.4.2.tar.gz 
mv makeself-release-2.4.2 makeself

wget https://ascend-cann-open.obs.cn-north-4.myhuaweicloud.com/ascend-cann/3rdparty/$(arch)/cpp-stub.tar.gz
tar -zxf cpp-stub.tar.gz

wget https://ascend-cann-open.obs.cn-north-4.myhuaweicloud.com/ascend-cann/3rdparty/$(arch)/pybind11.tar.gz
tar -zxf pybind11.tar.gz

cd ${WORKSPACE}/3rdparty/
git clone --branch ${TARGET_BRANCH} --depth 1 https://gitcode.com/cann/ascend-boost-comm.git Mind-KernelInfra
cd ${WORKSPACE}/3rdparty/Mind-KernelInfra
mkdir 3rdparty && cd 3rdparty
unzip -q ${WORKSPACE}/3rdparty/nlohmann-json-v3.11.3.zip && mv nlohmann-json-v3.11.3 nlohmannJson

cd ${WORKSPACE}/3rdparty/
git clone --branch ${TARGET_BRANCH} --depth 1 https://gitcode.com/cann/ascend-boost-comm.git
cd ${WORKSPACE}/3rdparty/ascend-boost-comm
mkdir 3rdparty && cd 3rdparty
unzip -q ${WORKSPACE}/3rdparty/nlohmann-json-v3.11.3.zip && mv nlohmann-json-v3.11.3 nlohmannJson

cd ${WORKSPACE}
[ -d /usr/local/Ascend/nnal ] && rm -rf /usr/local/Ascend/nnal
wget https://ascend-cann-open.obs.cn-north-4.myhuaweicloud.com/ascend-cann/nnal/Ascend-cann-nnal_${cann_version}_linux-$(arch).run
chmod +x *.run
source /usr/local/Ascend/ascend-toolkit/set_env.sh
yes | ./Ascend-cann-nnal_${cann_version}_linux-$(arch).run  --install --quiet

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export ATB_BUILD_DEPENDENCY_PATH=${ATB_HOME_PATH}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/Ascend/ascend-toolkit/latest/$(arch)-linux/devlib
export LD_LIBRARY_PATH=/usr/local/Ascend/cann/$(arch)-linux/lib64:${LD_LIBRARY_PATH}
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

echo "========================Build UT======================================"
bash scripts/build.sh testframework --no_werror --torch_atb --torch_atb_gcc_path=/usr/bin
