#!/bin/bash
set -ex

cd ${WORKSPACE}
cann_version="9.1.0"

mkdir 3rdparty && cd 3rdparty
wget -q https://ascend-cann-open.obs.cn-north-4.myhuaweicloud.com/ascend-cann/3rdparty/$(arch)/v1.13.0.tar.gz
tar -zxf v1.13.0.tar.gz
mv google-googletest-b796f7d googletest

wget -q https://ascend-cann-open.obs.cn-north-4.myhuaweicloud.com/ascend-cann/3rdparty/$(arch)/doxygen-1.9.3.src.tar.gz
tar -zxf doxygen-1.9.3.src.tar.gz
mv doxygen-1.9.3 doxygen

wget -q https://ascend-cann-open.obs.cn-north-4.myhuaweicloud.com/ascend-cann/3rdparty/$(arch)/nlohmann-json-v3.11.3.zip
unzip -q nlohmann-json-v3.11.3.zip
mv nlohmann-json-v3.11.3 nlohmannJson

wget -q https://ascend-cann-open.obs.cn-north-4.myhuaweicloud.com/ascend-cann/3rdparty/$(arch)/release-2.4.2.tar.gz
tar -zxf release-2.4.2.tar.gz
mv makeself-release-2.4.2 makeself

wget -q https://ascend-cann-open.obs.cn-north-4.myhuaweicloud.com/ascend-cann/3rdparty/$(arch)/cpp-stub.tar.gz
tar -zxf cpp-stub.tar.gz

wget -q https://ascend-cann-open.obs.cn-north-4.myhuaweicloud.com/ascend-cann/3rdparty/$(arch)/pybind11.tar.gz
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
wget -q https://ascend-cann-open.obs.cn-north-4.myhuaweicloud.com/ascend-cann/nnal/Ascend-cann-nnal_${cann_version}_linux-$(arch).run
chmod +x *.run
source /usr/local/Ascend/ascend-toolkit/set_env.sh
DEBIAN_FRONTEND=noninteractive ./Ascend-cann-nnal_${cann_version}_linux-$(arch).run  --install --quiet

source /opt/rh/devtoolset-7/enable
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export ATB_BUILD_DEPENDENCY_PATH=${ATB_HOME_PATH}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${ASCEND_HOME_PATH}/$(uname -i)-linux/devlib
export LD_LIBRARY_PATH=/usr/local/Ascend/cann/$(arch)-linux/lib64:${LD_LIBRARY_PATH}
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

cd ${WORKSPACE}
bash scripts/build.sh default --use_cxx11_abi=1 --torch_atb --torch_atb_gcc_path=/usr/bin

echo "---------------------check package--------------------------"
cd ${WORKSPACE}
cp -rf ${WORKSPACE}/output/$(arch)/Ascend-cann-atb_*.run ${WORKSPACE}/output/
export ATB_PACK_PATH=${WORKSPACE}/output
python3 tests/framework/python/ScriptTestTool/atb_script_test.py --dir tests/apitest/scripttest/root/ --mode check

ret=$?
if [ $ret -ne 0 ]; then
    echo "compile fail"
    exit 1
fi
exit 0
