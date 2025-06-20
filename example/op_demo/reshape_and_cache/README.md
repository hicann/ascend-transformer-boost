# 加速库ReshapeAndCacheOperation C++ Demo
## 介绍
该目录下为加速库ReshapeAndCacheOperation C++调用示例。

## 使用说明
- 首先source 对应的CANN和nnal包
    1. source [cann安装路径]/set_env.sh
        默认：source /usr/local/Ascend/ascend-toolkit/set_env.sh
    2. source [nnal安装路径]/set_env.sh
        默认：source /usr/local/Ascend/ascend-toolkit/set_env.sh
        1. 如果使用加速库源码编译，source [加速库源码路径]/output/atb/set_env.sh
        e.g. source ./ascend-transformer-boost/atb/set_env.sh

- 运行demo
    - bash build.sh
    **注意**：
    - 使用cxx_abi=0（默认）时，设置`D_GLIBCXX_USE_CXX11_ABI`为0，i.e.
        ```sh
        g++ -D_GLIBCXX_USE_CXX11_ABI=0 -I ...
        ```
    - 使用cxx_abi=1时，更改`D_GLIBCXX_USE_CXX11_ABI`为1，i.e.
        ```sh
        g++ -D_GLIBCXX_USE_CXX11_ABI=1 -I ...
        ```
    - 提供的build脚本仅用于编译和运行reshape_and_cache_demo.cpp，如需编译其他demo，需要替换“reshape_and_cache_demo”为对应的cpp文件名

## 额外说明
示例中生成的数据不代表实际场景，如需数据生成参考请查看python用例目录：
tests/apitest/opstest/python/operations/reshape_and_cache/

## 产品支持情况
本op在Atlas A2/A3系列和Atlas 推理系列产品上实现有所区别

### 场景说明
提供demo分别对应不同产品的基础场景，编译运行时需要对应更改build脚本：
1. Atlas A2/A3：
    reshape_and_cache_demo.cpp
    - 默认编译脚本可编译运行
    - 该demo仅支持在Atlas A2/A3系列上运行

2. Atlas推理系列产品：
    reshape_and_cache_inference_demo.cpp
    - 相较于A2/A3的demo，本示例主要有以下修改点：
        - kvCache shape改为：[block_num, head_size \times head_num / 16, block_size, 16]。
        - kvCache数据格式改为：ACL_FORMAT_FRACTAL_NZ。

    - 更改编译脚本为：
    `g++ -D_GLIBCXX_USE_CXX11_ABI=$cxx_abi -I "${ATB_HOME_PATH}/include" -I "${ASCEND_HOME_PATH}/include" -L "${ATB_HOME_PATH}/lib" -L "${ASCEND_HOME_PATH}/lib64" reshape_and_cache_inference_demo.cpp demo_util.h -l atb -l ascendcl -o reshape_and_cache_inference_demo`
    - 运行时调用：
    `./reshape_and_cache_inference_demo`
    - 该demo仅支持在Atlas 推理系列产品上运行
