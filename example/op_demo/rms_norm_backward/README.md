# 加速库RmsNormOperation C++ Demo
## 介绍
该目录下为加速库RmsNormOperation C++调用示例。

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
    - 提供的build脚本仅用于编译和运行rms_norm_backward_demo.cpp，如需编译其他demo，需要替换“rms_norm_backward_demo”为对应的cpp文件名

## 额外说明
示例中生成的数据不代表实际场景，如需数据生成参考请查看python用例目录：
tests/apitest/opstest/python/operations/rms_norm_backward/
