# ATB加速库外部开发者自定义算子目录

## 介绍

单独为外部开发者设置开发目录，外部开发者可以按照本目录下的customize_block_copy Operation实现自定义算子。本目录支持单独编译和测试，也支持与ATB加速库一同编译。

## 使用说明
以customize_block_copy Operation为例
### 单独编译
- 首先source 对应的CANN和nnal包
    1. source [cann安装路径]/set_env.sh
        
            e.g. source /usr/local/Ascend/ascend-toolkit/set_env.sh
    2. source [nnal安装路径]/set_env.sh
        
            e.g. source /usr/local/Ascend/nnal/atb/set_env.sh
        1. 如果使用加速库源码编译，source [加速库源码路径]/output/atb/set_env.sh
            
                e.g. source ./ascend-transformer-boost/output/atb/set_env.sh
- 编译自定义算子目录
    - bash build.sh
        
        该脚本目前支持: default|clean|unittest| --use_cxx11_abi=0|--use_cxx11_abi=1|--debug|--msdebug
- 执行用例
    - bash build.sh unittest

        执行customize_block_copy Operation的测试用例
### 与ATB加速库一同编译
- 首先source 对应的CANN包和设置编译所需的nnal包
    1. source [cann安装路径]/set_env.sh
        
            e.g. source /usr/local/Ascend/ascend-toolkit/set_env.sh
    2. 设置环境变量`ATB_BUILD_DEPENDENCY_PATH`
        
            e.g. source /usr/local/Ascend/nnal/atb/set_env.sh
                 export ATB_BUILD_DEPENDENCY_PATH=${ATB_HOME_PATH}
- 编译带有自定义算子的ATB加速库
    - bash scripts/build.sh customizeops

        需要注意此时为编译ATB加速库，需要位于ATB加速库工程的根目录 (e.g. `/home/ascend-transformer-boost`)
- 执行用例
    1. bash scripts/build.sh customizeops --customizeops_tests

        编译带有自定算子和测试用例的ATB加速库
    2. source ./output/atb/set_env.sh

        设置ATB加速库环境变量，需要位于ATB加速库工程的根目录 (e.g. `/home/ascend-transformer-boost`)
    3. cd ./build/ops_customize/ops/customize_blockcopy/tests/ && ./customize_blockcopy_test

        执行customize_block_copy Operation的测试用例