# Ascend Transformer Boost

## 介绍
Ascend Transformer Boost加速库（下文简称为ATB加速库）是一款高效、可靠的加速库，基于华为Ascend AI处理器，专门为Transformer模型的训练和推理而设计。

ATB加速库采用了一系列优化策略，包括算法优化、硬件优化和软件优化，能够显著提升Transformer模型的训练和推理速度，同时降低能耗和成本。具体来说，ATB加速库通过优化矩阵乘法等核心算子和注意力机制的实现方式，实现了对Transformer模型的高效加速。此外，ATB加速库还充分利用了Ascend AI处理器的硬件特性，如算力、存储带宽和内存带宽，通过硬件加速和数据重用等技术，进一步提升了性能和效率。ATB加速库目前提供了底层基础的高性能算子以及高效的算子组合技术（Graph图算子），同时上层支持对接多种模型框架如PyTorch、MindSpore、Paddle。

总而言之，ATB加速库中包含了各类Transformer类模型的高度优化模块，在各种应用场景中发挥重要作用，为模型的训练和推理提供了强有力的支持。


## 软件架构
加速库接口功能主要分成三部分：
- 提供基础原生的算子（Operation），用户可以根据需求使用对应的算子完成计算功能。
- 提供图算子机制，用户根据模型设计对应的图算子，使用加速库提供的原生算子和创建的自定义算子创建图算子，完成相应的计算。
- 提供插件（Plugin）机制，用户可以根据自己的需求创建自定义的算子。

## 环境构建
    - [安装CANN环境](https://www.hiascend.com/document/detail/zh/canncommercial/80RC22/softwareinst/instg/instg_0001.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit)
    - 设置cann环境变量
    ```sh
    source [cann安装路径]（默认为/usr/local/Ascend/ascend-toolkit）/set_env.sh
    ```

## 使用教程
 
 - 加速库编译<br>
    编译加速库，设置加速库环境变量：
    ```sh
    > cd ascend-transformer-boost
    > bash scripts/build.sh
    > source output/atb/set_env.sh
    ```
 - 无法获取ascend-op-common-lib代码仓时，可通过安装nnal软件包获取对应so文件<br>
    - 安装步骤可参考 `run包使用`
    - 代码及软件包版本对应关系：<br>
        nnal软件包需保持和toolkit及kernels软件包版本一致
        |CANN|代码分支|
        |-|-|
        |CANN 8.1.RC1|br_feature_cann_8.2.RC1_0515POC_20250630|

    - 执行 
        ```sh
        source {install path}/nnal/atb/set_env.sh
        export ATB_BUILD_DEPENDENCY_PATH=${ATB_HOME_PATH}
        ```
 - run包使用<br>
    - run包获取
    1. 进入网址：https://www.hiascend.com/developer/download/commercial
    2. 产品系列选择服务器，产品型号根据设备型号选择，选择所需解决方案版本，随后在CANN区域选择软件包跟随指引即可获取相关run包
    - 软件包名为：Ascend-cann-atb_{version}_linux-{arch}.run <br>
    其中，{version}表示软件版本号，{arch}表示CPU架构。
    - 安装run包（需要依赖cann环境）
        ```sh
        chmod +x 软件包名.run # 增加对软件包的可执行权限
        ./软件包名.run --check # 校验软件包安装文件的一致性和完整性
        ./软件包名.run --install # 安装软件，可使用--help查询相关安装选项
        ```
        出现提示`xxx install success!`则安装成功

## 使用说明
- 加速库算子单元测试
    - c++
    ```sh
    bash scripts/build.sh unittest
    ```
    - python
    ```sh
    bash scripts/build.sh pythontest
    ```
    - csv
    ```sh
    bash scripts/build.sh csvopstest
    ```

## 日志
- 加速库日志现在已经部分适配CANN日志 请参考
    - **[CANN商用版文档/环境变量参考/辅助功能/日志/场景说明](https://www.hiascend.com/document/detail/zh/canncommercial/80RC22/apiref/envvar/envref_07_0109.html)**
- 由于CANN日志暂时没有ATB模块，ASCEND_MODULE_LOG_LEVEL请勿设置ATB

## 参与贡献
 
1.  新建 br_personal/[employee_id]/[branch_name] 分支
2.  提交代码
3.  新建 merge Request

## 样例安全声明
`example`目录下的样例旨在提供快速上手、开发和调试ATB特性的最小化实现，其核心目标是使用最精简的代码展示ATB核心功能，**而非提供生产级的安全保障**。与成熟的生产级使用方法相比，此样例中的安全功能（如输入校验、边界校验）相对有限。

ATB不推荐用户直接将样例作为业务代码，也不保证此种做法的安全性。若用户将`example`中的示例代码应用在自身的真是业务场景中且发生了安全问题，则由用户自行承担。

## 参考文档
**[CANN商用版文档](https://www.hiascend.com/document/detail/zh/canncommercial/80RC2/quickstart/quickstart/quickstart_18_0001.html)**
**[ATB商用版文档](https://www.hiascend.com/document/detail/zh/canncommercial/80RC2/developmentguide/acce/ascendtb/ascendtb_0001.html)**