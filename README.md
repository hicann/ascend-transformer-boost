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
 - run包使用<br>
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

## 参与贡献
 
1.  新建 br_personal/[employee_id]/[branch_name] 分支
2.  提交代码
3.  新建 merge Request

## 参考文档
**[CANN商用版文档](https://www.hiascend.com/document/detail/zh/canncommercial/80RC2/quickstart/quickstart/quickstart_18_0001.html)**
**[ATB商用版文档](https://www.hiascend.com/document/detail/zh/canncommercial/80RC2/developmentguide/acce/ascendtb/ascendtb_0001.html)**