# ATB加速库对外头文件和库文件说明

## 接口分类

ATB（AscendTransformerBoost）加速库对外提供 C++ 和 C 两种接口风格，覆盖推理与训练场景。接口按功能分为以下几类：

| 接口类别 | 命名约定 | 描述 |
|---------|---------|------|
| 基础类型与工具 | `atb::` 命名空间，无特殊前缀 | 包含基本数据结构（`TensorDesc`、`Dims`、`VariantPack`）、错误码（`Status`）、工具函数（`Utils`）、容器（`SVector`）等 |
| 上下文管理 | `atb::Context` / `CreateContext` / `DestroyContext` | 运行时上下文管理，包括 Stream 设置、Tiling 属性、启动模式（单算子/图模式）等 |
| 算子定义与执行 | `atb::Operation` / `CreateOperation` / `DestroyOperation` | 算子抽象基类，提供 `Setup`（预计算 workspace 大小）、`Execute`（执行算子）等生命周期接口 |
| 图构建 | `atb::GraphOpBuilder` / `CreateGraphOpBuilder` | 将多个算子组合成计算图，支持 Reshape、AddOperation、Build 等图构建操作 |
| 推理算子参数 | `atb::infer::*Param` | 推理算子参数结构体，如 `LinearParam`、`SelfAttentionParam`、`RmsNormParam` 等 |
| 训练算子参数 | `atb::train::*Param` | 训练算子参数结构体，如 `FastSoftMaxParam`、`LaserAttentionParam`、`RopeGradParam` 等 |
| 通信域接口 | `atb::Comm::` 前缀 | HCCL 通信域管理，包括通信域创建/销毁等 |
| C 风格算子接口 | `Atb` 前缀，两阶段调用 | 针对特定异构算子的 C 接口，如 `AtbMLA`、`AtbFusedAddTopkDiv`、`AtbRingMLA` 等 |
| 通用算子参数 | `atb::common::*Param` | 通用算子参数，如 Event 同步（`EventParam`）、条件分支（`IfCondParam`） |
| 插件扩展基类 | `atb::OperationInfra` | 面向自定义/插件算子的基础设施类，继承自 `atb::Operation` |

## 调用接口依赖的头文件和库文件说明

安装 ATB 加速库后，编译、运行应用程序时需要引用 ATB 接口的头文件和库文件。

头文件位于 `${ATB_HOME_PATH}/include/atb/` 目录下，库文件位于 `${ATB_HOME_PATH}/lib/` 目录下。其中 `${ATB_HOME_PATH}` 为 ATB 加速库安装路径，通常通过 `source <安装目录>/atb/set_env.sh` 设置。

以 root 用户安装为例，安装后文件存储路径为：

- 头文件：`/usr/local/Ascend/atb/cxx_abi_1/include/atb/` 或 `/usr/local/Ascend/atb/cxx_abi_0/include/atb/`
- 库文件：`/usr/local/Ascend/atb/cxx_abi_1/lib/` 或 `/usr/local/Ascend/atb/cxx_abi_0/lib/`

> **说明**：`cxx_abi` 取值取决于编译环境的 C++ ABI 版本。若环境中的 PyTorch 使用 CXX11 ABI 编译，则使用 `cxx_abi_1`；否则使用 `cxx_abi_0`。可通过 `source atb/set_env.sh --cxx_abi=1` 或 `--cxx_abi=0` 指定。
> **须知**：编译 ATB 接口程序时，请按照 include 的头文件依赖对应的库文件。若仅使用推理功能，只需链接 `libatb.so`；若使用训练算子，需额外链接 `libatb_train.so`。引用多余的 so 文件可能导致版本功能异常或后续版本升级时存在兼容性问题。

用户需要根据实际使用的 ATB 接口来 include 依赖的头文件。推荐直接 include `atb/atb_infer.h` 作为总入口头文件，该文件已包含所有对外头文件的引用。各头文件用途及对应的库文件如下表所示：

| 定义接口的头文件 | 用途 | 对应的库文件 |
|---------------|------|------------|
| atb/atb_infer.h | 总入口头文件，包含所有对外头文件的引用 | libatb.so |
| atb/types.h | 基本数据结构定义，包括 `Status`（错误码）、`Dims`（维度描述）、`TensorDesc`（张量描述符）、`Tensor`（张量数据）、`VariantPack`（输入输出张量包）等 | libatb.so |
| atb/svector.h | 栈上 Vector 容器 `SVector<T>` 定义 | libatb.so |
| atb/context.h | `Context` 接口类，用于设置算子执行上下文，如执行 Stream、Tiling 属性、启动模式等；以及 `CreateContext`/`DestroyContext` 工厂函数 | libatb.so |
| atb/operation.h | `Operation` 算子抽象基类，提供算子生命周期接口（`Setup`/`Execute`）、形状推导（`InferShape`）以及 `CreateOperation`/`DestroyOperation` 等工厂函数 | libatb.so |
| atb/operation_infra.h | `OperationInfra` 插件算子基础设施类，继承自 `Operation` | libatb.so |
| atb/infer_op_params.h | 推理算子参数定义，包含 `LinearParam`、`SelfAttentionParam`、`RmsNormParam`、`ActivationParam`、`ElewiseParam`、`KvCacheParam`、`SoftmaxParam`、`LayerNormParam` 等 | libatb.so |
| atb/train_op_params.h | 训练算子参数定义，包含 `FastSoftMaxParam`/`FastSoftMaxGradParam`、`LaserAttentionParam`/`LaserAttentionGradParam`、`StridedBatchMatmulParam`、`RopeGradParam` 等 | libatb_train.so |
| atb/common_op_params.h | 通用算子参数定义，包括 `EventParam`（Stream 同步事件）和 `IfCondParam`（运行时条件分支） | libatb.so |
| atb/graph_op_builder.h | `GraphOpBuilder` 图构建器接口，支持将多个算子组合为计算图 | libatb.so |
| atb/utils.h | `Utils` 工具类，提供版本查询（`GetAtbVersion`）、张量大小/元素数计算、量化参数转换、日志级别设置等静态方法 | libatb.so |
| atb/atb_acl.h | C 风格 ACLNN 算子接口，针对特定异构算子（如 `AtbMLA`、`AtbFusedAddTopkDiv`、`AtbRingMLA`、`AtbPagedCacheLoad` 等）的两阶段调用函数 | libatb.so |
| atb/comm.h | 通信域接口，提供 HCCL 通信域管理函数：`CreateHcclComm`、`CreateHcclCommByRankTableFile`、`DestoryHcclComm` 等 | libatb.so |
