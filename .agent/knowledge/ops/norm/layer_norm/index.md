# layer_norm — ATB Agent 知识条目

> **模板版本**: v1.0 | **状态**: complete | **最后更新**: 2026-07-06
>
> 本文件由 `atb-op-knowledge-extract` Skill 自动生成，供 Agent 快速理解算子完整知识，避免反复阅读源码。

---

## YAML Frontmatter（元数据）

```yaml
op:
  name: "layer_norm"
  category: "norm"
  tier: "M"
  type: "single"

source:
  repo_path: "src/ops/ops_infer/layer_norm/"
  kernel_path: "src/kernels/kernels/norm/layernorm/"
  param_header: "include/atb/infer_op_params.h"
  ini_file: "atb_ops_info.ini"

knowledge:
  status: "complete"
  last_extracted: "2026-07-06"
  last_source_commit: "HEAD"
  extractor_version: "1.0.0"

test:
  atk_test_path: "atk_test/atk_cida_atb/atb/infer/LayerNormOperation/"
  has_atk_tests: true
```

---

## 1. Source File Map（源文件地图）

### 1.1 文件清单

| # | 文件（相对路径） | 角色 | 关键内容 |
|---|-----------------|------|---------|
| 1 | `layer_norm_operation.h` | Operation 定义 | 类声明、InferShape/Setup/CreateRunner 签名 |
| 2 | `layer_norm_operation.cpp` | Operation 定义 | 参数校验（3 种 NormType）、CreateRunner 决策树、InferShape（含量化 dtype 断点）、SetupCheck、维度一致性 |
| 3 | `layer_norm_aclnn_runner.h` | ACLNN Runner | LayerNormAclnnRunner 声明、LoadMethod |
| 4 | `layer_norm_aclnn_runner.cpp` | ACLNN Runner | BuildAclnnVariantPack（3 in/1 out）、WorkspaceSize 计算、LaunchAclnnKernel（aclnnLayerNorm 调用） |
| 5 | `layer_norm_ops_runner.h` | Ops Runner | LayerNormOpsRunner 声明、5 种 Graph Builder 签名 |
| 6 | `layer_norm_ops_runner.cpp` | Ops Runner | 8 种 Graph 构建（NORM/PRENORM/POSTNORM × 量化/非量化/DynamicQuant），统一通过 NormOperation kernel |
| 7 | `layernorm_kernel.cpp` | Kernel 入口 | Norm Tiling 常量、kernel 启动 |

### 1.2 外部依赖（关键头文件）

| 头文件 | 提供的类型/函数 |
|--------|----------------|
| `include/atb/infer_op_params.h` | `LayerNormParam` 结构体（含 NormParam / PreNormParam / PostNormParam / LayerNormType） |
| `atb/utils/aclnn_util.h` | CallAclCreateTensor、LoadFromSharedObjectFile |
| `mki/utils/platform/platform_info.h` | PlatformType（ASCEND_950 检测） |
| `<aclnn/opdev/op_errno.h>` | aclnnStatus、ACLNN_ERR_* |

---

## 2. Reading Order（推荐阅读顺序）

```
阶段 1: 参数层（回答"这个 Op 接受什么参数？"）
  ├── [1] include/atb/infer_op_params.h  L668-730 → 读 LayerNormParam 结构体
  └── [2] atb_ops_info.ini               → 查 LayerNorm 注册段

阶段 2: 调度层（回答"如何选择 Runner？支持哪些平台？"）
  ├── [3] layer_norm_operation.cpp  L104-155   → 读 CreateOperation()（参数校验 + 950 平台 Gate）
  ├── [4] layer_norm_operation.cpp  L348-357   → 读 CreateRunner()（ACLNN vs Ops 路由决策）
  └── [5] layer_norm_operation.h               → 读类声明

阶段 3: 执行层（回答"数据如何在 NPU 上计算？"）
  ├── [6] layer_norm_aclnn_runner.cpp          → 读 ACLNN 路径（仅 950 NORM 无量化）
  ├── [7] layer_norm_ops_runner.cpp  L80-214   → 读 8 种 Ops Graph 构建
  └── [8] layernorm_kernel.cpp                 → 读 NormOperation Kernel

阶段 4: 约束层（回答"有什么边界条件？"）
  ├── [9] layer_norm_operation.cpp  L259-321   → 读 InferShapeCheck + SetupCheck + ParamCheck
  └── [10] 测试用例 YAML / JSON                → 读参数范围限制
```

---

## 3. Parameter Constraints（参数约束）

### 3.1 参数结构体

**位置**: `include/atb/infer_op_params.h`（L668-730）

```cpp
struct LayerNormParam {
    enum LayerNormType : int {
        LAYER_NORM_UNDEFINED = 0,
        LAYER_NORM_NORM,       // 标准 LayerNorm
        LAYER_NORM_PRENORM,    // 前置 Norm（含残差）
        LAYER_NORM_POSTNORM,   // 后置 Norm（含残差）
    };

    struct NormParam {
        QuantType quantType = QUANT_UNQUANT;  // QUANT_UNQUANT | QUANT_INT8
        float epsilon = 1e-5;
        int32_t beginNormAxis = 0;
        int32_t beginParamsAxis = 0;
        DynamicQuantType dynamicQuantType = DYNAMIC_QUANT_UNDEFINED;
    };

    struct PreNormParam {
        QuantType quantType = QUANT_UNQUANT;  // 仅支持 QUANT_UNQUANT
        float epsilon = 1e-5;
        uint64_t opMode = 0;                  // 仅支持 0（高精度）
        float zoomScaleValue = 1.0f;
    };

    struct PostNormParam {
        QuantType quantType = QUANT_UNQUANT;  // QUANT_UNQUANT | QUANT_INT8
        float epsilon = 1e-5;
        uint64_t opMode = 0;                  // 仅支持 0（高精度）
        float zoomScaleValue = 1.5f;
    };
};
```

### 3.2 参数约束表

| 参数名 | 类型 | 默认值 | 合法范围 | 约束条件 | 来源 |
|--------|------|--------|---------|---------|------|
| `layerType` | enum | 0 | 0-3 | NORM/PRENORM/POSTNORM；Ascend950 仅支持 NORM | param struct |
| `normParam.quantType` | enum | QUANT_UNQUANT | QUANT_UNQUANT, QUANT_INT8 | PRENORM 不支持量化 | CreateOperation |
| `normParam.epsilon` | float | 1e-5 | ≥ 2e-38 | 非零正值，否则返回 ERROR_INVALID_PARAM | NormParamCheck |
| `normParam.beginNormAxis` | int32 | 0 | -dimNum..dimNum-1 | beginNormAxis + gammaDimNum == xDimNum | ParamCheck |
| `normParam.dynamicQuantType` | enum | UNDEFINED | UNDEFINED, SYMMETRIC | 不支持 ASYMMETRIC | NormParamCheck |
| `preNormParam.quantType` | enum | QUANT_UNQUANT | QUANT_UNQUANT | 仅支持非量化 | PreNormParamCheck |
| `preNormParam.opMode` | uint64 | 0 | 0 | 仅支持 0（高精度） | PreNormParamCheck |
| `preNormParam.zoomScaleValue` | float | 1.0f | — | PreNorm 缩放因子 | param struct |
| `postNormParam.quantType` | enum | QUANT_UNQUANT | QUANT_UNQUANT, QUANT_INT8 | — | PostNormParamCheck |
| `postNormParam.opMode` | uint64 | 0 | 0 | 仅支持 0 | PostNormParamCheck |

### 3.3 Shape 约束

| 约束项 | 条件 | 错误信息 |
|--------|------|---------|
| gamma 维度 | `gammaDimNum ≤ xDimNum`（NORM 非量化） | "gammaTensor dimNum should be smaller or equal to xTensor" |
| beginNormAxis 匹配 | `beginNormAxis + gammaDimNum == xDimNum`（非负）/ `abs(beginNormAxis) == gammaDimNum`（负） | "beginNormAxis must make remaining dims match gamma shape" |
| 最后维对齐 | `lastDim % 16 == 0`（量化/PRENORM/POSTNORM 模式） | "should align 32"（32 bytes = 16 × uint16） |
| PRENORM/POSTNORM 残差 | residual shape == x shape | "Residual tensor should be the same as x tensor" |
| 所有输入最后维一致 | 所有 inTensors 最后一维相等 | "last dims of inTensors does not match" |
| DynamicQuant dimNum | x dimNum > 1（SymDynamicQuant） | "dim numbers of inTensor[0] should be greater than one" |
| DynamicQuant lastDim | lastDim ≤ 12288（SymDynamicQuant） | "Length of the last dim is invalid" |
| Quant scale/offset | dimNum==1, dims[0]==1 | "dim numbers of quant param intensors should be one" |

### 3.4 Dtype 约束

| Tensor | 支持 Dtype | 条件 |
|--------|-----------|------|
| x (input) | fp16, fp32, bf16(非910B) | — |
| gamma/beta | 同 x dtype | — |
| residual | 同 x dtype | PRENORM/POSTNORM |
| scale/offset | int8 path 特定 | NORM Quant 静态量化 |
| output (非量化) | 同 x dtype | QUANT_UNQUANT |
| output (静态量化) | int8 | QUANT_INT8 且非动态量化 |
| output scale (动态量化) | fp32 | DynamicQuant |
| output offset (非对称动态量化) | fp32 | DynamicQuant + ASYMMETRIC（当前不支持） |

---

## 4. Computation Pipeline（计算管线）

### 4.4 单阶段算子

```yaml
pipeline_type: single_stage
note: >
  LayerNorm 为单阶段归一化算子：逐元素计算均值/方差 → 归一化 → affine 变换。
  无跨阶段 dtype 转换。
  唯一 dtype 断点：静态量化时 input(fp16) → output(int8)。
  动态量化时：input(fp16) → output(int8) + scale(fp32) [+ offset(fp32)]。
```

### 4.3 Dtype 断点（Input ≠ Output）

| 条件 | Input Dtype | Output Dtype | 触发文件 |
|------|------------|-------------|---------|
| NORM + QUANT_INT8 + 非动态量化 | fp16 | int8 | `infer_op_params.h:L236` |
| POSTNORM + QUANT_INT8 | fp16 | int8 | `infer_op_params.h:L252` |
| NORM + DynamicQuant | fp16 | int8 + fp32(scale) | `infer_op_params.h:L240-242` |

### 4.2 精度控制标志

| 标志 | 位置 | 取值 | 效果 |
|------|------|------|------|
| `preNormParam.opMode` | `infer_op_params.h` | 0 | `0`=高精度（唯一支持） |
| `postNormParam.opMode` | `infer_op_params.h` | 0 | `0`=高精度（唯一支持） |

---

## 5. Execution Paths（执行路径）

### 5.1 Runner 选择决策树

```
LayerNormOperation::CreateRunner()
  │
  ├── [条件] PlatformType == ASCEND_950
  │     && layerType == LAYER_NORM_NORM
  │     && normParam.quantType == QUANT_UNQUANT
  │     └── → LayerNormAclnnRunner
  │           调用: aclnnLayerNormGetWorkspaceSize() → aclnnLayerNorm()
  │           输入: x, gamma, beta (3 in / 1 out)
  │
  └── [fallback] 所有其他平台 + 所有其他变体
        └── → LayerNormOpsRunner
              8 种 Graph 变体:
              ├── BuildLayerNormGraph          (NORM, 非量化, 3in→1out)
              ├── BuildLayerNormQuantGraph      (NORM, 静态量化, 5in→1out)
              ├── BuildLayerNormDynamicQuantGraph (NORM, 动态量化, 3in→3out)
              ├── BuildPreLayerNormGraph        (PRENORM, 非量化, 4in→2out)
              ├── BuildPostLayerNormGraph       (POSTNORM, 非量化, 4in→1out)
              ├── BuildPostLayerNormQuantGraph  (POSTNORM, 静态量化, 6in→2out)
              └── (PRENORM 量化不支持)
```

### 5.2 平台支持矩阵

| 平台 | Runner | 特性支持 | 限制 |
|------|--------|---------|------|
| Atlas 910B | OpsRunner | NORM/PRENORM/POSTNORM + 量化/动态量化 | bf16 不支持 |
| Atlas 950 | AclnnRunner (NORM) / OpsRunner (其他) | 仅 NORM 无量化走 ACLNN | 拦截 PRENORM/POSTNORM/量化（CreateOperation 直接拒绝） |

### 5.3 关键 API 调用链

```
用户 API: atb::CreateOperation(param, &op)
  └─ LayerNormOperation 构造 → 设置 operationIr_ key
       └─ op->Setup(inTensors, outTensors) → SetupCheckImpl
            ├─ ParamCheck (x vs gamma 维度关系)
            ├─ InTensorsDimCheck (残差 shape 一致性)
            ├─ LastDimCheck (对齐 + 动态量化约束)
            ├─ QuantTensorCheck (量化参数 shape)
            └─ InferShapeImpl → 设置 outTensor 描述

       └─ op->Execute(context, inTensors, outTensors)
            ├─ [ACLNN路径/950] LayerNormAclnnRunner
            │    ├─ BuildAclnnVariantPack (aclCreateTensor ×6)
            │    ├─ aclnnLayerNormGetWorkspaceSize(...)
            │    └─ aclnnLayerNorm(...)
            └─ [OPS路径/其他] LayerNormOpsRunner
                 └─ Mki::NormOperation kernel (单节点 Graph)
```

---

## 6. Kernel Dependencies（Kernel 依赖）

### 6.1 Kernel 清单

| Kernel 名称 | 文件 | 用途 | 平台 |
|------------|------|------|------|
| `NormOperation` | `layernorm_kernel.cpp` | 统一 Norm kernel（LayerNorm/PreNorm/PostNorm + 量化变体） | 910B/950 |

### 6.2 Tiling 约束

| 约束项 | 值 | 来源 |
|--------|---|------|
| 最后维对齐（量化/PRENORM/POSTNORM） | 32 bytes（16 × uint16） | `layer_norm_operation.cpp:ALIGNMENT=32` |
| DynamicQuant 最大 lastDim | 12288 | `DYNAMIC_QUANT_MAX_TENSOR_LENGTH` |
| 单 kernel 节点 | 1 node per graph | 所有 Graph Builder 一致 |

### 6.3 Workspace 大小

| 场景 | Workspace 来源 | 说明 |
|------|---------------|------|
| ACLNN 路径 | `aclnnLayerNormGetWorkspaceSize()` 动态计算 | 由 CANN 运行时返回 |
| OPS 路径 | `atbVariantPack_.workspaceBufferSize` | Mki 框架自动管理 |

---

## 7. Known Issues（已知问题与边界条件）

### 7.1 已知问题

| # | 问题描述 | 影响范围 | 平台 | 规避方式 | 状态 |
|---|---------|---------|------|---------|------|
| 1 | Ascend950 仅支持 NORM + 非量化模式 | PRENORM/POSTNORM/量化被 CreateOperation 直接拦截 | 950 | 使用 910B 或等待 950 后续版本 | wa |
| 2 | PRENORM 不支持量化 | PRENORM + quantType 任意非 QUANT_UNQUANT | 全部 | 使用 POSTNORM + 量化 或 NORM + 量化 | lim |
| 3 | PreNormParam.opMode==1（高性能）暂不支持 | PRENORM 非默认 opMode | 全部 | 使用 opMode=0（高精度） | lim |
| 4 | 动态量化不支持 ASYMMETRIC | NORM + DynamicQuant + ASYMMETRIC | 全部 | 使用 SYMMETRIC 动态量化 | lim |

### 7.2 边界条件

| 条件 | 行为 | 错误码 |
|------|------|--------|
| epsilon < 2e-38 | "Invalid epsilon, it's recommended to init a nonzero value for eps." | ERROR_INVALID_PARAM（由 NormParamCheck） |
| beginNormAxis 不匹配 gamma 维度 | "beginNormAxis must make remaining dims match gamma shape" | ERROR_INVALID_PARAM（由 ParamCheck） |
| dynamicQuantType == ASYMMETRIC | "dynamicQuantType not support DYNAMIC_QUANT_ASYMMETRIC" | ERROR_INVALID_PARAM |
| 量化且 lastDim 不对齐 32B | "inTensor shape is not support, should align 32" | ERROR_INVALID_TENSOR_SIZE |
| PRENORM/POSTNORM 残差 shape 不一致 | "Residual tensor should be the same as x tensor." | ERROR_INVALID_TENSOR_DIM |
| 静态量化 scale/offset dimNum ≠ 1 | "dim numbers of quant param intensors should be one" | ERROR_INVALID_TENSOR_SIZE |
| DynamicQuant + lastDim > 12288 | "Length of the last dim is invalid, it should be no larger than 12288." | ERROR_INVALID_TENSOR_SIZE |

---

## 8. Test Coverage（测试覆盖）

### 8.1 ATK 测试用例

| 测试文件 | 类型 | 覆盖参数 |
|---------|------|---------|
| `all_LayerNormOperation_Accu_Generalization.json` | 精度泛化 | NORM 非量化，全面 dtype/shape 组合 |
| `all_LayerNormOperation_Dete_Generalization.json` | 确定性 | NORM 算子收敛性/稳定性 |
| `all_LayerNormOperation_Dynamic_Quant_Accu_Generalization.json` | 精度 | NORM 动态量化 |
| `all_LayerNormOperation_Abnormal.json` | 异常 | 边界/非法参数 |
| `all_LayerNormOperation_Perf.json` | 性能 | NORM Benchmark |
| `all_LayerNormOperation_Memo_Generalization.json` | 内存 | NORM Workspace |

### 8.2 测试执行命令

```bash
# 生成用例
cd atk_test/atk_cida_atb/atb/infer/LayerNormOperation/
atk case -f nodes_LayerNormOperation_Accu.yaml

# 执行精度测试
atk task -n nodes_LayerNormOperation_Accu.yaml \
    -c result/<yaml>/json/all_LayerNormOperation_Accu_Generalization.json \
    -p ./ --task accuracy -ap ../../common
```

### 8.3 覆盖盲区

| 未覆盖的参数组合 | 原因 | 优先级 |
|----------------|------|--------|
| PRENORM 精度泛化 | PRENORM 变体较少使用 | low |
| POSTNORM 量化精度 | POSTNORM + INT8 组合较少 | medium |
| Ascend950 精度验证 | 950 仅支持 NORM 无量化（覆盖范围小） | low |

---

## 9. Related Ops（相关算子）

### 9.1 功能相似

| 算子 | 关系 | 区别 |
|------|------|------|
| `rms_norm` | 同类 Norm 算子 | RMS Norm 无中心化（仅 scale，无 bias） |
| `cohere_layernorm` | Cohere 自定义 Norm | 自定义 epsilon/axis 行为 |
| `layer_norm_with_stride` | 带 stride 变体 | 增加 stride 参数支持非连续 Tensor |

### 9.2 组合使用（上下游）

```
SelfAttention → LayerNorm (POSTNORM) → FFN → LayerNorm (PRENORM) → ...
```

### 9.3 变体关系

| 变体 | 与主 Op 的区别 |
|------|---------------|
| `layer_norm_with_stride` | 增加 stride 参数 |
| `rms_norm` | 无中心化（无 mean 减除） |

---

## Appendix A: 自动提取元数据

| 字段 | 值 |
|------|-----|
| 提取耗时 | ~120s |
| 读取文件数 | 6 |
| 置信度评分 | 90/100 |
| 需人工审核 | false |
| 审核原因 | — |

---

## Appendix B: 变更日志

| 日期 | 变更类型 | 变更章节 | 触发 Commit | 说明 |
|------|---------|---------|------------|------|
| 2026-07-06 | auto | all | `HEAD` | 初始提取（atb-op-knowledge-extract v1.0.0） |
