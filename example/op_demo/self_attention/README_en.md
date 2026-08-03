# SelfAttentionOperation C++ Demo

## Introduction

This directory contains the C++ call sample of SelfAttentionOperation.

## Instruction

- Source the installation paths of the CANN and NNAL packages.
    1. source [CANN installation path]/set_env.sh
        Default: source /usr/local/Ascend/ascend-toolkit/set_env.sh
    2. source [NNAL installation path]/set_env.sh
        Default: source /usr/local/Ascend/nnal/atb/set_env.sh
        ①. If building from the acceleration library source code, run `source [Acceleration library source path]/output/atb/set_env.sh`.
        Example: source ./ascend-transformer-boost/output/atb/set_env.sh

- Run the demo.

    ```sh
    bash build.sh
    ```

    **Note:**
    - When using cxx_abi=0 (default), set `D_GLIBCXX_USE_CXX11_ABI` to `0`:

        ```sh
        g++ -D_GLIBCXX_USE_CXX11_ABI=0 -I ...
        ```

    - When using cxx_abi=1, change `D_GLIBCXX_USE_CXX11_ABI` to `1`:

        ```sh
        g++ -D_GLIBCXX_USE_CXX11_ABI=1 -I ...
        ```

    - For the generated binary file `***_demo`, you can pass an additional int parameter as the device ID, which defaults to `0`. For example:

        ```sh
        ./self_attention_encoder_demo 0
        ```

    - The provided build script is used only to build and run `self_attention_encoder_demo.cpp`. To build other demos, replace `self_attention_encoder_demo` with the corresponding .cpp file name.

## Remarks

The data generated in the example does not represent the actual outputs. For details about data generation, see the Python use case directory under the root directory:
tests/apitest/opstest/python/operations/self_attention/

## Supported Products

SelfAttention is supported only in certain scenarios on the Atlas A2/A3 products. With encoder, the call method on the Atlas inference products is different from that on the Atlas A2/A3 products.

### Description

The provided demos apply to different products/scenarios. For details about the differences between scenarios, see the official website. You need to modify the build script accordingly.

#### self_attention_encoder_demo.cpp

+ Scenario: Basic FA encoder scenario, with key, CacheK, value, and CacheV passed separately.
+ The default build script can build and run the demo.
+ This demo can run only on the Atlas A2/A3 products.
+ This demo uses a full upper-triangular mask for demonstration.

**Parameter settings**

| Name  | Value                |
| :--------- | :------------------- |
| headNum    | 32                   |
| kvHeadNum  | 32                   |
| qkScale    | 1/sqrt(128)          |
| calcType   | `ENCODER`            |
| kernelType | `KERNELTYPE_DEFAULT` |
| maskType   | `MASK_TYPE_NORM`     |

**Data specifications**

| Tensor     | Data Type| Data Format| Dimension           | cpu/npu |
| --------------- | -------- | -------- | ------------------- | ------- |
| `query`         | float16  | nd       | [160,2048]          | npu     |
| `key`           | float16  | nd       | [160,2048]          | npu     |
| `value`         | float16  | nd       | [160,2048]          | npu     |
| `cacheK`        | float16  | nz       | [1, 10, 1024, 2048] | npu     |
| `cacheV`        | float16  | nz       | [1, 10, 1024, 2048] | npu     |
| `attentionMask` | float16  | nd       | [10, 1024, 1024]    | npu     |
| `tokenOffset`   | int32    | nd       | [10]                | cpu     |
| `seqLen`        | int32    | nd       | [10]                | cpu     |
| `layerId`       | int32    | nd       | [1]                 | npu     |
| **Output**      |
| `output`        | float16  | nd       | [160, 2048]         | npu     |

+ The first dimension of `q`, `k`, and `v` is the total sequence length, that is, `sum(seqlen)`.
+ The second dimension of `q`, `k`, and `v` combines the `headNum` and `headSize` axes. The actual value is `headNum(32) x headSize(128)`.

#### self_attention_encoder_inference_demo.cpp

+ Scenario: Basic FA encoder scenario on Atlas inference products, with key, CacheK, value, and CacheV passed separately.
+ This demo can run only on Atlas inference products.

**Parameter settings**

| Name  | Value                 |
| :--------- | :-------------------- |
| headNum    | 16                    |
| kvHeadNum  | 16                    |
| qkScale    | 1/sqrt(128)           |
| calcType   | `ENCODER`             |
| kernelType | `KERNELTYPE_DEFAULT`  |
| maskType   | `MASK_TYPE_UNDEFINED` |

**Data specifications**

| Tensor   | Data Type| Data Format| Dimension           | cpu/npu |
| ------------- | -------- | -------- | ------------------- | ------- |
| `query`       | float16  | nd       | [16, 256]           | npu     |
| `key`         | float16  | nd       | [16, 256]           | npu     |
| `value`       | float16  | nd       | [16, 256]           | npu     |
| `cacheK`      | float16  | nz       | [1, 1, 16, 256, 16] | npu     |
| `cacheV`      | float16  | nz       | [1, 1, 16, 256, 16] | npu     |
| `tokenOffset` | int32    | nd       | [1]                 | cpu     |
| `seqLen`      | int32    | nd       | [1]                 | cpu     |
| `layerId`     | int32    | nd       | [1]                 | npu     |
| **Output**    |
| `output`      | float16  | nd       | [16, 256]           | npu     |

#### self_attention_pa_encoder_demo.cpp

+ Scenario: FA with the PA encoder and the FA input format, with only key and value passed.
  + Different `headNum` and `kvHeadNum` values, with `headNum` divisible by `kvHeadNum`, enable Grouped Query Attention (GQA).
+ This demo can run only on Atlas A2/A3 products.
+ This demo uses a full upper-triangular mask for demonstration.

**Parameter settings**

| Name  | Value                |
| :--------- | :------------------- |
| headNum    | 32                   |
| kvHeadNum  | 16                   |
| qkScale    | 1/sqrt(128)          |
| calcType   | `PA_ENCODER`         |
| kernelType | `KERNELTYPE_DEFAULT` |
| maskType   | `MASK_TYPE_NORM`     |

**Data specifications**

| Tensor     | Data Type| Data Format| Dimension       | cpu/npu |
| --------------- | -------- | -------- | --------------- | ------- |
| `query`         | float16  | nd       | [3072, 32, 128] | npu     |
| `key`           | float16  | nd       | [3072, 16, 128] | npu     |
| `value`         | float16  | nd       | [3072, 16, 128] | npu     |
| `attentionMask` | float16  | nd       | [4, 1024, 1024] | npu     |
| `seqLen`        | int32    | nd       | [4]             | cpu     |
| **Output**      |
| `output`        | float16  | nd       | [3072, 32, 128] | npu     |

#### self_attention_pa_encoder_qwen_demo.cpp

+ Scenario: FA with the PA encoder and the FA input format, with only key and value passed.
  + Different `headNum` and `kvHeadNum` values, with `headNum` divisible by `kvHeadNum`, enable Grouped Query Attention (GQA).
+ This demo can run only on Atlas A2/A3 products.
+ The demo uses a compressed upper-triangular mask to handle long sequences.

**Parameter settings**

| Name  | Value                     |
| :--------- | :------------------------ |
| headNum    | 5                         |
| kvHeadNum  | 1                         |
| qkScale    | 1/sqrt(128)               |
| isTriuMask | 1                         |
| calcType   | `PA_ENCODER`              |
| kernelType | `KERNELTYPE_DEFAULT`      |
| maskType   | `MASK_TYPE_NORM_COMPRESS` |

**Data specifications**

| Tensor     | Data Type| Data Format| Dimension      | cpu/npu |
| --------------- | -------- | -------- | -------------- | ------- |
| `query`         | float16  | nd       | [1024, 5, 128] | npu     |
| `key`           | float16  | nd       | [1024, 1, 128] | npu     |
| `value`         | float16  | nd       | [1024, 1, 128] | npu     |
| `attentionMask` | float16  | nd       | [128, 128]     | npu     |
| `seqLen`        | int32    | nd       | [1]            | cpu     |
| **Output**      |
| `output`        | float16  | nd       | [1024, 5, 128] | npu     |

#### self_attention_prefix_encoder_demo.cpp

+ Scenario: FA with the prefix encoder and the PA input format, with key and value passed via blockTables.
  + This scenario allows `q` and `kv` to have different lengths, but requires:
    $$\forall i \lt len(seqLen), kvSeqLen[i] - seqLen[i] = 0 \ (mod \ 128) $$
+ This demo can run only on Atlas A2/A3 products.
+ This demo uses an Alibi upper-triangular mask and bias slopes for demonstration.

**Parameter settings**

| Name  | Value                       |
| :--------- | :-------------------------- |
| headNum    | 32                          |
| kvHeadNum  | 8                           |
| qkScale    | 1/sqrt(128)                 |
| isTriuMask | 1                           |
| calcType   | `PREFIX_ENCODER`            |
| kernelType | `KERNELTYPE_HIGH_PRECISION` |
| maskType   | `MASK_TYPE_ALIBI_COMPRESS`  |

**Data specifications**

| Tensor   | Data Type| Data Format| Dimension          | cpu/npu |
| ------------- | -------- | -------- | ------------------ | ------- |
| `query`       | float16  | nd       | [96, 32, 128]      | npu     |
| `key`         | float16  | nd       | [480, 128, 8, 128] | npu     |
| `value`       | float16  | nd       | [480, 128, 8, 128] | npu     |
| `blockTables` | float16  | nd       | [4, 4]             | npu     |
| `mask`        | float16  | nd       | [32, 96, 128]      | npu     |
| `seqLen`      | int32    | nd       | [4]                | cpu     |
| `kvSeqLen`    | int32    | nd       | [4]                | cpu     |
| `slopes`      | float32  | nd       | [128]              | npu     |
| **Output**    |
| `output`      | float16  | nd       | [96, 32, 128]      | npu     |
