# MlaPreprocessOperation C++ Demo

## Introduction

This directory contains the C++ call sample of MlaPreprocessOperation.

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

    - The provided build script is used only to build and run `mlapo_demo.cpp`. To build other demos, replace `mlapo_demo` with the corresponding .cpp file name.

## Remarks

The data generated in the example does not represent the actual outputs. For details about data generation, see the Python use case directory under the root directory:
tests/apitest/opstest/python/operations/mla_preprocess/

### Description

The provided demos apply to different products/scenarios. For details about the differences between scenarios, see the official website.
When calling the demo binary file, enter `dtype`, `tokenNum`, and `headNum` for the key to control the data type and shape.
    `dtype` corresponds to the input data type, which can be either float16 or bf16, corresponding to the two columns in the following data specifications.

mlapo_demo.cpp:

+ Scenario: MLAPO INT8 quantization with RoPE splitting
  + ctkv and qNope are per_head statically and symmetrically quantized to INT8 type; kvCache and query are split using RoPE, and krope and ctkv are converted to NZ format for output.

**Parameter settings**

| Name | Value        |
| :-------- | :----------- |
| cacheMode | INT8_NZCACHE |

**Data specifications**

| Tensor    | Data Type    | Data Format| Dimension                       |
| -------------- | ------------ | -------- | ------------------------------- |
| **Input**      |
| `input`        | float16/bf16 | nd       | [tokenNum, 7168]                |
| `gamma0`       | float16/bf16 | nd       | [7168]                          |
| `beta0`        | float16/bf16 | nd       | [7168]                          |
| `quantScale0`  | float16/bf16 | nd       | [1]                             |
| `quantOffset0` | int8         | nd       | [1]                             |
| `wdqkv`        | int8         | nz       | [1, 224, 2112, 32]              |
| `deScale`      | int64/float  | nd       | [2112]                          |
| `bias0`        | int32        | nd       | [2112]                          |
| `gamma1`       | float16/bf16 | nd       | [1536]                          |
| `beta1`        | float16/bf16 | nd       | [1536]                          |
| `quantScale1`  | float16/bf16 | nd       | [1]                             |
| `quantOffset1` | int8         | nd       | [1]                             |
| `wuq`          | int8         | nz       | [1, 48, 24576, 32]              |
| `deScale1`     | int64/float  | nd       | [24576]                         |
| `bias1`        | int32        | nd       | [headNum * 192]                 |
| `gamma2`       | float16/bf16 | nd       | [512]                           |
| `cos`          | float16/bf16 | nd       | [tokenNum, 64]                  |
| `sin`          | float16/bf16 | nd       | [tokenNum, 64]                  |
| `wuk`          | float16/bf16 | nz       | [headNum, 32, 128, 16]          |
| `kvCache`      | int8         | nz       | [64, headNum * 512/32, 128, 32] |
| `kvCacheRope`  | float16/bf16 | nd       | [64, headNum * 64/16, 128, 16]  |
| `slotmapping`  | int32        | nd       | [tokenNum]                      |
| `ctkvScale`    | float16/bf16 | nd       | [1]                             |
| `qNopeScale`   | float16/bf16 | nd       | [headNum]                       |
| **Output**     |
| `qOut0`        | int8         | nd       | [tokenNum, headNum, 512]        |
| `kvCacheOut0`  | int8         | nz       | [64, headNum * 512/32, 128, 32] |
| `qOut1`        | float16/bf16 | nd       | [tokenNum, headNum, 64]         |
| `kvCacheOut1`  | float16/bf16 | nz       | [64, headNum * 64/16, 128, 16]  |

> Default values: `dtype = float16`, `tokenNum = 4`, `headNum = 128`

mlapo_ds_demo.cpp:

+ Scenario: MLAPo for DeepSeek
  + Use RoPE to split the kvCache and query, and perform per_tensor static asymmetric quantization.

**Parameter settings**

| Name    | Value        |
| :----------- | :----------- |
| wdqDim       | 1536         |
| qRopeDim     | 64           |
| kRopeDim     | 64           |
| epsilon      | 1e-5         |
| qRotaryCoeff | 2            |
| kRotaryCoeff | 2            |
| transposeWdq | true         |
| transposeWuq | true         |
| transposeWuk | true         |
| cacheMode    | INT8_NZCACHE |

**Data specifications**

| Tensor    | Data Type    | Data Format| Dimension                |
| -------------- | ------------ | -------- | ------------------------ |
| **Input**      |
| `input`        | float16/bf16 | nd       | [tokenNum, 7168]         |
| `gamma0`       | float16/bf16 | nd       | [7168]                   |
| `beta0`        | float16/bf16 | nd       | [7168]                   |
| `quantScale0`  | float16/bf16 | nd       | [1]                      |
| `quantOffset0` | int8         | nd       | [1]                      |
| `wdqkv`        | int8         | nz       | [2112, 7168]             |
| `deScale`      | int64/float  | nd       | [2112]                   |
| `bias0`        | int32        | nd       | [2112]                   |
| `gamma1`       | float16/bf16 | nd       | [1536]                   |
| `beta1`        | float16/bf16 | nd       | [1536]                   |
| `quantScale1`  | float16/bf16 | nd       | [1]                      |
| `quantOffset1` | int8         | nd       | [1]                      |
| `wuq`          | int8         | nz       | [24576, 1536]            |
| `deScale1`     | int64/float  | nd       | [24576]                  |
| `bias1`        | int32        | nd       | [headNum * 192]          |
| `gamma2`       | float16/bf16 | nd       | [512]                    |
| `cos`          | float16/bf16 | nd       | [tokenNum, 64]           |
| `sin`          | float16/bf16 | nd       | [tokenNum, 64]           |
| `wuk`          | float16/bf16 | nd       | [tokenNum, 128, 512]     |
| `kvCache`      | float16/bf16 | nd       | [161, 128, 1, 512]       |
| `kvCacheRope`  | float16/bf16 | nd       | [161, 128, 1, 64]        |
| `slotmapping`  | int32        | nd       | [tokenNum]               |
| `ctkvScale`    | float16/bf16 | nd       | [1]                      |
| `qNopeScale`   | float16/bf16 | nd       | [1]                      |
| **Output**     |
| `qOut0`        | int8         | nd       | [tokenNum, headNum, 512] |
| `kvCacheOut0`  | int8         | nz       | [161, 128, 1, 512]       |
| `qOut1`        | float16/bf16 | nd       | [tokenNum, headNum, 64]  |
| `kvCacheOut1`  | float16/bf16 | nz       | [161, 128, 1, 64]       |

> Default values: `dtype = float16`, `tokenNum = 32`, `headNum = 128`
