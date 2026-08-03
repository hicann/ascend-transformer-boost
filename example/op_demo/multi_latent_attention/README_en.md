# MultiLatentAttentionOperation C++ Demo

## Introduction

This directory contains the C++ call sample of MultiLatentAttentionOperation.

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

    - The provided build script is used only to build and run `mlapa_demo.cpp`. To build other demos, replace `mlapa_demo` with the corresponding .cpp file name.

## Remarks

The data generated in the example does not represent the actual outputs. For details about data generation, see the Python use case directory under the root directory:
tests/apitest/opstest/python/operations/multi_latent_attention/

## Description

  The demos provided by this operator can run only on Atlas A2/A3 products. They run in the following scenarios:

- mlapa_demo.cpp

    **Parameter settings**

    | Name  | Value                 |
    | :-------- | :-------------------- |
    | headNum   | 128                   |
    | qkScale   | 1/sqrt(576)          |
    | kvHeadNum | 1                     |
    | maskType  | `UNDEFINED`           |
    | calcType  | `CALC_TYPE_UNDEFINED` |
    | cacheMode | `INT8_NZCACHE`        |

    > Note: The value of `qkScale` is the headSize before MLA performs RoPE projection, that is, `512 (original) + 64 (projection) = 576`.

    **Data specifications**

    | Tensor   | Data Type | Data Format | Dimension         | cpu/npu |
    | ------------- | -------- | -------- | ----------------- | ------- |
    | `qNope`       | int8     | nd       | [4, 128, 512]     | npu     |
    | `qRope`       | float16  | nd       | [4, 128, 64]      | npu     |
    | `ctKV`        | int8     | nz       | [48, 16, 128, 32] | npu     |
    | `kRope`       | float16  | nz       | [48, 4, 128, 16]  | npu     |
    | `blockTables` | int32    | nd       | [4, 12]           | npu     |
    | `contextLens` | int32    | nd       | [4]               | cpu     |
    | `qkDescale`   | float    | nd       | [128]             | npu     |
    | `pvDescale`   | float    | nd       | [128]             | npu     |
    | `attenOut`    | float16  | nd       | [4, 128, 512]     | npu     |

- mlapa_ds_demo.cpp

    **Parameter settings**

    | Name  | Value                 |
    | :-------- | :-------------------- |
    | headNum   | 128                   |
    | qkScale   | 0.1352667747812271    |
    | kvHeadNum | 1                     |
    | maskType  | `UNDEFINED`           |
    | calcType  | `CALC_TYPE_UNDEFINED` |
    | cacheMode | `KROPE_CTKV`          |

    **Data specifications**

    | Tensor    | Data Type | Data Format | Dimension          | cpu/npu |
    | ------------- | -------- | -------- | ------------------ |-------- |
    | `qNope`       | float16  | nd       | [32, 128, 512]     | npu     |
    | `qRope`       | float16  | nd       | [7168, 128, 64]    | npu     |
    | `ctKV`        | float16  | nd       | [160, 128, 1, 512] | npu     |
    | `kRope`       | float16  | nd       | [160, 128, 1, 64]  | npu     |
    | `blockTables` | int32    | nd       | [32, 5]            | npu     |
    | `contextLens` | int32    | nd       | [32]               | cpu     |
    | `attenOut`    | float16  | nd       | [32, 128, 512]     | npu     |
