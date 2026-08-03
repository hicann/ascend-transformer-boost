# PagedAttentionOperation C++ Demo

## Introduction

This directory contains the C++ call sample of PagedAttentionOperation.

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

    - The provided build script is used only to build and run `paged_attention_demo.cpp`. To build other demos, replace `paged_attention_demo` with the corresponding .cpp file name.

## Remarks

The data generated in the example does not represent the actual outputs. For details about data generation, see the Python use case directory under the root directory:
tests/apitest/opstest/python/operations/paged_attention/

## Supported Products

The implementation of this OP differs between the Atlas A2/A3 products and the Atlas inference products.

## Scenarios

When building and running the provided demo, modify the build script accordingly:

1. Scenario without parallel decoding and with mask:

    - paged_attention_demo.cpp

        The default build script can build and run this demo. This demo can run only on Atlas A2/A3 products.

        **Parameter settings**

        | Name       | Value                     |
        | :------------- | :------------------------ |
        | headNum        | 32                        |
        | qkScale        | 1 / sqrt(HEAD_SIZE)       |
        | kvHeadNum      | 32                        |
        | batchRunStatus | 0                         |
        | quantType      | `TYPE_QUANT_UNQUANT`      |
        | hasQuantOffset | false                     |
        | calcType       | `CALC_TYPE_UNDEFINED`     |
        | compressType   | `COMPRESS_TYPE_UNDEFINED` |
        | maskType       | `MASK_TYPE_NORM`          |
        | mlaVHeadSize   | 0                         |

        **Data specifications**

        | Tensor   | Data Type| Data Format  | Dimension           | cpu/npu |
        | ------------- | -------- | -------- | ------------------ |-------- |
        | `query`       | float16  | nd       | [2, 32, 128]       | npu     |
        | `keyCache`    | float16  | nd       | [16, 128, 32, 128] | npu     |
        | `valueCache`  | float16  | nd       | [16, 128, 32, 128] | npu     |
        | `blockTables` | int32    | nd       | [2, 8]             | npu     |
        | `contextLens` | int32    | nd       | [2]                | cpu     |
        | `mask`        | int32    | nd       | [2, 1, 1024]       | npu     |
        | `attnOut`     | float16  | nd       | [2, 32, 128]       | npu     |

   - paged_attention_qwen_demo.cpp

        This demo can run only on Atlas A2/A3 products.

        **Parameter settings**

        | Name       | Value                     |
        | :------------- | :------------------------ |
        | headNum        | 5                         |
        | qkScale        | 1 / sqrt(HEAD_SIZE)       |
        | kvHeadNum      | 1                         |
        | batchRunStatus | 0                         |
        | quantType      | `TYPE_QUANT_UNDEFINED`    |
        | hasQuantOffset | false                     |
        | calcType       | `CALC_TYPE_UNDEFINED`     |
        | compressType   | `COMPRESS_TYPE_UNDEFINED` |
        | maskType       | `UNDEFINED`               |
        | mlaVHeadSize   | 0                         |

        **Data specifications**

        | Tensor   | Data Type | Data Format|  Dimension        | cpu/npu |
        | ------------- | -------- | -------- | ---------------- |---------|
        | `query`       | bf16     | nd       | [1, 5, 128]      | npu     |
        | `qkScale`     | bf16     | nd       | [9, 128, 1, 128] | npu     |
        | `valueCache`  | bf16     | nd       | [9, 128, 1, 128] | npu     |
        | `blockTables` | int32    | nd       | [1, 8]           | npu     |
        | `contextLens` | int32    | nd       | [1]              | cpu     |
        | `attnOut`     | bf16     | nd       | [1, 5, 128]      | npu     |

2. Scenario without mask:
   - paged_attention_inference_demo.cpp
    This demo can run only on Atlas inference products.
    **Parameter settings**

        | Name       | Value                     |
        | :------------- | :------------------------ |
        | headNum        | 32                        |
        | qkScale        | 1 / sqrt(HEAD_SIZE)       |
        | kvHeadNum      | 32                        |
        | batchRunStatus | 0                         |
        | quantType      | `TYPE_QUANT_UNQUANT`      |
        | hasQuantOffset | false                     |
        | calcType       | `CALC_TYPE_UNDEFINED`     |
        | compressType   | `COMPRESS_TYPE_UNDEFINED` |
        | maskType       | `UNDEFINED`               |
        | mlaVHeadSize   | 0                         |

        **Data specifications**

        | Tensor   | Data Type | Data Format | Dimension           | cpu/npu |
        | ------------- | -------- | -------- | ------------------- |---------|
        | `query`       | bf16     | nd       | [2, 32, 128]        | npu     |
        | `qkScale`     | bf16     | nd       | [16, 1024, 128, 16] | npu     |
        | `valueCache`  | bf16     | nd       | [16, 1024, 128, 16] | npu     |
        | `blockTables` | int32    | nd       | [2, 8]              | npu     |
        | `contextLens` | int32    | nd       | [2]                 | cpu     |
        | `attnOut`     | bf16     | nd       | [2, 32, 128]        | npu     |
