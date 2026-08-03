# FusedAddTopkDivOperation C++ Demo

## Introduction

This directory contains the C++ call sample of FusedAddTopkDivOperation.

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

## Remarks

The data generated in the example does not represent the actual outputs. For details about data generation, see the Python use case directory under the root directory:
tests/apitest/opstest/python/operations/fused_add_topk_div/

## Description

  The demos provided by this operator can run only on Atlas A2/A3 products. They run in the following scenarios:

- fused_add_topk_div_demo

    **Parameter settings**

    | Name           | Value                |
    | :------------------ | :------------------- |
    | groupNum            | 8                    |
    | groupTopk           | 4                    |
    | n                   | 2                    |
    | k                   | 8                    |
    | activationType      | `ACTIVATION_SIGMOID` |
    | isNorm              | `true`               |
    | scale               | 2.5                  |
    | enableExpertMapping | `false`              |

    **Data specifications**

    | Tensor| Data Type| Data Format| Dimension  | cpu/npu |
    | ---------- | -------- | -------- | ---------- | ------- |
    | `x`        | float16  | nd       | [512, 256] | npu     |
    | `add_num`  | float16  | nd       | [256]      | npu     |
    | `y`        | float    | nd       | [512, 8]   | npu     |
    | `indices`  | int32    | nd       | [512, 8]   | npu     |
