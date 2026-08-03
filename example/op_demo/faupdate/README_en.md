# FaupdateOperation C++ Demo

## Introduction

This directory contains the C++ call sample of FaupdateOperation.

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

    - The provided build script is used only to build and run `faupdate_demo.cpp`. To build other demos, replace `faupdate_demo` with the corresponding .cpp file name.

## Remarks

The data generated in the example does not represent the actual outputs. For details about data generation, see the Python use case directory under the root directory:
`tests/apitest/opstest/python/operations/faupdate/`

## Supported Products

Atlas A2/A3 products only.

## Scenarios

Description

- faupdate_demo.cpp

  This demo can run only on Atlas A2/A3 products.

    **Parameter Setting**

    | Name      | Value         |
    | :------------- | :------------ |
    | `faUpdateType` | DECODE_UPDATE |
    | sp             | 8             |

    **Data specifications**

    | Tensor| Data Type | Data Format | Dimension       | cpu/npu |
    | :--------- | :------- | :------- | :-------------- | ------- |
    | `lse`      | float    | nd       | [8, 16384]      | npu     |
    | `localout` | float    | nd       | [8, 16384, 128] | npu     |
    | `output`   | float    | nd       | [16384, 128]    | npu     |
