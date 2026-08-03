# LinearOperation C++ Demo

## Introduction

This directory contains the C++ call sample of LinearOperation.

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

  - The provided build script is used only to build and run `linear_demo.cpp`. To build other demos, replace `linear_demo` with the corresponding .cpp file name.

## Remarks

The data generated in the example does not represent the actual outputs. For details about data generation, see the Python use case directory under the root directory:
`tests/apitest/opstest/python/operations/linear/`

## Supported Products

The implementation of this OP varies on the Atlas A2 training/inference products, Atlas A3 training/inference products, and other Atlas inference products.

### Description

The provided demos correspond to the following scenarios. You need to modify the build script accordingly.

1. Basic scenario:
    - linear_demo.cpp

        The default build script can build and run the demo. Unless otherwise specified, the demo can run on the Atlas A2 training/inference products, Atlas A3 training/inference products, and Atlas inference products.

        **Parameter settings**

        |  Name   | Value              |
        | :---------- | :----------------- |
        | transposeA  | false              |
        | transposeB  | false              |
        | hasBias     | true               |
        | outDataType | `ACL_DT_UNDEFINED` |
        | enAccum     | false              |
        | matmulType  | `MATMUL_UNDEFINED` |

        **Data specifications**

        | Tensor| Data Type| Data Format | Dimension | cpu/npu |
        | :--------- | :------- | :------- | :------- | :------ |
        | `x`        | float16  | nd       | [2, 3]   | npu     |
        | `weight`   | float16  | nd       | [3, 2]   | npu     |
        | `bias`     | float16  | nd       | [1, 2]   | npu     |
        | `output`   | float16  | nd       | [2, 2]   | npu     |

    - linear_ds_demo.cpp

        This demo can run only on the Atlas A2 training/inference products and Atlas A3 training/inference products.

        **Parameter settings**

        | Name    | Value              |
        | :---------- | :----------------- |
        | transposeA  | false              |
        | transposeB  | true               |
        | hasBias     | false              |
        | outDataType | `ACL_DT_UNDEFINED` |
        | enAccum     | false              |
        | matmulType  | `MATMUL_UNDEFINED` |

        **Data specifications**

        | Tensor| Data Type| Data Format | Dimension    | cpu/npu |
        | ---------- | -------- | -------- | ----------- | :------ |
        | `x`        | float    | nd       | [512, 7168] | npu     |
        | `weight`   | float    | nd       | [256, 7168] | npu     |
        | `output`   | float    | nd       | [512, 256]  | npu     |

    - linear_qwen_demo.cpp

        This demo can run only on the Atlas A2 training/inference products and Atlas A3 training/inference products.

        **Parameter settings**

        |  Name   | Value              |
        | :---------- | :----------------- |
        | transposeA  | false              |
        | transposeB  | false              |
        | hasBias     | false              |
        | outDataType | `ACL_DT_UNDEFINED` |
        | enAccum     | false              |
        | matmulType  | `MATMUL_UNDEFINED` |

        **Data specifications**

        | Tensor| Data Type | Data Format | Dimension     | cpu/npu |
        | ---------- | -------- | -------- | ------------ | :------ |
        | `x`        | bf16     | nd       | [1, 1728]    | npu     |
        | `weight`   | bf16     | nz       | [1728, 5120] | npu     |
        | `output`   | bf16     | nd       | [1, 5120]    | npu     |

    - linear_qwen_bias_demo.cpp

        This demo can run only on the Atlas A2 training/inference products and Atlas A3 training/inference products.

        **Parameter settings**

        | Name    | Value              |
        | :---------- | :----------------- |
        | transposeA  | false              |
        | transposeB  | false              |
        | hasBias     | true               |
        | outDataType | `ACL_DT_UNDEFINED` |
        | enAccum     | false              |
        | matmulType  | `MATMUL_UNDEFINED` |

        **Data specifications**

        | Tensor| Data Type| Data Format| Dimension      | cpu/npu |
        | ---------- | -------- | -------- | ------------ |  ------ |
        | `x`        | bf16     | nd       | [1024, 5120] | npu     |
        | `weight`   | bf16     | nz       | [5120, 896]  | npu     |
        | `bias`     | bf16     | nd       | [1, 896]     | npu     |
        | `output`   | bf16     | nd       | [1024, 896]  | npu     |

2. Einstein summation scenario:

    linear_einsum_demo.cpp

    - Change the build script as follows:
    `g++ -D_GLIBCXX_USE_CXX11_ABI=$cxx_abi -I "${ATB_HOME_PATH}/include" -I "${ASCEND_HOME_PATH}/include" -L "${ATB_HOME_PATH}/lib" -L "${ASCEND_HOME_PATH}/lib64" linear_einsum_demo.cpp demo_util.h -l atb -l ascendcl -o linear_einsum_demo`
    - Runtime calling:
    `./linear_einsum_demo`
    - This demo can run only on the Atlas A2 training/inference products and Atlas A3 training/inference products.
    - linear_einsum_demo.cpp

        **Parameter settings**

        | Name    | Value              |
        | :---------- | :----------------- |
        | transposeA  | false              |
        | transposeB  | false              |
        | hasBias     | false              |
        | outDataType | `ACL_DT_UNDEFINED` |
        | enAccum     | false              |
        | matmulType  | `MATMUL_EIN_SUM`   |

        **Data specifications**

        | Tensor| Data Type | Data Format |  Dimension       | cpu/npu |
        | ---------- | -------- | -------- | --------------- | ------- |
        | `x`        | float16  | nd       | [32, 128, 512]  | npu     |
        | `weight`   | float16  | nd       | [128, 512, 128] | npu     |
        | `output`   | float16  | nd       | [32, 128, 512]  | npu     |

3. Quantization scenario

    - linear_dequant_demo.cpp

        This demo can run only on the Atlas A2 training/inference products and Atlas A3 training/inference products.

        **Parameter settings**

        | Name    | Value              |
        | :---------- | :----------------- |
        | transposeA  | false              |
        | transposeB  | false              |
        | hasBias     | true               |
        | outDataType | `ACL_BF16`         |
        | enAccum     | false              |
        | matmulType  | `MATMUL_UNDEFINED` |

        **Data specifications**

        | Tensor| Data Type | Data Format | Dimension| cpu/npu |
        | ---------- | -------- | -------- | -------- |-------- |
        | `x`        | int8     | nd       | [2, 3]   | npu     |
        | `weight`   | int8     | nd       | [3, 2]   | npu     |
        | `bias`     | int32    | nd       | [1, 2]   | npu     |
        | `deqScale` | float    | nd       | [1, 2]   | npu     |
        | `output`   | bf16     | nd       | [2, 2]   | npu     |


    - linear_dequant_ds_demo.cpp

        This demo can run on the Atlas A2 training/inference products, Atlas A3 training/inference products, and Atlas inference products.

        **Parameter settings**

        | Name    | Value              |
        | :---------- | :----------------- |
        | transposeA  | false              |
        | transposeB  | true               |
        | hasBias     | true               |
        | outDataType | `ACL_FLOAT16`      |
        | enAccum     | false              |
        | matmulType  | `MATMUL_UNDEFINED` |

        **Data specifications**

        | Tensor| Data Type | Data Format | Dimension     | cpu/npu |
        | ---------- | -------- | -------- | ------------- |-------- |
        | `x`        | int8     | nd       | [32, 16384]   | npu     |
        | `weight`   | int8     | nd       | [7168, 16384] | npu     |
        | `bias`     | int32    | nd       | [1, 7168]     | npu     |
        | `deqScale` | int64    | nd       | [1, 7168]     | npu     |
        | `output`   | float16  | nd       | [32, 7168]    | npu     |
