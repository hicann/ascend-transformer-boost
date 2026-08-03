# RingMLA C++ Demo

## Introduction

This directory contains the C++ call sample of RingMLA.

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
        ./ring_mla_demo 0
        ```

## Remarks

The data generated in the example does not represent the actual outputs. For details about data generation, see the Python use case directory under the root directory:
tests/apitest/opstest/python/operations/ring_mla/

## Supported Products

Atlas A2/A3 products only

### Scenarios

1. RingMLA:
    + Basic scenarios: For query and key, matrices with and without RoPE transposition are passed separately.
    + A fixed-shape 512x512 upper-triangular mask is passed.
    + The default build script can build and run the demo.
    + This demo can run only on Atlas A2/A3 products.

#### Demo Segment Parameters/Tensor Specification

RingMLA is different from other operators. The first computation does not use the previously generated prevOut and prevLse. However, from the second computation onwards, they must be included. The following describes the parameters in two segments:

1. Computation 1

**Parameter settings**

| Name  | Value                |
| :--------- | :------------------- |
| calcType   | CALC_TYPE_FISRT_RING |
| headNum    | 16                   |
| kvHeadNum  | 8                    |
| qkScale    | 1/sqrt(192)          |
| kernelType | `KERNELTYPE_DEFAULT` |
| maskType   | `MASK_TYPE_TRIU`     |

> Note: The `qkScale` value is set to `1/sqrt(headSize)` where `headSize` is the unified head size of the query and key before RoPE transposition, i.e., `128 (nope) + 64 (rope) = 192`.

**Data specifications**

| Tensor  | Data Type| Data Format| Dimension       | cpu/npu |
| ------------ | -------- | -------- | --------------- | ------- |
| `queryNope`  | bf16     | nd       | [1228, 16, 128] | npu     |
| `queryRope`  | bf16     | nd       | [1228, 16, 64]  | npu     |
| `keyNope`    | bf16     | nd       | [828, 8, 128]   | npu     |
| `keyRope`    | bf16     | nd       | [828, 8, 64]    | npu     |
| `value`      | bf16     | nd       | [828, 8, 128]   | npu     |
| `mask`       | bf16     | nd       | [512, 512]      | npu     |
| `seqLen`     | int32    | nd       | [2, 3]          | cpu     |
| **Output**   |
| `output`     | bf16     | nd       | [1228, 16, 128] | npu     |
| `softmaxLse` | float    | nd       | [16, 1228]      | npu     |

> The first dimension of `q` is the total sequence length, corresponding to `sum(seqlen[0])`. The first dimensions of `k` and `v` correspond to `sum(seqlen[1])`.

1. Computation 2
In the second round, the newly generated `output` and `softmaxLse` from the first round are used for computation.

**Parameter settings**

| Name    | Value                |
| :----------- | :------------------- |
| **calcType** | CALC_TYPE_DEFAULT    |
| headNum      | 16                   |
| kvHeadNum    | 8                    |
| qkScale      | 1/sqrt(192)          |
| kernelType   | `KERNELTYPE_DEFAULT` |
| maskType     | `MASK_TYPE_TRIU`     |

> You need to change `calcType` in param to `CALC_TYPE_DEFAULT`. Other parameters remain unchanged.

**Data specifications**

| Tensor   | Data Type| Data Format| Dimension       | cpu/npu |
| ------------- | -------- | -------- | --------------- | ------- |
| `queryNope`   | bf16     | nd       | [1228, 16, 128] | npu     |
| `queryRope`   | bf16     | nd       | [1228, 16, 64]  | npu     |
| `keyNope`     | bf16     | nd       | [828, 8, 128]   | npu     |
| `keyRope`     | bf16     | nd       | [828, 8, 64]    | npu     |
| `value`       | bf16     | nd       | [828, 8, 128]   | npu     |
| `mask`        | bf16     | nd       | [512, 512]      | npu     |
| `seqLen`      | bf16     | nd       | [2, 3]          | cpu     |
| **`prevOut`** | bf16     | nd       | [1228, 16, 128] | npu     |
| **`prevLse`** | float    | nd       | [16, 1228]      | npu     |
| **Output**    |
| `output`      | bf16     | nd       | [1228, 16, 128] | npu     |
| `softmaxLse`  | float    | nd       | [16, 1228]      | npu     |

> In the second round, the newly generated `output` and `softmaxLse` from the first round are used as `prevOut` and `prevLse`.
