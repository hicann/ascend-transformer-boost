# ReshapeAndCacheOperation C++ Demo

## Introduction

This directory contains the C++ call sample of ReshapeAndCacheOperation.

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

    - The provided build script is used only to build and run `reshape_and_cache_demo.cpp`. To build other demos, replace `reshape_and_cache_demo` with the corresponding .cpp file name.

## Remarks

The data generated in the example does not represent the actual outputs. For details about data generation, see the Python use case directory under the root directory:
tests/apitest/opstest/python/operations/reshape_and_cache/

## Supported Products

The implementation of this OP differs between the Atlas A2/A3 products and the Atlas inference products.

### Description

The provided demos correspond to the basic scenarios of different products. You need to modify the build script accordingly.

1. Atlas A2/A3 products:

   **Parameter settings**

    | Name    | Value                   |
    | :----------- | :---------------------- |
    | compressType | COMPRESS_TYPE_UNDEFINED |
    | kvCacheCfg   | K_CACHE_V_CACHE         |

    The following demos can run only on Atlas A2/A3 products.

    - reshape_and_cache_demo.cpp

        | Tensor     | Data Type| Data Format| Dimension           |
        | :-------------- | :------- | :------- | :------------------ |
        | `key`           | float16  | nd       | [2, 32, 128]        |
        | `value`         | float16  | nd       | [2, 32, 128]        |
        | `keyCache`      | float16  | nd       | [512, 128, 32, 128] |
        | `valueCache`    | float16  | nd       | [512, 128, 32, 128] |
        | `slotMapping`   | int32    | nd       | [2]                 |
        | `keyCacheOut`   | float16  | nd       | [512, 128, 32, 128] |
        | `valueCacheOut` | float16  | nd       | [512, 128, 32, 128] |

    - reshape_and_cache_demo_ds1.cpp

        | Tensor     | Data Type| Data Format| Dimension        |
        | :-------------- | :------- | :------- | :--------------- |
        | `key`           | bf16     | nd       | [5, 1, 128]      |
        | `value`         | bf16     | nd       | [5, 1, 128]      |
        | `keyCache`      | bf16     | nd       | [9, 128, 1, 128] |
        | `valueCache`    | bf16     | nd       | [9, 128, 1, 128] |
        | `slotMapping`   | int32    | nd       | [5]              |
        | `keyCacheOut`   | bf16     | nd       | [9, 128, 1, 128] |
        | `valueCacheOut` | bf16     | nd       | [9, 128, 1, 128] |

    - reshape_and_cache_demo_ds2.cpp

        | Tensor     | Data Type| Data Format| Dimension        |
        | :-------------- | :------- | :------- | :--------------- |
        | `key`           | bf16     | nd       | [1024, 1, 128]   |
        | `value`         | bf16     | nd       | [1024, 1, 128]   |
        | `keyCache`      | bf16     | nd       | [9, 128, 1, 128] |
        | `valueCache`    | bf16     | nd       | [9, 128, 1, 128] |
        | `slotMapping`   | int32    | nd       | [1024]           |
        | `keyCacheOut`   | bf16     | nd       | [9, 128, 1, 128] |
        | `valueCacheOut` | bf16     | nd       | [9, 128, 1, 128] |

    - reshape_and_cache_demo_ds3.cpp

        | Tensor     | Data Type| Data Format| Dimension        |
        | :-------------- | :------- | :------- | :--------------- |
        | `key`           | bf16     | nd       | [1, 1, 128]      |
        | `value`         | bf16     | nd       | [1, 1, 128]      |
        | `keyCache`      | bf16     | nd       | [9, 128, 1, 128] |
        | `valueCache`    | bf16     | nd       | [9, 128, 1, 128] |
        | `slotMapping`   | int32    | nd       | [1]              |
        | `keyCacheOut`   | bf16     | nd       | [9, 128, 1, 128] |
        | `valueCacheOut` | bf16     | nd       | [9, 128, 1, 128] |

2. Atlas inference products:
    reshape_and_cache_inference_demo.cpp
    - Compared with the A2/A3 demos, this sample features the following modifications:
        - The kvCache shape is changed to [block_num, head_size \times head_num / 16, block_size, 16].
        - The kvCache data format is changed to ACL_FORMAT_FRACTAL_NZ.

        **Parameter settings**

        | Name    | Value                   |
        | :----------- | :---------------------- |
        | compressType | COMPRESS_TYPE_UNDEFINED |
        | kvCacheCfg   | K_CACHE_V_CACHE         |

        | Tensor     | Data Type| Data Format| Dimension        |
        | :-------------- | :------- | :------- | :----------------- |
        | `key`           | bf16     | nd       | [3, 4, 128]        |
        | `value`         | bf16     | nd       | [3, 4, 128]        |
        | `keyCache`      | bf16     | nd       | [512, 32, 128, 16] |
        | `valueCache`    | bf16     | nd       | [512, 32, 128, 16] |
        | `slotMapping`   | int32    | nd       | [3]                |
        | `keyCacheOut`   | bf16     | nd       | [512, 32, 128, 16] |
        | `valueCacheOut` | bf16     | nd       | [512, 32, 128, 16] |

    - Modify the build script as follows:
    `g++ -D_GLIBCXX_USE_CXX11_ABI=$cxx_abi -I "${ATB_HOME_PATH}/include" -I "${ASCEND_HOME_PATH}/include" -L "${ATB_HOME_PATH}/lib" -L "${ASCEND_HOME_PATH}/lib64" reshape_and_cache_inference_demo.cpp demo_util.h -l atb -l ascendcl -o reshape_and_cache_inference_demo`
    - Runtime calling:
    `./reshape_and_cache_inference_demo`
    - This demo can run only on Atlas inference products.
