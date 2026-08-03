# PagedCacheLoadOperation C++ Demo

## Introduction

This directory contains the C++ call sample of PagedCacheLoadOperation. The following samples apply only to Atlas A2/A3 training products, Atlas 800I A2 inference products, and Atlas A3 inference products.

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

    - The provided build script is used only to build and run `paged_cache_load_demo.cpp`. To build other demos, replace `paged_cache_load_demo` with the corresponding .cpp file name.

## Remarks

The data generated in the example does not represent the actual outputs. For details about data generation, see the Python use case directory under the root directory:
tests/apitest/opstest/python/operations/paged_cache_load/

### Description

The provided demos correspond to the following scenarios. You need to modify the build script accordingly.

- **paged_cache_load_demo.cpp**

    Note: The default build script can build and run the demo.

    **Parameter settings**

    |        Parameter       |                        Value                       |
    | :-----------------: | :-------------------------------------------------: |
    |     kvCacheCfg      | atb::infer::PagedCacheLoadParam::K_CACHE_V_CACHE_NZ |
    |    hasSeqStarts     |                        false                        |
    | isSeqLensCumsumMode |                        false                        |

    **Input**

    | Tensor | Data Type| Data Format|      Shape      |
    | :---------: | :------: | :--------: | :-------------: |
    |  keyCache   |   int8   | fractal_nz | [4, 4, 128, 32] |
    | valueCache  |   int8   | fractal_nz | [4, 4, 128, 32] |
    | blocktable  |  int32   |     nd     |     [3, 1]      |
    | contextlens |  int32   |     nd     |       [3]       |
    |     key     |   int8   |     nd     |   [384, 128]    |
    |    Value   |   int8   |     nd     |   [384, 128]    |

    **Output**

    | Tensor| Data Type| Data Format|   Shape    |
    | :--------: | :------: | :--------: | :--------: |
    |    key     |   int8   |     nd     | [384, 128] |
    |   Value   |   int8   |     nd     | [384, 128] |
