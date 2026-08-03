# LinearParallelOperation C++ Demo

## Introduction

This directory contains the C++ call sample of LinearParallelOperation.

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

    - The provided build script is used only to build and run `linear_parallel_demo.cpp`. To build other demos, replace `linear_parallel_demo` with the corresponding .cpp file name.

## Remarks

The data generated in the example does not represent the actual outputs. For details about data generation, see the Python use case directory under the root directory:
tests/apitest/opstest/python/operations/linear/

## Supported Products

This OP supports only Atlas A2 training/inference products and Atlas A3 training/inference products.

### Scenarios

The provided demos correspond to the following scenarios. You need to modify the build script accordingly.

1. Basic scenario:
    linear_parallel_demo.cpp
    - The default build script can build and run the demo.
2. Quantization scenario:
    linear_parallel_demo.cpp
    - Change the build script as follows:
    `g++ -D_GLIBCXX_USE_CXX11_ABI=$cxx_abi -I "${ATB_HOME_PATH}/include" -I "${ASCEND_HOME_PATH}/include" -L "${ATB_HOME_PATH}/lib" -L "${ASCEND_HOME_PATH}/lib64" linear_parallel_demo.cpp demo_util.h -l atb -l ascendcl -o linear_parallel_demo`
    - Runtime calling:
    `./linear_parallel_demo`
