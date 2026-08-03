# Multi-Stream Demo

## Introduction

This directory contains demos for the multi-stream functionality of the Acceleration Library. `multiStream_singleGraph_demo.cpp` is a demo for single-graph multi-stream parallelism, and `multiStream_multiGraph_demo.cpp` is a demo for inter-graph synchronization.

### Multi-Stream Parallelism in a Graph

multiStream_singleGraph_demo.cpp

### Inter-graph Synchronization

multiStream_multiGraph_demo.cpp

## Instruction

- Source the installation paths of the CANN and NNAL packages.
    - source [CANN installation path] (default: /usr/local/Ascend/ascend-toolkit)/set_env.sh
    - source [nnal installation path] (default: /usr/local/Ascend/nnal/atb)/set_env.sh
    - If building from the acceleration library source code, run `source [Source code path]/output/atb/set_env.sh`.
- Modify add_executable in the CMakeLists.txt file in the current directory.
    - To run the single-graph multi-stream parallelism example, modify it as follows:

        ```sh
        add_executable(multiStreamDemo multiStream_singleGraph_demo.cpp)
        ```

    - To run the inter-graph synchronization example, modify it as follows:

        ```sh
        add_executable(multiStreamDemo multiStream_multiGraph_demo.cpp)
        ```

- Generate the build system.
    - Use cxx_abi=0.

        ```sh
        mkdir build && cd build    # Create and go to the build directory.
        cmake .. -DUSE_CXX11_ABI=OFF                   # Generate the build system.
        ```

    - Use cxx_abi=1.

        ```sh
        mkdir build && cd build    # Create and enter the build directory.
        cmake .. -DUSE_CXX11_ABI=ON                   # Generate the build system.
        ```

- Build and run.

    ```sh
    cmake --build .            # Build the project.
    ./multiStreamDemo         # Run the program.
    ```

- View profiling.

    ```sh
    msprof --application="multiStreamDemo"     # Generate a profiling file.
    ```
