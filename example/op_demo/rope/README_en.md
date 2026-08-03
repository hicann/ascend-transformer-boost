# RopeOperation C++ Demo

## Introduction

This directory contains the C++ call sample of RopeOperation.

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

    - The provided build script is used only to build and run `rope_demo.cpp`. To build other demos, replace `rope_demo` with the corresponding .cpp file name.

## Remarks

The data generated in the example does not represent the actual outputs. For details about data generation, see the Python use case directory under the root directory:
tests/apitest/opstest/python/operations/rope/

### Description

The provided demos correspond to the following scenarios. You need to modify the build script accordingly.

- **rope_demo.cpp**

    Note: The default build script can build and run the demo.

    **Parameter settings**

    |    Parameter   | Value|
    | :---------: | :---: |
    |  cosFormat  |   0   |
    | rotaryCoeff |   4   |

    **Input**

    | Tensor| Data Type| Data Format|  Shape  |
    | :--------: | :------: | :--------: | :-----: |
    |   query    | float16  |     nd     | [4, 16] |
    |    key     | float16  |     nd     | [4, 16] |
    |    cos     | float16  |     nd     | [4,  8] |
    |    sin     | float16  |     nd     | [4,  8] |
    |   seqlen   |  uint32  |     nd     |   [1]   |

    **Output**

    | Tensor| Data Type| Data Format|  Shape  |
    | :--------: | :------: | :--------: | :-----: |
    |   ropeQ    | float16  |     nd     | [4, 16] |
    |   ropeK    | float16  |     nd     | [4, 16] |

    ---

- **rope_qwen_demo_0.cpp**

    Note: Replace `rope_demo.cpp` with `rope_qwen_demo_0.cpp` in the build script to build and run the demo.
           This sample applies only to Atlas A2/A3 training products, Atlas 800I A2 inference products, and Atlas A3 inference products.
    **Parameter settings**

    |    Parameter   | Value|
    | :---------: | :---: |
    |  cosFormat  |   0   |
    | rotaryCoeff |   2   |

    **Input**

    | Tensor| Data Type| Data Format|    Shape    |
    | :--------: | :------: | :--------: | :---------: |
    |   query    |   bf16   |     nd     | [1024, 640] |
    |    key     |   bf16   |     nd     | [1024, 128] |
    |    cos     |   bf16   |     nd     | [1024, 128] |
    |    sin     |   bf16   |     nd     | [1024, 128] |
    |   seqlen   |  uint32  |     nd     |     [1]     |

    **Output**

    | Tensor| Data Type| Data Format|    Shape    |
    | :--------: | :------: | :--------: | :---------: |
    |   ropeQ    |   bf16   |     nd     | [1024, 640] |
    |   ropeK    |   bf16   |     nd     | [1024, 128] |

    ---

- **rope_qwen_demo_1.cpp**

    Note: Replace `rope_demo.cpp` with `rope_qwen_demo_1.cpp` in the build script to build and run the demo.
           This sample applies only to Atlas A2/A3 training products, Atlas 800I A2 inference products, and Atlas A3 inference products.
    **Parameter settings**

    |    Parameter   | Value|
    | :---------: | :---: |
    |  cosFormat  |   0   |
    | rotaryCoeff |   2   |

    **Input**

    | Tensor| Data Type| Data Format|  Shape   |
    | :--------: | :------: | :--------: | :------: |
    |   query    |   bf16   |     nd     | [1, 640] |
    |    key     |   bf16   |     nd     | [1, 128] |
    |    cos     |   bf16   |     nd     | [1, 128] |
    |    sin     |   bf16   |     nd     | [1, 128] |
    |   seqlen   |  uint32  |     nd     |   [1]    |

    **Output**

    | Tensor| Data Type| Data Format|  Shape   |
    | :--------: | :------: | :--------: | :------: |
    |   ropeQ    |   bf16   |     nd     | [1, 640] |
    |   ropeK    |   bf16   |     nd     | [1, 128] |

    ---

- **rope_qwen_demo_2.cpp**

    Note: Replace `rope_demo.cpp` with `rope_qwen_demo_2.cpp` in the build script to build and run the demo.
           This sample applies only to Atlas A2/A3 training products, Atlas 800I A2 inference products, and Atlas A3 inference products.
    **Parameter settings**

    |    Parameter   | Value|
    | :---------: | :---: |
    |  cosFormat  |   0   |
    | rotaryCoeff |   2   |

    **Input**

    | Tensor| Data Type| Data Format|  Shape   |
    | :--------: | :------: | :--------: | :------: |
    |   query    |   bf16   |     nd     | [5, 640] |
    |    key     |   bf16   |     nd     | [5, 128] |
    |    cos     |   bf16   |     nd     | [5, 128] |
    |    sin     |   bf16   |     nd     | [5, 128] |
    |   seqlen   |  uint32  |     nd     |   [1]    |

    **Output**

    | Tensor| Data Type| Data Format|  Shape   |
    | :--------: | :------: | :--------: | :------: |
    |   ropeQ    |   bf16   |     nd     | [5, 640] |
    |   ropeK    |   bf16   |     nd     | [5, 128] |
