# RMSNormOperation C++ Demo

## Introduction

This directory contains the C++ call sample of RMSNormOperation.

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

    - The provided build script is used only to build and run `rms_norm_demo.cpp`. To build other demos, replace `rms_norm_demo` with the corresponding .cpp file name.

## Remarks

The data generated in the example does not represent the actual outputs. For details about data generation, see the Python use case directory under the root directory:
tests/apitest/opstest/python/operations/rms_norm/

### Description

The provided demos correspond to the following scenarios. You need to modify the build script accordingly.

- **rms_norm_demo.cpp**

    Note: The default build script can build and run the demo.

    **Parameter settings**

    |        Parameter       |                        Value                        |
    | :-----------------: | :--------------------------------------------------: |
    |      layerType      | atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM |
    | normParam.quantType |         atb::infer::QuantType::QUANT_UNQUANT         |
    |       epsilon       |                         1e-5                         |

    **Input**

    | Tensor| Data Type| Data Format|      Shape      |
    | :--------: | :------: | :--------: | :-------------: |
    |     x      | float16  |     nd     | [4, 1024, 5120] |
    |   gamma    | float16  |     nd     |     [5120]      |

    **Output**

    | Tensor| Data Type| Data Format|      Shape      |
    | :--------: | :------: | :--------: | :-------------: |
    |   output   | float16  |     nd     | [4, 1024, 5120] |

    ---

- **rms_norm_qwen_demo_0.cpp**

    Note: Replace `rms_norm_demo.cpp` with `rms_norm_qwen_demo_0.cpp` in the build script to build and run the demo.
           This sample applies only to Atlas A2/A3 training products, Atlas 800I A2 inference products, and Atlas A3 inference products.
    **Parameter settings**

    |        Parameter       |                        Value                        |
    | :-----------------: | :--------------------------------------------------: |
    |      layerType      | atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM |
    | normParam.quantType |         atb::infer::QuantType::QUANT_UNQUANT         |
    |       epsilon       |                         1e-6                         |

    **Input**

    | Tensor| Data Type| Data Format|    Shape     |
    | :--------: | :------: | :--------: | :----------: |
    |     x      |   bf16   |     nd     | [1024, 5120] |
    |   gamma    |   bf16   |     nd     |    [5120]    |

    **Output**

    | Tensor| Data Type| Data Format|    Shape     |
    | :--------: | :------: | :--------: | :----------: |
    |   output   |   bf16   |     nd     | [1024, 5120] |

    ---

- **rms_norm_qwen_demo_1.cpp**

    Note: Replace `rms_norm_demo.cpp` with `rms_norm_qwen_demo_1.cpp` in the build script to build and run the demo.
           This sample applies only to Atlas A2/A3 training products, Atlas 800I A2 inference products, and Atlas A3 inference products.
    **Parameter settings**

    |        Parameter       |                        Value                        |
    | :-----------------: | :--------------------------------------------------: |
    |      layerType      | atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM |
    | normParam.quantType |         atb::infer::QuantType::QUANT_UNQUANT         |
    |       epsilon       |                         1e-6                         |

    **Input**

    | Tensor| Data Type| Data Format|   Shape   |
    | :--------: | :------: | :--------: | :-------: |
    |     x      |   bf16   |     nd     | [1, 5120] |
    |   gamma    |   bf16   |     nd     |  [5120]   |

    **Output**

    | Tensor| Data Type| Data Format|   Shape   |
    | :--------: | :------: | :--------: | :-------: |
    |   output   |   bf16   |     nd     | [1, 5120] |

    ---

- **rms_norm_qwen_demo_2.cpp**

    Note: Replace `rms_norm_demo.cpp` with `rms_norm_qwen_demo_2.cpp` in the build script to build and run the demo.
           This sample applies only to Atlas A2/A3 training products, Atlas 800I A2 inference products, and Atlas A3 inference products.
    **Parameter settings**

    |        Parameter       |                        Value                        |
    | :-----------------: | :--------------------------------------------------: |
    |      layerType      | atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM |
    | normParam.quantType |         atb::infer::QuantType::QUANT_UNQUANT         |
    |       epsilon       |                         1e-6                         |

    **Input**

    | Tensor| Data Type| Data Format|   Shape   |
    | :--------: | :------: | :--------: | :-------: |
    |     x      |   bf16   |     nd     | [5, 5120] |
    |   gamma    |   bf16   |     nd     |  [5120]   |

    **Output**

    | Tensor| Data Type| Data Format|   Shape   |
    | :--------: | :------: | :--------: | :-------: |
    |   output   |   bf16   |     nd     | [5, 5120] |

    ---

- **rms_norm_deepseek_demo_0.cpp**

    Note: Replace `rms_norm_demo.cpp` with `rms_norm_deepseek_demo_0.cpp` in the build script to build and run the demo.
           This sample applies only to Atlas A2/A3 training products, Atlas 800I A2 inference products, and Atlas A3 inference products.
    **Parameter settings**

    |        Parameter       |                        Value                        |
    | :-----------------: | :--------------------------------------------------: |
    |      layerType      | atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM |
    | normParam.quantType |         atb::infer::QuantType::QUANT_UNQUANT         |
    |       epsilon       |                         1e-6                         |

    **Input**

    | Tensor| Data Type| Data Format|    Shape    |
    | :--------: | :------: | :--------: | :---------: |
    |     x      | float16  |     nd     | [512, 7168] |
    |   gamma    | float16  |     nd     |   [7168]    |

    **Output**

    | Tensor| Data Type| Data Format|    Shape    |
    | :--------: | :------: | :--------: | :---------: |
    |   output   | float16  |     nd     | [512, 7168] |

    ---

- **rms_norm_deepseek_demo_1.cpp**

    Note: Replace `rms_norm_demo.cpp` with `rms_norm_deepseek_demo_1.cpp` in the build script to build and run the demo.
           This sample applies only to Atlas A2/A3 training products, Atlas 800I A2 inference products, and Atlas A3 inference products.
    **Parameter settings**

    |        Parameter       |                        Value                        |
    | :-----------------: | :--------------------------------------------------: |
    |      layerType      | atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM |
    | normParam.quantType |         atb::infer::QuantType::QUANT_UNQUANT         |
    |       epsilon       |                         1e-6                         |

    **Input**

    | Tensor| Data Type| Data Format|   Shape    |
    | :--------: | :------: | :--------: | :--------: |
    |     x      | float16  |     nd     | [32, 7168] |
    |   gamma    | float16  |     nd     |   [7168]   |

    **Output**

    | Tensor| Data Type| Data Format|   Shape    |
    | :--------: | :------: | :--------: | :--------: |
    |   output   | float16  |     nd     | [32, 7168] |
