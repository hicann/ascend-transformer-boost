# 加速库RopeOperation C++ Demo
## 介绍
该目录下为加速库RopeOperation C++调用示例。

## 使用说明
- 首先source 对应的CANN和nnal包
    1. source [cann安装路径]/set_env.sh
        默认：source /usr/local/Ascend/ascend-toolkit/set_env.sh
    2. source [nnal安装路径]/set_env.sh
        默认：source /usr/local/Ascend/nnal/atb/set_env.sh
        1. 如果使用加速库源码编译，source [加速库源码路径]/output/atb/set_env.sh
        e.g. source ./ascend-transformer-boost/output/atb/set_env.sh

- 编译、运行demo
    - bash build.sh

## 额外说明
示例中生成的数据不代表实际场景，如需数据生成参考请查看python用例目录：
tests/apitest/opstest/python/operations/rope/

### 场景说明
提供demo分别对应，编译运行时需要对应更改build脚本：
- **rope_demo.cpp**

    【注】：默认编译脚本可编译运行

    **参数设置**
    |    Param    | value |
    | :---------: | :---: |
    |  cosFormat  |   0   |
    | rotaryCoeff |   4   |

    **输入**
    | TensorName | DataType | DataFormat |  Shape  |
    | :--------: | :------: | :--------: | :-----: |
    |   query    | float16  |     nd     | [4, 16] |
    |    key     | float16  |     nd     | [4, 16] |
    |    cos     | float16  |     nd     | [4,  8] |
    |    sin     | float16  |     nd     | [4,  8] |
    |   seqlen   |  uint32  |     nd     |   [1]   |
    
    **输出**
    | TensorName | DataType | DataFormat |  Shape  |
    | :--------: | :------: | :--------: | :-----: |
    |   ropeQ    | float16  |     nd     | [4, 16] |
    |   ropeK    | float16  |     nd     | [4, 16] |

    ---

- **rope_qwen_demo_0.cpp**

    【注】：编译脚本内替换 rope_demo.cpp 为 rope_qwen_demo_0.cpp 可编译运行

    **参数设置**
    |    Param    | value |
    | :---------: | :---: |
    |  cosFormat  |   0   |
    | rotaryCoeff |   2   |

    **输入**
    | TensorName | DataType | DataFormat |    Shape    |
    | :--------: | :------: | :--------: | :---------: |
    |   query    |   bf16   |     nd     | [1024, 640] |
    |    key     |   bf16   |     nd     | [1024, 128] |
    |    cos     |   bf16   |     nd     | [1024, 128] |
    |    sin     |   bf16   |     nd     | [1024, 128] |
    |   seqlen   |  uint32  |     nd     |     [1]     |

    **输出**
    | TensorName | DataType | DataFormat |    Shape    |
    | :--------: | :------: | :--------: | :---------: |
    |   ropeQ    |   bf16   |     nd     | [1024, 640] |
    |   ropeK    |   bf16   |     nd     | [1024, 128] |

    ---

- **rope_qwen_demo_1.cpp**

    【注】：编译脚本内替换 rope_demo.cpp 为 rope_qwen_demo_1.cpp 可编译运行

    **参数设置**
    |    Param    | value |
    | :---------: | :---: |
    |  cosFormat  |   0   |
    | rotaryCoeff |   2   |

    **输入**
    | TensorName | DataType | DataFormat |  Shape   |
    | :--------: | :------: | :--------: | :------: |
    |   query    |   bf16   |     nd     | [1, 640] |
    |    key     |   bf16   |     nd     | [1, 128] |
    |    cos     |   bf16   |     nd     | [1, 128] |
    |    sin     |   bf16   |     nd     | [1, 128] |
    |   seqlen   |  uint32  |     nd     |   [1]    |

    **输出**
    | TensorName | DataType | DataFormat |  Shape   |
    | :--------: | :------: | :--------: | :------: |
    |   ropeQ    |   bf16   |     nd     | [1, 640] |
    |   ropeK    |   bf16   |     nd     | [1, 128] |

    ---

- **rope_qwen_demo_2.cpp**

    【注】：编译脚本内替换 rope_demo.cpp 为 rope_qwen_demo_2.cpp 可编译运行

    **参数设置**
    |    Param    | value |
    | :---------: | :---: |
    |  cosFormat  |   0   |
    | rotaryCoeff |   2   |

    **输入**
    | TensorName | DataType | DataFormat |  Shape   |
    | :--------: | :------: | :--------: | :------: |
    |   query    |   bf16   |     nd     | [5, 640] |
    |    key     |   bf16   |     nd     | [5, 128] |
    |    cos     |   bf16   |     nd     | [5, 128] |
    |    sin     |   bf16   |     nd     | [5, 128] |
    |   seqlen   |  uint32  |     nd     |   [1]    |
    
    **输出**
    | TensorName | DataType | DataFormat |  Shape   |
    | :--------: | :------: | :--------: | :------: |
    |   ropeQ    |   bf16   |     nd     | [5, 640] |
    |   ropeK    |   bf16   |     nd     | [5, 128] |