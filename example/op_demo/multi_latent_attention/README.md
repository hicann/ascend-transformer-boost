# 加速库MultiLatentAttentionOperation C++ Demo
## 介绍
该目录下为加速库MultiLatentAttentionOperation C++调用示例。

## 使用说明
- 首先source 对应的CANN和nnal包
    1. source [cann安装路径]/set_env.sh
        默认：source /usr/local/Ascend/ascend-toolkit/set_env.sh
    2. source [nnal安装路径]/set_env.sh
        默认：source /usr/local/Ascend/ascend-toolkit/set_env.sh
        1. 如果使用加速库源码编译，source [加速库源码路径]/output/atb/set_env.sh
        e.g. source ./ascend-transformer-boost/atb/set_env.sh

- 运行demo
    - bash build.sh
    **注意**：
    - 使用cxx_abi=0（默认）时，设置`D_GLIBCXX_USE_CXX11_ABI`为0，i.e.
        ```sh
        g++ -D_GLIBCXX_USE_CXX11_ABI=0 -I ...
        ```
    - 使用cxx_abi=1时，更改`D_GLIBCXX_USE_CXX11_ABI`为1，i.e.
        ```sh
        g++ -D_GLIBCXX_USE_CXX11_ABI=1 -I ...
        ```

## 额外说明
示例中生成的数据不代表实际场景，如需数据生成参考请查看python用例目录：
tests/apitest/opstest/python/operations/multi_latent_attention/

## 场景说明

  所给Demo的场景说明如下：

- mlapa_demo.cpp
  
    **参数设置**：

    | 成员名称    | 取值               |
    | :------------ | :----------------------- |
    | headNum  | 128 |
    | qkScale  | 0.0416666679084301|
    | kvHeadNum     | 1|
    | maskType | `UNDEFINED`|
    | calcType     | `CALC_TYPE_UNDEFINED`|
    | cacheMode  | `INT8_NZCACHE`|

    **数据规格**：

    | tensor名字| 数据类型 | 数据格式 | 维度信息|
    | --- | --- | --- | --- |
    | `qNope` | int8| nd | [4, 128, 512]|
    |`qRope`  |float16| nd |  [4, 128, 64]|
    |`ctKV`  |  int8| nz  |[48, 16, 128, 32]  |
    | `kRope` | float16 | nz  | [48, 4, 128, 16] |
    | `blockTables` | int32 | nd  | [4, 12] |
    | `contextLens` | int32 | nd  | [4] |
    | `qkDescale` | float | nd  | [128] |
    | `pvDescale` | float | nd  | [128] |
    | `attenOut` | float16| nd | [4, 128, 512] |

- mlapa_ds_demo.cpp  

    **参数设置**：

    | 成员名称    | 取值               |
    | :------------ | :----------------------- |
    | headNum  | 128 |
    | qkScale  | 0.1352667747812271|
    | kvHeadNum     | 1|
    | maskType | `UNDEFINED`|
    | calcType     | `CALC_TYPE_UNDEFINED`|
    | cacheMode  | `KROPE_CTKV`|

    **数据规格**：

    | tensor名字| 数据类型 | 数据格式 | 维度信息|
    | --- | --- | --- | --- |
    | `qNope` | float16| nd | [32, 128, 512]|
    |`qRope`  |float16| nd |  [7168, 128, 64]|
    |`ctKV`  |  float16| nd  |[160, 128, 1, 512]  |
    | `kRope` | float16 | nd  | [160, 128, 1, 64] |
    | `blockTables` | int32 | nd  | [32, 5] |
    | `contextLens` | int32 | nd  | [32] |
    | `attenOut` | float16| nd | [32, 128, 512] |
