# 用户自定义算子目录

## 介绍

此目录为用户自定义算子目录，用户可根据实际场景创建自定义算子并基于ATB框架编译运行。

## 目录概览

```
ascend-transformer-boost/
├── CMakeLists.txt
├── src/
├── include/
├── 3rdparty/
├── ops_configs/
├── scripts/
├── tests/             
├── output/…
└── ops_customize/
    ├── CMakeLists.txt
    ├── configs/
    │   └── // 存放自动生成的yaml
    ├── customize_ops_configs/
    │   └── customize_ops_info.ini
    ├── include/
    │   └── customize_op_params.h
    └── ops/
        ├── example_op/ // 单一op实现
        |   ├── kernel_implement/
        │   │   ├── include/example.h // 实现算子在 AICore/硬件上的底层逻辑
        │   │   ├── op_kernel/example.cpp
        │   │   ├── tiling/
        │   │   │   ├── example_tiling.cpp
        │   │   │   ├── example_tiling.h
        │   │   │   └── tiling_data.h
        │   │   ├── example_kernel.cpp
        │   │   ├── example_operation.cpp
        │   │   └── CMakeLists.txt
        │   ├── operation_implement/ // 实现算子在ATB上层框架中的包装与调用
        │   │   ├── example_operation.cpp // 算子的高层接口与前置逻辑，如：校验输入、推断输出形状、数据转换(atb::Tensor -> LaunchParam)等
        │   │   ├── example_operation.h
        │   │   ├── example_ops_runner.cpp // 构建kernel graph节点
        │   │   └── example_ops_runner.h
        │   └── tests/
        |       ├── example_op_test.cpp
        │       └── CMakeLists.txt
        ├── ops.cpp
        ├── param_to_json.cpp
        ├── sys_check.cpp
        └── CMakeLists.txt
```

## 调用结构

![](https://wiki.huawei.com/vision-file-storage/api/file/download/upload-v2/WIKI202505307002410/23032147/bd08f158aa3e4c7cad0d6bdf6f7f6f07.png)

## 宏与注册机制

- Operation注册

```
// ascend-transformer-boost/ops_customize/ops/xxx/kernel_implement/op_operation.cpp
REG_OPERATION(ExampleOperation);
```

- Kernel注册

```
// ascend-transformer-boost/ops_customize/ops/xxx/kernel_implement/op_kernel.cpp
REG_KERNEL_BASE(ExampleKernel);
```

- CMake宏

```
# ascend-transformer-boost/ops_customize/ops/xxx/kernel_implement/CMakeLists.txt
add_operation(ExampleOperation "${srcs}") # including operation kernel and tiling files in kernel_implement
add_kernel(example ascend910b vector "${kernel_src}" ) # files in kernel_implement/op_kernel; ascend910b can be replaced with ascend310p etc.
```

## 如何创建一个自定义算子

具体样例可参考 `customize_blockcopy` 实现

1. 定义参数
   
   - 在 `ascend-transformer-boost/ops_customize/include/customize_op_params.h` 中新增atb层级接口
   - 在 `ascend-transformer-boost/ops_customize/customize_ops_configs/customize_ops_info.ini`  中配置算子输入输出约束信息
2. 添加测试
   在 `ops/xxx_op/tests/` 下添加自定义算子测试文件
3. 实现Operation
   
   新建 `ops/xxx_op/operation_implement/xxx_operation.h/.cpp`，派生自 `OperationBase`
4. 实现Runner
   同目录下 `xxx_ops_runner.h/.cpp`，派生自 `OpsRunner`，用于搭建 `kernelGraph`
5. 实现tiling & kernel
   
   - tiling相关文件实现：
     `ops/xxx_op/kernel_implement/tiling/xxx_tiling.h`
     `ops/xxx_op/kernel_implement/tiling/xxx_tiling.cpp`
     `ops/xxx_op/kernel_implement/tiling/tiling_data.h`
   - 算子侧注册到mki及部分校验：
     `ops/xxx_op/kernel_implement/xxx_kernel.cpp`
     `ops/xxx_op/kernel_implement/xxx_operation.cpp`
     `ops/xxx_op/kernel_implement/CMakelists.txt` -> Notice: 注册operation及kernel
   - kernel相关文件实现：
     `ops/xxx_op/kernel_implement/op_kernel/xxx.cpp`

## 编译并运行

执行编译脚本时添加编译选项 `customizeops`，对应test目录下的可执行文件位于build目录下

```sh
bash scripts/build.sh customizeops
./build/ops_customize/ops/xxx_op/tests
```

## 常见FAQ
