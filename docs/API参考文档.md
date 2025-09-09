# API参考文档
## Python

### 完整调用示例

```Python
import torch
import torch_atb#导入ATB Python API模块

#创建参数对象
linear_param = torch_atb.LinearParam()
linear_param.has_bias = False

#创建算子对象
op = torch_atb.Operation(linear_param)

#准备输入数据
x = torch.randn(2, 3, dtype=torch.float16).npu()  
y = torch.randn(2, 3, dtype=torch.float16).npu()

#使用forward方法完成操作，并获取输出
outputs = op.forward([x, y]) 
torch.npu.synchronize()
```

### 编写指导

[算子使用指导（ATB Python API）-CANN商用版8.2.RC1-昇腾社区](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/acce/ascendtb/ascendtb_0077.html)

## C++

### 完整调用示例

在ATB仓库的`example/op_demo`目录下，存放了多个不依赖测试框架、即编可用的算子调用Demo示例。进入对应目录执行如下命令就可完成一个算子的调用执行。

### 文件编译说明

可查看`example/op_demo`任意子目录下对应的README.md，获取编译说明。

以`example/op_demo/linear`下的调用示例为例，编译并执行后，输出以下内容，表示调用成功：

```
Using cxx_abi=0
faupdate demo success!
```

若是输出其它内容，则调用失败，使用如下命令可输出执行过程的详细日志，帮助进行问题定位：

```sh
export ASCEND_SLOG_PRINT_TO_STDOUT=1 && export ASCEND_MODULE_LOG_LEVEL=ATB=1:$ASCEND_MODULE_LOG_LEVEL
```

### 算子调用示例编写

[单算子-CANN商用版8.2.RC1-昇腾社区](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/acce/ascendtb/ascendtb_0046.html)
