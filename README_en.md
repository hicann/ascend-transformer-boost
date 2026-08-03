# Ascend Transformer Boost

🔥 [Sept 2025] Initial release.

## 1. What Is ATB?

### Introduction

The Ascend Transformer Boost (ATB for short) is an efficient and reliable acceleration library designed for Transformer model training and inference based on Huawei Ascend AI processors. For details on its working mechanisms, see [ATB Acceleration Mechanisms](docs/ATB_mechanisms.md).

### Software Architecture
<!--
![Architecture](docs/images/architecture.png)
-->
The preceding figure shows the ATB architecture. Its interface functions are divided into three parts:

- Provides optimized fused operators (operations) so that you can use corresponding operators to complete desired computations as required.
- Provides a graph operator mechanism so that you can design graph operators based on specific models, use the native operators provided by the ATB and the created custom operators to build graph operators, and complete the computations.
- Provides a plugin mechanism so that you can customize operators as required.

### ATB Directory Structure

ATB's directory structure is as follows:

```
ascend-transformer-boost
├── 3rdparty            // Third-party dependency library folder
├── build               // Directory for storing files output after build
├── ci                  // Configuration files related to continuous integration
├── docs                // Documentation
├── example             // Operator call sample code, including directly runnable demos
├── include             // Directory for storing public header files
├── ops_configs         // Directory for storing operator input/output data specification constraint files
├── ops_customize       // Directory for storing files related to custom operations
├── output              // Build output folder
├── scripts             // Directory for storing script files
├── src                 // Main source code directory
│   ├── atb
│   ├── kernels
│   │   ├── configs     // Supported configurations
│   │   ├── include     // Directory for storing header files of each operator
│   │   ├── kernels     // Directory for storing a single operator
│   │   ├── lcal        // Directory for storing communication operators
│   │   ├── mixkernels  // Directory for storing fused operators
│   │   ├── tbe_adapter // Source code related to the TBE adapter
│   │   └── CMakeLists.txt
│   ├── ops
│   │   ├── ops_common
│   │   ├── ops_infer   // Inference operators
│   │   └── ops_train   // Training operators
│   ├── torch_atb       // ATB library files related to PyTorch
│   └── CMakeLists.txt
├── tests               // Test code
└── torch_atb
```

### Why ATB?

- The ATB accelerates Transformer models by optimizing core operators such as matrix multiplication and the implementation of attention mechanisms.
- The ATB makes full use of the hardware features of the Ascend AI Processor, such as compute, storage bandwidth, and memory bandwidth, and further improves performance and efficiency through technologies like hardware acceleration and data reuse.
- It provides basic high-performance operators at the underlying layer and efficient operator combination.
- It supports multiple model frameworks, such as PyTorch, MindSpore, and PaddlePaddle.

## 2. Environment Setup

### Version Compatibility

ATB's APIs guarantee ABI compatibility for one year forward and backward. Without involving new features, if a caller upgrades to any ATB version released within one year, no compatibility issues will occur. Due to adjustments in the CANN package directory structure, ATB version 8.5 and the master branch must be used with toolkit version 8.5 or higher.

### Quick Installation of CANN Software

This section provides example commands for quickly installing the CANN software. For more installation steps, see the [Detailed Installation Guide](#detailed-cann-installation-guide).

#### Preparations

For both online and offline installation, ensure that the Python environment and pip3 are available. Currently, CANN supports Python 3.7.x to 3.11.4.
For offline installation, click the [Download Link]](https://www.hiascend.com/developer/download/community/result?module=cann) to download the CANN software package and upload it to any path on the installation environment.

#### Installing CANN

Due to adjustments in the CANN package directory structure, ATB version 8.5 and the master branch must be used with toolkit version 8.5 or higher.

```shell
chmod +x Ascend-cann-toolkit_${VERSION}_linux-$(arch).run  # ${VERSION} represents the CANN version, for example, 8.2.RC1.
./Ascend-cann-toolkit_${VERSION}_linux-$(arch).run --install
```

#### Post-installation Configuration

Configure the environment variable script `set_env.sh`. The following example uses `${HOME}/Ascend` as the installation path.

```sh
source ${HOME}/Ascend/ascend-toolkit/set_env.sh
```

Install the Python third-party libraries required for service runtime (if installing as the root user, remove `--user` from the command).

```sh
pip3 install attrs cython 'numpy>=1.19.2,<=1.24.0' decorator sympy cffi pyyaml pathlib2 psutil protobuf==3.20.0 scipy requests absl-py --user
```

#### Installing the ops Operator Package

Before installation, ensure that the compatible version of the toolkit is installed and the environment variables are configured.

```shell
chmod +x Ascend-cann-${chip_type}-ops_${VERSION}_linux-$(arch).run  # ${chip_type} represents the Ascend product type, for example, A3.
./Ascend-cann-${chip_type}-ops_${VERSION}_linux-$(arch).run --install
```

### Detailed CANN Installation Guide

You can refer to the [Ascend documentation](https://www.hiascend.com/document) > CANN Community Edition > Software Installation to view the CANN software installation guide. Select according to the machine, operating system, and use case, and then read the detailed installation steps.

### ATB Installation and Deployment Dependencies

Before building the ATB, refer to [ATB Package Installation and Deployment](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/850alpha001/acce/ascendtb/ascendtb_0034.html) to check the version requirements for dependencies and proceed to install and deploy dependencies.

### Tool Version Requirements and Installation

After installing CANN, you can install some tools to facilitate subsequent development. For details, see the following:

* [CANN Dependencies](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/850alpha001/softwareinst/instg/instg_0045.html?Mode=PmIns&InstallType=local&OS=Debian&Software=cannToolKit)
* [Post-Installation Operations for CANN](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/850alpha001/softwareinst/instg/instg_0094.html?Mode=PmIns&InstallType=local&OS=Debian&Software=cannToolKit)

## 3. Quick Start

### ATB Build

 - Download ATB

    ```sh
    git clone https://gitcode.com/cann/ascend-transformer-boost.git
    ```

   You can choose your desired branch.
 - Set environment variables
    Before compilation, you need to install NNAL (see [FAQ](docs/faq.md#run-package-usage) for installation instructions) and set the environment variable `ATB_BUILD_DEPENDENCY_PATH` according to the NNAL installation path:

    ```sh
    export ATB_BUILD_DEPENDENCY_PATH={nnal install path}/nnal/atb/latest/atb/cxx_abi_{cxx_abi_version}
    ```

    Note: If not set, the default path `/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_{cxx_abi_version}` will be used.
 - Build ATB
    Build the ATB and set its environment variables.

    ```sh
    cd ascend-transformer-boost
    bash scripts/build.sh
    source output/atb/set_env.sh
    ```

    Note: This build process involves two steps: 1. Pulling and building the operator library/MKI; 2. Building the ATB.
 - For more information about build commands, see [Compilation and Build](docs/compile_and_build.md).

### Call Sample

The sample code in this section demonstrates how to call operators using Python and C++.

#### Python

Before running Python code, you need to import the ATB Python API module torch_atb. The operation of this plugin depends on PyTorch and torch_npu. For details about the version requirements and installation guide, see [ATB Package Installation and Deployment](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/850alpha001/acce/ascendtb/ascendtb_0034.html).
After installing PyTorch and torch_npu, you need to manually install torch_atb. There are two installation methods:

- Run the `./Ascend-cann-nnal_${version_info}_linux-aarch64.run --install --torch_atb` command.
- Run the `bash scripts/build.sh --torch_atb` command during the build process. A `.whl` file of torch_atb will be generated in the `output/whl` folder. You can run the following command to install it:

    ```sh
    pip3 install torch_atb-{version}-py3-none-any.whl
    ```

The following code shows how to call an operator using Python. Do not run the code in the directory with the same name in the ATB code repository.

```Python
import torch
import torch_atb# Import the ATB Python API module.

#Create a parameter object.
linear_param = torch_atb.LinearParam()
linear_param.has_bias = False

#Create an operator object.
op = torch_atb.Operation(linear_param)

#Prepare input data.
x = torch.randn(2, 3, dtype=torch.float16).npu()
y = torch.randn(2, 3, dtype=torch.float16).npu()

#Use the forward method to complete the operation and obtain the output.
outputs = op.forward([x, y])
torch.npu.synchronize()
```

To view the call result, print the first output.

```Python
result=outputs[0].cpu().numpy()
print(result)
```

For code writing guidance, refer to [Operator Usage Guide (ATB Python API)](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/850alpha001/acce/ascendtb/ascendtb_0077.html).

#### C++

In the `example/op_demo` directory of the ATB repository, there are multiple operator call demos that do not depend on a test framework and can run immediately after build. Navigate to the corresponding directory and run the following command to complete the call and execution of an operator. For the complete code, refer to `example/op_demo/faupdate/faupdate_demo.cpp`. The following shows only the essential:

```c++
// Set the device ID, create a context, and set the stream.
atb::Context *context = nullptr;
void *stream = nullptr;

CHECK_STATUS(aclInit(nullptr));
CHECK_STATUS(aclrtSetDevice(DEVICE_ID));
CHECK_STATUS(atb::CreateContext(&context));
CHECK_STATUS(aclrtCreateStream(&stream));
context->SetExecuteStream(stream);

// Create an operator.
atb::Operation *faupdateOp = nullptr;
CHECK_STATUS(CreateFaUpdateOperation(&faupdateOp));
// Prepare the input tensor.
atb::VariantPack variantPack;
CHECK_STATUS(PrepareInTensor(context, stream, variantPack.inTensors)); // Put in the input tensor.
// Prepare the output tensor.
atb::Tensor output;
CHECK_STATUS(CreateTensor(ACL_FLOAT, aclFormat::ACL_FORMAT_ND, {LOCALOUT_DIM_1, LOCALOUT_DIM_2}, output));
variantPack.outTensors = {output}; // Put in the output tensor.

uint64_t workspaceSize = 0;
// Calculate the workspace size.
CHECK_STATUS(faupdateOp->Setup(variantPack, workspaceSize, context));
uint8_t *workspacePtr = nullptr;
if (workspaceSize > 0) {
    CHECK_STATUS(aclrtMalloc((void **)(&workspacePtr), workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
}
// Execute faupdate.
CHECK_STATUS(faupdateOp->Execute(variantPack, workspacePtr, workspaceSize, context));
CHECK_STATUS(aclrtSynchronizeStream(stream)); // Synchronize the stream and wait until the computation on the device is complete.

// Free resources.
for (atb::Tensor &inTensor : variantPack.inTensors) {
    CHECK_STATUS(aclrtFree(inTensor.deviceData));
}
for (atb::Tensor &outTensor : variantPack.outTensors) {
    CHECK_STATUS(aclrtFree(outTensor.deviceData));
}
if (workspaceSize > 0) {
    CHECK_STATUS(aclrtFree(workspacePtr));
}
CHECK_STATUS(atb::DestroyOperation(faupdateOp)); // operation, an object concept, released first
CHECK_STATUS(aclrtDestroyStream(stream));
CHECK_STATUS(DestroyContext(context)); // context, a global resource, released later
CHECK_STATUS(aclFinalize());
```

File build: Go to the `example/op_demo/faupdate` directory and run the `bash build.sh` command to complete build and execution.

A success message appears as shown below:

```sh
faupdate demo success!
```

For code writing guidance, refer to [Single-Operator](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/850alpha001/acce/ascendtb/ascendtb_0046.html).

#### Sample Security Statement

The examples in the `example` directory are intended to provide a minimal implementation for learning, developing, and debugging ATB features. Their core objective is to demonstrate ATB's major functionalities using the most streamlined code possible, **not to provide production-grade security guarantees**. Compared with mature production-grade usage, the security features in these examples (such as input validation and boundary checking) are relatively limited.

ATB does not recommend that you directly use the sample as the service code, and does not ensure the security of such practices. If you apply the sample code in the `example` to your own real-world services and security issues arise, ATB does not bear the responsibility.

### Logs and Environment Variables

- ATB logs:
  [Logs and Debugging](https://gitcode.com/cann/ascend-transformer-boost/blob/master/docs/%E6%97%A5%E5%BF%97%E4%B8%8E%E8%B0%83%E8%AF%95.md)
- ATB logs have partially adapted to CANN logs. Details about environment variables:
  [CANN Community Edition Documentation/Environment Variables](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/83RC1alpha002/maintenref/envvar/envref_07_0119.html)

## 4. Custom Operator Development

You can refer to the following documents to develop custom operators:

- [Starting with a Simple Operator](docs/starting_from_a_simple_operator.md): Using the addition of a simple `Add` operator as an example, this document introduces the deliverables and development process for ATB operator development, ideal for beginners.
- [Development Guide](docs/development_guide.md): Using a fused operator as an example, this document details the ATB operator development process and how to perform functional, precision, and performance testing on operators.
Note: If you encounter any issues during development, refer to [ATB Logs and Debugging](docs/logging_and_debugging.md) for troubleshooting.

## 5. Contributing

1. Fork the repository.
2. Modify and commit code.
3. Create a pull request (PR).

For details, see [Contributing](docs/contributing.md).

## 6. Learning Resources

- [Compilation and Build](docs/compile_and_build.md): Description of ATB build commands.
- [Starting with a Simple Operator](docs/starting_from_a_simple_operator.md): Using the addition of a simple `Add` operator as an example, this document introduces the deliverables and development process for ATB operator development.
- [Development Guide](docs/development_guide.md): Using a fused operator as an example, this document details the ATB operator development process and how to perform functional, precision, and performance testing on operators.
- [Contributing](docs/contributing.md): Describes how to contribute code.
- [Logs and Debugging](docs/logging_and_debugging.md): Introduces ATB's log-related environment variables and debugging methods.
- [API Reference](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/850alpha001/API/ascendtbapi/ascendtb_01_0098.html): Lists the ATB APIs and terms.
- [FAQ](docs/faq.md): Presents common issues and solutions during ATB build, installation, and use.
- [Issues](https://gitcode.com/cann/ascend-transformer-boost/issues): Submit issues found.

## 7. References

[CANN Community Edition Documentation](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/850alpha001/index/index.html)
[ATB Community Edition Documentation](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/850alpha001/acce/ascendtb/ascendtb_0001.html)
