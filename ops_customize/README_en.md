# Custom Operator Directory for External Developers of ATB

## Introduction

An independent development directory is set for external developers, who are allowed to customize operators based on the customize_block_copy operation in this directory. This directory supports independent build and tests, as well as build together with the ATB.

## Instruction

The following uses the customize_block_copy operation as an example.

### Method 1: Independent Build

#### Installing CANN

```shell
chmod +x Ascend-cann-toolkit_$(version)_linux-$(arch).run
./Ascend-cann-toolkit_$(version)_linux-$(arch).run --install
```

#### Post-installation Configuration

Configure the environment variable script `set_env.sh`. The following example uses `${HOME}/Ascend` as the installation path.

```shell
source ${HOME}/Ascend/ascend-toolkit/set_env.sh
```

#### Installing NNAL

```shell
chmod +x Ascend-cann-nnal_$(version)_linux-$(arch).run
./Ascend-cann-nnal_$(version)_linux-$(arch).run --install
```

#### Post-installation Configuration

Configure the environment variable script `set_env.sh`. The following example uses `${HOME}/Ascend` as the installation path.

```shell
source ${HOME}/Ascend/nnal/atb/set_env.sh
```

#### Building the Custom Operator Directory

```shell
cd ascend-transformer-boost/ops_customize
bash build.sh
```

Currently, this script supports the following options: default|clean|unittest| --use_cxx11_abi=0|--use_cxx11_abi=1|--debug|--msdebug.
Specifically,

- `default` (default): Builds `ops_customize`.
- `clean`: Clears all build history and deletes the build directory.
- `unittest`: Builds and runs unit tests for `ops_customize`.
- `--use_cxx11_abi=0`: Disables the `C++11 ABI`.
- `--use_cxx11_abi=1`: Enables `C++11 ABI`.
- `--debug`: Sets the build type to `Debug`.
- `--msdebug`: Enables the `MSDebug` mode for debugging the operator kernel code.

#### Test Case Execution

Execute the test case for the customize_block_copy operation.

```shell
bash build.sh unittest
```

### Method 2: Build Together with ATB

#### Preparing Environment Variables

The following example uses `${HOME}/Ascend` as the installation path.

```shell
source ${HOME}/Ascend/ascend-toolkit/set_env.sh
source ${HOME}/Ascend/nnal/atb/set_env.sh
export ATB_BUILD_DEPENDENCY_PATH=${ATB_HOME_PATH}
```

#### Building ATB with Custom Operators

```shell
cd ascend-transformer-boost
bash scripts/build.sh customizeops
```

#### Test Case Execution

Build the ATB with custom operators and test cases, and execute the test case for the customize_block_copy operation.

```shell
bash scripts/build.sh customizeops --customizeops_tests
source ./output/atb/set_env.sh
cd ./build/ops_customize/ops/customize_blockcopy/tests/ && ./customize_blockcopy_test
```
