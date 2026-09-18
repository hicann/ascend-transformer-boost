# Compilation and Build

## ATB Build

### Downloading Source Code

```shell
git clone https://gitcode.com/cann/ascend-transformer-boost.git
```

You can choose your desired branch.

### Build

Go to the ATB root directory and build the library.

```shell
cd ascend-transformer-boost
bash scripts/build.sh
```

Note: This build process involves two steps: 1. Pulling and building the operator library/MKI; 2. Building the ATB. For more information about the commands, see the `README_en.md` and `scripts/build.sh` files in the ATB root directory.

### ATB Build Description

The basic ATB build command is `bash build.sh`. The default build mode generates version information and creates an installation package. By default, Python APIs are not built. The following parameters can be added to implement different functions:

- `--use_cxx11_abi=1`: Enables `C++11 ABI`.
- `--use_cxx11_abi=0`: Disables `C++11 ABI`.
- `--verbose`: Enables detailed build output.
- `--asan`: Enables memory error detection and forcibly sets the mode to `Debug`.
- `--skip_build`: Skips the build process.
- `--debug`: Sets the build type to `Debug`.
- `--msdebug`: Enables the `MSDebug` mode for debugging the operator kernel code.
- `--ascendc_dump`: Enables the `AscendC_Dump` mode for debugging the operator kernel code.
- `--clean-first`: Clears all build history and deletes the build directory before the build.
- `--src-only`: Builds only the source code.
- `--torch_atb`: Builds pybind11 and generates a .whl package. Then, you can use the Python APIs after installing it using pip.
- `--customizeops_tests`: Builds unit tests for `ops_customize`.
- `default` (default): Builds the ATB.
- `testframework`: Builds the test framework and the C++ unit test binaries (`atb_unittest`, `atb_cinterface`, and `kernels_unittest`), generates version information, and packages the test framework. The test binaries are built but not run; run them with `unittest` or `kernelunittest` (optionally with `--skip_build`).
- `unittest`: Builds and runs unit tests and kernel interface tests.
- `kernelunittest`: Builds and runs kernel unit tests.
- `pythontest`: Builds and runs Python tests.
- `kernelpythontest`: Builds and runs kernel Python tests.
- `torchatbtest`: Builds and runs the Torch ATB test.
- `csvopstest`: Builds and runs CSV operation tests.
- `infratest`: builds and runs the infrastructure test.
- `hitest`: builds HiTest, sets HiTest environment variables, generates version information, and packages the test framework and HiTest.
- `fuzztest`: Builds a fuzz test, generates fuzz test cases, and runs the fuzz test.
- `alltest`: Builds and runs all tests.
- `clean`: Clears all build history and deletes the build directory.
- `gendoc`: Generate documentation.
- `customizeops`: Builds `ops_customize`, generates version information, and creates an installation package.

### Key ATB Files

1. `scripts` directory
   - `install.sh`: installation script
   - `uninstall.sh`: uninstallation script
   - `build.sh`: build script
   - `release.sh`: automatic build and packaging script
2. `include/atb` directory
   - `set_env.sh`: ATB environment variable settings
3. `output` directory
   - `version.info`: version information
   - `{arch}/Ascend-cann-atb_{version}_linux-{arch}.run`: built ATB package
4. `output/atb/cxx_abi_0/lib` directory (or `output/atb/cxx_abi_1/lib`, depending on the ABI version)
   - `libatb.so`: dynamic link library file of the Transformer acceleration library
   - `libasdops.so`: dynamic link library file of the operator package
5. `ops_configs` directory
   - `atb_ops_info.ini`: description file of the operator input and output specifications

## Configuration Files

### `build.sh`

File name: `scripts/build.sh`
You can set the log storage directory, log file, and compiler version in this file. Generally, you do not need to modify this file.

### `set_env.sh`

​File name: `scripts/set_env.sh`
After the acceleration library is installed, the process-level environment variable setting script `set_env.sh` is provided to automatically set environment variables. The environment variables automatically become invalid after the user process ends.
For details about the variables, see [Environment Variable Reference](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/850alpha001/acce/ascendtb/ascendtb_0032.html).
