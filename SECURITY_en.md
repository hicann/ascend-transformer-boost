# Security Statement

## Recommended Running Users

To ensure security and minimize permissions, you are not advised to use administrator accounts such as `root`.

## File Permission Control

- You are advised to set the system `umask` value to `0027` or higher on hosts (both physical and virtual hosts) and containers. This ensures that new folders have a default maximum permission of `750` and new files have a default maximum permission of `640`.
- You are advised to take security measures such as permission control on sensitive content, including personal privacy data, business assets, source files, and various files saved during operator development. For example, permissions for the project installation directory and public input data files must follow the recommendations in [A-Recommended Maximum Permissions for Files and Folders in Different Scenarios](#arecommended-maximum-permissions-for-files-and-folders-in-different-scenarios).
- During installation and usage, you must enforce proper permission control, referring to the same [A–Recommended Maximum Permissions for Files and Folders in Different Scenarios](#arecommended-maximum-permissions-for-files-and-folders-in-different-scenarios).

## Build Security Statement

When you are building and installing this project from the source code, some intermediate files will be generated. After the build is complete, you are advised to perform permission control on the intermediate files to ensure file security.

## Runtime Security Statement

- You are advised to write an operator calling script based on the operating environment resources. If the operator calling script does not match the resource status, for example, the space used for generating input data and benchmark computing results exceeds the memory capacity limit, or the data stored locally in the script exceeds the disk space, an error may occur and the process may exit unexpectedly.
- When an operator encounters an exception at runtime, it will exit the process and print an error message. It is advised to locate the specific error cause based on the error message, including methods such as setting the operator to execute synchronously and viewing log files.
- When calling an operator via [PyTorch](https://gitcode.com/ascend/pytorch), a runtime error may occur due to version mismatch. For details, see [PyTorch Security Statement](https://gitcode.com/Ascend/pytorch/blob/master/SECURITYNOTE.md).

## Public Network Address Statement

The public network addresses contained in the code of this project are as follows:

| Type | Open-Source Code Address| File Name                    | Public IP/Public URL/Domain Name/Email/Archive File Address                                          | Description                                        |
| :---: | :----------: | :------------------------- | :------------------------------------------------------------------------------------------ | :----------------------------------------------- |
| Dependency |    N/A   | 3rdparty/pybind11          | https://github.com/pybind/pybind11.git                                                      | Download the source code from GitHub as a build dependency.                  |
| Dependency |    N/A   | 3rdparty/nlohmannjson      | https://github.com/nlohmann/json.git                                                        | Download the nlohmann/json source code from GitHub as a build dependency.     |
| Dependency |    N/A   | 3rdparty/catlass           | https://gitcode.com/cann/catlass.git                                                        | Download the catlass source code from GitCode as a build dependency.          |
| Dependency |    N/A   | 3rdparty/ascend-boost-comm | https://gitcode.com/cann/ascend-boost-comm.git                                              | Download the ascend-boost-comm source code from GitCode as a build dependency.|
| Dependency |    N/A   | 3rdparty/doxygen           | https://github.com/doxygen/doxygen/releases/download/Release_1_9_6/doxygen-1.9.6.src.tar.gz | Download the Doxygen-1.9.6 source code from GitHub as a build dependency.     |
| Dependency |    N/A   | 3rdparty/cpp-stub          | https://github.com/coolxv/cpp-stub.git                                                      | Download the cpp-stub source code from GitHub as a build dependency.          |
| Dependency |    N/A   | 3rdparty/googletest        | https://github.com/google/googletest.git                                                    | Download the GoogleTest source code from GitHub as a build dependency.        |

---

## Vulnerability Handling Mechanism

[Vulnerability Management](https://gitcode.com/cann/community/blob/master/security/security.md)

## Appendix

### A–Recommended Maximum Permissions for Files and Folders in Different Scenarios

| Type          | Maximum Linux Permission|
| -------------- | ---------------  |
| User's home directory                       |   750 (rwxr-x---)           |
| Program files (including scripts and library files)      |   550 (r-xr-x---)            |
| Program file directory                     |   550 (r-xr-x---)           |
| Configuration files                         |  640 (rw-r-----)            |
| Configuration file directory                     |   750 (rwxr-x---)           |
| Log files (recorded or archived)       |  440 (r--r-----)            |
| Log files (being recorded)               |    640 (rw-r-----)          |
| Log file directory                     |   750 (rwxr-x---)           |
| Debug files                        |  640 (rw-r-----)        |
| Debug file directory                    |   750 (rwxr-x---) |
| Temporary file directory                     |   750 (rwxr-x---)  |
| Maintenance and upgrade file directory                 |   770 (rwxrwx---)   |
| Service data files                     |   640 (rw-r-----)   |
| Service data file directory                 |   750 (rwxr-x---)     |
| Key components, private keys, certificates, and ciphertext file directory   |  700 (rwx------)     |
| Key components, private keys, certificates, and ciphertext files       | 600 (rw-------)     |
| APIs and scripts for encryption and decryption           |   500 (r-x------)       |
