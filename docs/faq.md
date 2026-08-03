
# FAQ

1. When `libtbe_adapter.so does not exist` pops up,
   you can install the NNAL software package and set `ATB_BUILD_DEPENDENCY_PATH` correctly to solve this problem.
    - For details about the installation procedure, see [Using the RUN Package](#run-package-usage).
    - The code and software package versions map as follows:
        The NNAL software package version must be the same as that of the toolkit and kernels software packages.
    - Execution

        ```sh
        export ATB_BUILD_DEPENDENCY_PATH={nnal install path}/nnal/atb/latest/atb/cxx_abi_{cxx_abi_version}
        ```

        Note: If not set, the default path `/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_{cxx_abi_version}` will be used.
    - RUN package usage<a id="run-package-usage"></a>
       - Obtaining the RUN package
         1. Visit https://www.hiascend.com/developer/download/community.
         2. Select the server for the product type, the product model based on the device model, and the required solution version. Then select the software package in the CANN area to obtain the related `.run` package as prompted.
       - The software package is named `Ascend-cann-nnal_{version}_linux-{arch}.run`.
       {version} indicates the software version, and {arch} indicates the CPU architecture.
       - Installing the .run package (depending on the CANN environment)

           ```sh
           chmod +x Software package name.run # Grant the execute permission on the software package.
           ./Software package name.run --check # Check the consistency and integrity of the software package installation file.
           ./Software package name.run --install # Install the software. You can use `--help` to query installation options.
           ```

           If `xxx install success!` is displayed, the installation is successful.
