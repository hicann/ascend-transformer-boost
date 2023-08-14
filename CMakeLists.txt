cmake_minimum_required(VERSION 3.8)
project("acltransformer")

#options, these can be pass in by cmake argument
option(USE_UT_TEST "USE_UT_TEST" OFF)
option(USE_GCOV "USE_GCOV" OFF)
option(USE_CXX11_ABI "USE_CXX11_ABI" ON)
option(USE_OPTIMIZE "USE_OPTIMIZE" ON)

message(STATUS "USE_UT_TEST:${USE_UT_TEST}")


# compile flag
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/output/acltransformer"
        CACHE PATH "..." FORCE)
endif()
message(STATUS "CMAKE_INSTALL_PREFIX:${CMAKE_INSTALL_PREFIX}")
set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)
set(CMAKE_SKIP_RPATH TRUE)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fexceptions")
if(USE_GCOV)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
    set(CMAKE_CXX_OUTPUT_EXTENSION_REPLACE 1)
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wno-conversion-null")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pipe -fstack-protector-all")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fno-use-cxa-atexit")
if(USE_OPTIMIZE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wl,--build-id=none")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-strict-aliasing -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions")
set(LD_FLAGS_GLOBAL "-shared -rdynamic -ldl -Wl,-z,relro \
    -Wl,-z,now -Wl,-z,noexecstack -Wl,--build-id=none")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} \
    ${LD_FLAGS_GLOBAL} -fexceptions")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${LD_FLAGS_GLOBAL} \
    -pie -fPIE")
if(TORCH_18)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTORCH_18")
endif()
if(USE_TORCH_RUNNER)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_TORCH_RUNNER")
endif()
if(USE_LCCL_RUNNER)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_LCCL_RUNNER")
endif()
if(USE_HCCL_RUNNER)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_HCCL_RUNNER")
endif()
if(USE_CXX11_ABI)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
endif()
if(TORCH_GET_TENSOR_NPU_FORMAT_OLD)
    message(STATUS "define TORCH_GET_TENSOR_NPU_FORMAT_OLD")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTORCH_GET_TENSOR_NPU_FORMAT_OLD")
endif()
if(TORCH_SETCUSTOMHANDLER)
    message(STATUS "define TORCH_SETCUSTOMHANDLER")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTORCH_SETCUSTOMHANDLER")
endif()
include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/include
    ${PROJECT_SOURCE_DIR}/3rdparty/asdops/include
    ${PROJECT_SOURCE_DIR}/3rdparty/googletest/include
    ${PROJECT_SOURCE_DIR}/3rdparty/nlohmannJson/include
    ${PROJECT_SOURCE_DIR}/3rdparty/lccl/include
    $ENV{ASCEND_HOME_PATH}/include
    $ENV{PYTHON_INCLUDE_PATH}
    $ENV{PYTORCH_INSTALL_PATH}/include
    $ENV{PYTORCH_INSTALL_PATH}/include/torch/csrc/api/include
    $ENV{PYTORCH_NPU_INSTALL_PATH}/include)

link_directories(
    ${PROJECT_SOURCE_DIR}/3rdparty/asdops/lib
    ${PROJECT_SOURCE_DIR}/3rdparty/googletest/lib
    ${PROJECT_SOURCE_DIR}/3rdparty/lccl/lib
    $ENV{ASCEND_HOME_PATH}/lib64
    $ENV{PYTHON_LIB_PATH}
    $ENV{PYTORCH_INSTALL_PATH}/lib
    $ENV{PYTORCH_NPU_INSTALL_PATH}/lib)

add_subdirectory(core)
if(USE_EXAMPLES)
    add_subdirectory(models)
    add_subdirectory(torch)
endif()
if(USE_UT_TEST)
    add_subdirectory(tests)
endif()
add_subdirectory(tools)

install(FILES ${PROJECT_SOURCE_DIR}/scripts/set_env.sh DESTINATION ./)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/3rdparty/asdops/bin DESTINATION .)
install(FILES ${PROJECT_SOURCE_DIR}/3rdparty/asdops/lib/libasdops.so
        DESTINATION lib)
install(FILES ${PROJECT_SOURCE_DIR}/3rdparty/asdops/lib/libops_common.so
        DESTINATION lib)
if(USE_LCCL_RUNNER)
    install(FILES ${PROJECT_SOURCE_DIR}/3rdparty/lccl/lib/liblccl.so
            DESTINATION lib)
    install(FILES ${PROJECT_SOURCE_DIR}/3rdparty/lccl/lib/lccl.o
            DESTINATION lib)
endif()
install(DIRECTORY ${PROJECT_SOURCE_DIR}/3rdparty/asdops/host DESTINATION .)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/3rdparty/asdops/device DESTINATION .)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/3rdparty/asdops/platform_configs
        DESTINATION .)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/3rdparty/asdops/include/asdops
        DESTINATION include)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/core/include/acltransformer
        DESTINATION include)