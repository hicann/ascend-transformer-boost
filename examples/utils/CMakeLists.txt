file(GLOB_RECURSE SOURCE_FILES "${CMAKE_CURRENT_LIST_DIR}/*.cpp")
include_directories($ENV{PYTHON_INCLUDE_PATH}
    $ENV{PYTORCH_INSTALL_PATH}/include
    $ENV{PYTORCH_INSTALL_PATH}/include/torch/csrc/api/include
    $ENV{PYTORCH_NPU_INSTALL_PATH}/include)

link_directories($ENV{PYTHON_LIB_PATH}
        $ENV{PYTORCH_INSTALL_PATH}/lib
        $ENV{PYTORCH_NPU_INSTALL_PATH}/lib)

add_library(examples_util OBJECT ${SOURCE_FILES})
