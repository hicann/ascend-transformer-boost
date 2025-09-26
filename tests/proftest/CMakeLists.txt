#
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#

file(GLOB_RECURSE UTILS_SOURCES "utils/src/*.cpp")
add_library(atb_proftest_utils STATIC ${UTILS_SOURCES})
target_include_directories(atb_proftest_utils PRIVATE utils/include)
target_include_directories(atb_proftest_utils PRIVATE ${PROJECT_SOURCE_DIR}/)
target_include_directories(atb_proftest_utils PRIVATE ${PROJECT_SOURCE_DIR}/3rdparty)
target_include_directories(atb_proftest_utils PRIVATE ${PROJECT_SOURCE_DIR}/3rdparty/googletest/include)
target_include_directories(atb_proftest_utils PRIVATE ${ASCEND_HOME_PATH}/include/)
target_link_directories(atb_proftest_utils PRIVATE ${PROJECT_SOURCE_DIR}/3rdparty/googletest/lib ${PROJECT_SOURCE_DIR}/3rdparty/benchmark/build/src ${ASCEND_HOME_PATH}/lib64/ ${ATB_HOME_PATH}/lib)
target_compile_options(atb_proftest_utils PRIVATE "-Wno-write-strings")
target_compile_options(atb_proftest_utils PRIVATE -Wno-sign-compare -Wno-narrowing)
set_target_properties(atb_proftest_utils PROPERTIES LINK_FLAGS "-Wl,-rpath-link,${PROJECT_SOURCE_DIR}/3rdparty/asdops/lib/")
target_link_libraries(atb_proftest_utils PRIVATE atb -lgtest -lgtest_main -lc_sec)


file(GLOB_RECURSE UTILS_SOURCES "layer_test_framework/*.cpp")
add_library(layer_test_framework STATIC ${UTILS_SOURCES})
target_include_directories(layer_test_framework PRIVATE layer_test_framework)
target_include_directories(layer_test_framework PRIVATE layer_test_framework/core/include)
target_include_directories(layer_test_framework PRIVATE layer_test_framework/models)
target_include_directories(layer_test_framework PRIVATE layer_test_framework/operations)
target_include_directories(layer_test_framework PRIVATE ${PROJECT_SOURCE_DIR}/)
target_include_directories(layer_test_framework PRIVATE ${PROJECT_SOURCE_DIR}/3rdparty)
target_include_directories(layer_test_framework PRIVATE ${PROJECT_SOURCE_DIR}/3rdparty/googletest/include)
target_include_directories(layer_test_framework PRIVATE ${PROJECT_SOURCE_DIR}/3rdparty/nlohmannJson/include)
target_include_directories(layer_test_framework PRIVATE ${ASCEND_HOME_PATH}/include/)
target_link_directories(layer_test_framework PRIVATE ${PROJECT_SOURCE_DIR}/3rdparty/googletest/lib ${PROJECT_SOURCE_DIR}/3rdparty/benchmark/build/src ${ASCEND_HOME_PATH}/lib64/ ${ATB_HOME_PATH}/lib $ENV{ASCEND_TOOLKIT_HOME}/lib64)
target_compile_options(layer_test_framework PRIVATE "-Wno-write-strings")
target_compile_options(layer_test_framework PRIVATE -Wno-sign-compare -Wno-narrowing)
set_target_properties(layer_test_framework PROPERTIES LINK_FLAGS "-Wl,-rpath-link,${PROJECT_SOURCE_DIR}/3rdparty/asdops/lib/")
target_link_libraries(layer_test_framework PRIVATE atb -lgtest -lgtest_main -lc_sec -lopapi)

find_package(Threads REQUIRED)
file(GLOB_RECURSE TEST_CASES "test_cases/*")
add_executable(atb_proftest main.cpp ${TEST_CASES})
target_include_directories(atb_proftest PRIVATE utils/include)
target_include_directories(atb_proftest PRIVATE layer_test_framework)
target_include_directories(atb_proftest PRIVATE layer_test_framework/core/include)
target_include_directories(atb_proftest PRIVATE layer_test_framework/models)
target_include_directories(atb_proftest PRIVATE layer_test_framework/operations)
target_include_directories(atb_proftest PRIVATE ${PROJECT_SOURCE_DIR}/)
target_include_directories(atb_proftest PRIVATE ${PROJECT_SOURCE_DIR}/3rdparty)
target_include_directories(atb_proftest PRIVATE ${PROJECT_SOURCE_DIR}/3rdparty/googletest/include)
target_include_directories(atb_proftest PRIVATE ${PROJECT_SOURCE_DIR}/3rdparty/benchmark/include)
target_include_directories(atb_proftest PRIVATE ${PROJECT_SOURCE_DIR}/3rdparty/nlohmannJson/include)
target_include_directories(atb_proftest PRIVATE ${ASCEND_HOME_PATH}/include/)
target_link_directories(atb_proftest PRIVATE ${PROJECT_SOURCE_DIR}/3rdparty/googletest/lib ${PROJECT_SOURCE_DIR}/3rdparty/benchmark/build/src ${ASCEND_HOME_PATH}/lib64/ ${ATB_HOME_PATH}/lib $ENV{ASCEND_TOOLKIT_HOME}/lib64)
target_compile_options(atb_proftest PRIVATE "-Wno-write-strings")
target_compile_options(atb_proftest PRIVATE -Wno-sign-compare -Wno-narrowing)
set_target_properties(atb_proftest PROPERTIES LINK_FLAGS "-Wl,-rpath-link,${PROJECT_SOURCE_DIR}/3rdparty/asdops/lib/")
target_link_libraries(atb_proftest PRIVATE atb_proftest_utils layer_test_framework atb Threads::Threads benchmark -lgtest -lgtest_main -lc_sec -lopapi)
install(TARGETS atb_proftest DESTINATION bin)
