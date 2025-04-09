#
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#

file(GLOB_RECURSE SOURCE_FILES "${CMAKE_CURRENT_LIST_DIR}/*.cpp")
add_executable(atb_unittest ${SOURCE_FILES})
target_include_directories(atb_unittest PRIVATE ${CMAKE_CURRENT_LIST_DIR}/mixops)
target_include_directories(atb_unittest PRIVATE ${PROJECT_SOURCE_DIR}/3rdparty)
target_include_directories(atb_unittest PRIVATE ${PROJECT_SOURCE_DIR}/3rdparty/googletest/include)
target_include_directories(atb_unittest PRIVATE ${ASCEND_HOME_PATH}/include/)
target_include_directories(atb_unittest PRIVATE ${PROJECT_SOURCE_DIR}/src/ops)
target_include_directories(atb_unittest PRIVATE ${PROJECT_SOURCE_DIR}/src/ops/include)
target_include_directories(atb_unittest PRIVATE ${PROJECT_SOURCE_DIR}/tests/framework/c++/mixops/)
target_include_directories(atb_unittest PRIVATE ${PROJECT_SOURCE_DIR}/tests/framework/c++/mixops/autogen)
target_link_directories(atb_unittest PRIVATE ${PROJECT_SOURCE_DIR}/3rdparty/googletest/lib ${ASCEND_HOME_PATH}/lib64/)
target_compile_options(atb_unittest PRIVATE "-Wno-write-strings")
target_compile_options(atb_unittest PRIVATE -Wno-sign-compare -Wno-narrowing)
set_target_properties(atb_unittest PROPERTIES LINK_FLAGS "-Wl,-rpath-link,${PROJECT_SOURCE_DIR}/3rdparty/asdops/lib/")
target_link_libraries(atb_unittest PRIVATE atb_test_utils atb_mixops_test_utils mki_test_autogen -lgtest -lgtest_main -lc_sec)
target_compile_definitions(atb_unittest PRIVATE OpSpace=AtbOps)
install(TARGETS atb_unittest DESTINATION bin)