# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
set(NAMESPACE "AtbOps")
set(OPS_PROJECT_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(OPS_THIRD_PARTY_DIR "${PROJECT_SOURCE_DIR}/3rdparty")
set(MKI_PACKAGE_DIR "${OPS_THIRD_PARTY_DIR}/mki")
set(OP_LIST_YAML_DIR "${CMAKE_CURRENT_LIST_DIR}/configs")
set(MKI_SCRIPT_DIR "${MKI_PACKAGE_DIR}/scripts")
option(USE_MSDEBUG "USE_MSDEBUG" OFF)
message(STATUS "USE_MSDEBUG:${USE_MSDEBUG}")
option(USE_MSSANITIZER "USE_MSSANITIZER" OFF)
message(STATUS "USE_MSSANITIZER:${USE_MSSANITIZER}")

include(${MKI_PACKAGE_DIR}/cmake/host_config.cmake)
include(${MKI_PACKAGE_DIR}/cmake/kernel_config.cmake)

include_directories(
    ${CMAKE_CURRENT_LIST_DIR}
    $ENV{ASCEND_HOME_PATH}/include
)

add_subdirectory(ops)

if(BUILD_TEST_FRAMEWORK OR USE_UNIT_TEST OR USE_PYTHON_TEST OR USE_FUZZ_TEST OR USE_CSV_OPS_TEST OR USE_INFRA_TEST OR USE_ALL_TEST)
    add_subdirectory(tests)
endif()
