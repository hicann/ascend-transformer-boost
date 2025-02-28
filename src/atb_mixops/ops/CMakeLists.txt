# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
if(NOT EXISTS ${OP_LIST_YAML_DIR}/op_list.yaml)
    execute_process(COMMAND python3 ${MKI_PACKAGE_DIR}/scripts/op_list_utils.py -s ${CMAKE_CURRENT_LIST_DIR} -d ${OP_LIST_YAML_DIR}
                    ERROR_VARIABLE RESULT_INFO
                    RESULT_VARIABLE RESULT)
    if(NOT RESULT EQUAL 0)
        message(FATAL_ERROR "create op_build.yaml failed, error code: ${RESULT}, error info:\n${RESULT_INFO}")
    endif()
endif()

set(OP_LIST_CMD_PARAMS "'${OP_LIST_YAML_DIR}/op_list.yaml', '${MKI_PACKAGE_DIR}/cmake/op_build.cmake'")
execute_process(COMMAND python3 -c "import op_list_utils\nop_list_utils.build_cmake_options(${OP_LIST_CMD_PARAMS})"
                WORKING_DIRECTORY ${MKI_PACKAGE_DIR}/scripts
                ERROR_VARIABLE RESULT_INFO
                RESULT_VARIABLE RESULT)
if(NOT RESULT EQUAL 0)
    message(FATAL_ERROR "create op_build.cmake failed, error code: ${RESULT}, error info:\n${RESULT_INFO}")
endif()

include(${MKI_PACKAGE_DIR}/cmake/op_build.cmake)
 
file(GLOB_RECURSE CMAKELISTS_PATHS "${CMAKE_CURRENT_SOURCE_DIR}/*/CMakeLists.txt")
string(REPLACE "/CMakeLists.txt" "" CMAKELISTS_PATHS "${CMAKELISTS_PATHS}")
foreach(SUB_DIR ${CMAKELISTS_PATHS})
    message(STATUS "add subdirectory: ${SUB_DIR}")
    add_subdirectory(${SUB_DIR})
endforeach()

set_source_files_properties(${BINARY_SRC_LIST} PROPERTIES GENERATED TRUE)

list(APPEND BINARY_SRC_LIST ${CMAKE_CURRENT_LIST_DIR}/param_to_json.cpp)
add_library(atb_mixops SHARED ${BINARY_SRC_LIST})
add_dependencies(atb_mixops ${BINARY_TARGET_LIST})
target_link_libraries(atb_mixops PRIVATE ${ops_objects} mki tiling_api register)
target_link_libraries(atb_mixops PRIVATE "-Wl,--whole-archive" mki_loader "-Wl,--no-whole-archive")

add_library(atb_mixops_static STATIC ${BINARY_SRC_LIST})
add_dependencies(atb_mixops_static ${BINARY_TARGET_LIST})
target_link_libraries(atb_mixops_static PRIVATE ${ops_objects} mki tiling_api register)
target_link_libraries(atb_mixops_static PRIVATE "-Wl,--whole-archive" mki_loader "-Wl,--no-whole-archive")

install(TARGETS atb_mixops DESTINATION lib)
install(TARGETS atb_mixops_static DESTINATION lib)