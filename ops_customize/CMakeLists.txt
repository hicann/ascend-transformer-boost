file(GLOB_RECURSE USR_COMMON_SRC "${CMAKE_CURRENT_LIST_DIR}/common/*.cpp")
file(GLOB_RECURSE USR_OPS_SRC "${CMAKE_CURRENT_LIST_DIR}ops/*/*.cpp")

add_library(atb_customize SHARED
    ${USR_COMMON_SRC}
    ${USR_OPS_SRC}
)

add_library(atb_customize_static STATIC
    ${USR_COMMON_SRC}
    ${USR_OPS_SRC}
)

target_include_directories(atb_customize
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/include
    ${CMAKE_CURRENT_LIST_DIR}/common
    $ENV{ASCEND_HOME_PATH}/include
    $ENV{ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/include
    ${CMAKE_SOURCE_DIR}/src/kernels/include
)

target_include_directories(atb_customize_static
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/include
    ${CMAKE_CURRENT_LIST_DIR}/common
    $ENV{ASCEND_HOME_PATH}/include
    $ENV{ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/include
    ${CMAKE_SOURCE_DIR}/src/kernels/include
)

target_link_libraries(atb_customize 
  PUBLIC 
    dl mki asdops atb_mixops ascendcl profapi lcal hccl pthread acl_op_compiler
)

install(TARGETS atb_customize atb_customize_static DESTINATION lib)
install(DIRECTORY include/ DESTINATION include/customize)

if (BUILD_USR_OPS_TEST)
    enable_testing()
    add_subdirectory(ops/example_op/tests)
endif()