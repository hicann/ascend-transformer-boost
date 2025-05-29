file(GLOB_RECURSE USR_COMMON_SRC "${CMAKE_CURRENT_LIST_DIR}/common/*.cpp")
file(GLOB_RECURSE ALL_OPS_SRC "${CMAKE_CURRENT_LIST_DIR}/ops/*.cpp")

set(USR_OPS_SRC "")

foreach(_f IN LISTS ALL_OPS_SRC)
    if (NOT _f MATCHES "/kernel_implement/" AND
        NOT _f MATCHES "/tests/")
        list(APPEND USR_OPS_SRC "${_f}")
    endif()
endforeach()

add_library(atb_customize SHARED
    ${USR_COMMON_SRC}
    ${USR_OPS_SRC}
)

add_library(atb_customize_static STATIC
    ${USR_COMMON_SRC}
    ${USR_OPS_SRC}
)

set(USR_INCLUDE_DIRS
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/ops
    ${CMAKE_CURRENT_LIST_DIR}/ops/example_op/operation_implement
    ${CMAKE_CURRENT_LIST_DIR}/ops/example_op/operation_implement/tiling
    ${CMAKE_CURRENT_LIST_DIR}/ops/customize_blockcopy/operation_implement
    ${CMAKE_CURRENT_LIST_DIR}/ops/customize_blockcopy/operation_implement/tiling
    ${CMAKE_CURRENT_LIST_DIR}/include
    ${CMAKE_CURRENT_LIST_DIR}/common
    $ENV{ASCEND_HOME_PATH}/include
    $ENV{ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/include
    ${CMAKE_SOURCE_DIR}/src/kernels/include
    ${CMAKE_SOURCE_DIR}/src/kernels/tbe_adapter/include
    ${PROJECT_SOURCE_DIR}/src/kernels/tiling
    ${PROJECT_SOURCE_DIR}/3rdparty/mki/include
)

target_include_directories(atb_customize PUBLIC ${USR_INCLUDE_DIRS})
target_include_directories(atb_customize_static PUBLIC ${USR_INCLUDE_DIRS})

set(USR_LINK_LIBS
  dl
  mki
  tbe_adapter
  asdops
  atb_mixops
  ascendcl
  profapi
  lcal
  hccl
  pthread
  acl_op_compiler
)

target_link_libraries(atb_customize PUBLIC ${USR_LINK_LIBS})
target_link_libraries(atb_customize_static PUBLIC ${USR_LINK_LIBS})

install(TARGETS atb_customize atb_customize_static DESTINATION lib)
install(DIRECTORY include/ DESTINATION include/customize)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/ops_customize/customize_ops_configs DESTINATION ./configs)

add_subdirectory(ops/example_op/kernel_implement)
add_subdirectory(ops/customize_blockcopy/kernel_implement)

if (BUILD_USR_OPS_TEST)
    enable_testing()
    add_subdirectory(ops/example_op/tests)
    add_subdirectory(ops/customize_blockcopy/tests)
endif()