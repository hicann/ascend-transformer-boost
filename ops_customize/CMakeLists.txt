file(GLOB_RECURSE CUSTOMIZE_SRCS
     "${CMAKE_CURRENT_SOURCE_DIR}/common/*.cpp"
     "${CMAKE_CURRENT_SOURCE_DIR}/ops/*.cpp")

list(FILTER CUSTOMIZE_SRCS EXCLUDE REGEX ".*/kernel_implement/.*")
list(FILTER CUSTOMIZE_SRCS EXCLUDE REGEX ".*/tests/.*")

add_library(customize_objs OBJECT ${CUSTOMIZE_SRCS})

add_library(atb_customize SHARED $<TARGET_OBJECTS:customize_objs>)
add_library(atb_customize_static STATIC $<TARGET_OBJECTS:customize_objs>)

set(USR_INCLUDE_DIRS
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/ops
    ${CMAKE_CURRENT_LIST_DIR}/ops/customize_blockcopy/operation_implement
    ${CMAKE_CURRENT_LIST_DIR}/ops/customize_blockcopy/operation_implement/tiling
    ${CMAKE_CURRENT_LIST_DIR}/include
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

target_link_libraries(atb_customize
  PUBLIC
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

target_link_libraries(atb_customize_static
  PUBLIC
    $<TARGET_LINKS:atb_customize>
)

install(TARGETS atb_customize atb_customize_static DESTINATION lib)
install(DIRECTORY include/ DESTINATION include/customize)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/ops_customize/customize_ops_configs DESTINATION ./configs)

add_subdirectory(ops/customize_blockcopy/kernel_implement)

# 添加位置以及是否需要add_subdirectory
find_package(GTest 1.13.0 REQUIRED)
enable_testing()
add_subdirectory(ops/customize_blockcopy/tests)