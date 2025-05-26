set(GTest_ROOT "${PROJECT_SOURCE_DIR}/3rdparty/googletest")
find_package(GTest 1.13.0 EXACT REQUIRED
             HINTS "${GTest_ROOT}/lib/cmake/GTest"
             PATHS "${GTest_ROOT}" NO_DEFAULT_PATH)
# set(CMAKE_PREFIX_PATH "${PROJECT_SOURCE_DIR}/3rdparty/googletest" ${CMAKE_PREFIX_PATH})
# find_package(GTest REQUIRED)
enable_testing()

# include_directories(
#     ${GTEST_INCLUDE_DIRS}
#     ${CMAKE_CURRENT_SOURCE_DIR}/../operation_implement
# )

# link_directories(
#     ${CMAKE_PREFIX_PATH}/lib
# )

add_executable(example_op_test example_op_test.cpp)

target_link_libraries(example_op_test
  PRIVATE
    atb_customize
    atb
    pthread
    GTest::gtest_main
)

add_test(NAME example_op_test COMMAND example_op_test)