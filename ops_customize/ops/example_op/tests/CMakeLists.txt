cmake_minimum_required(VERSION 3.15)
project(example_op_test CXX)

find_package(GTest REQUIRED)
enable_testing()

include_directories(
    ${GTEST_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../operation_implement
)

link_directories(
    ${CMAKE_PREFIX_PATH}/lib
)

add_executable(example_op_test example_op_test.cpp)

target_link_libraries(example_op_test
    PRIVATE
      atb_customize
      GTest::GTest
      GTest::Main
      pthread
)

add_test(NAME example_op_test COMMAND example_op_test)