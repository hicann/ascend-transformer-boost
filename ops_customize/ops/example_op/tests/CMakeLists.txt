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
      GTest::gtest
      GTest::gtest_main
      atb
      atb_customize
      pthread
)

add_test(NAME example_op_test COMMAND example_op_test)