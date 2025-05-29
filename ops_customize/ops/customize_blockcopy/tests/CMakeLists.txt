set(GTest_ROOT "${PROJECT_SOURCE_DIR}/3rdparty/googletest")
find_package(GTest 1.13.0 EXACT REQUIRED
             HINTS "${GTest_ROOT}/lib/cmake/GTest"
             PATHS "${GTest_ROOT}" NO_DEFAULT_PATH)
enable_testing()

# include_directories(
#     ${GTEST_INCLUDE_DIRS}
#     ${CMAKE_CURRENT_SOURCE_DIR}/../operation_implement
# )

# link_directories(
#     ${CMAKE_PREFIX_PATH}/lib
#     ${CAMKE_SOURCE_DIR}/src/kernels/tbe
# )

add_executable(customize_blockcopy_test customize_blockcopy_test.cpp)

target_link_libraries(customize_blockcopy_test
  PRIVATE
    atb_customize
    atb
    pthread
    GTest::gtest_main
)

add_test(NAME customize_blockcopy_test COMMAND customize_blockcopy_test)