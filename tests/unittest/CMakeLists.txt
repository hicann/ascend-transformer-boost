file(GLOB_RECURSE SOURCE_FILES "${CMAKE_CURRENT_LIST_DIR}/*.cpp")

add_executable(acltransformer_unittest ${SOURCE_FILES})
target_link_libraries(acltransformer_unittest PRIVATE acltransformer asdops gtest gmock pthread dl torch c10 torch_cpu torch_python torch_npu ascendcl)
install(TARGETS acltransformer_unittest DESTINATION bin)