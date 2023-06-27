file(GLOB_RECURSE SOURCE_FILES "${CMAKE_CURRENT_LIST_DIR}/*.cpp")

add_library(acltransformer_torch SHARED ${SOURCE_FILES})
target_link_libraries(acltransformer_torch PRIVATE acltransformer asdops torch c10 torch_cpu torch_python torch_npu examples_util examples_layers examples_workspace)
install(TARGETS acltransformer_torch DESTINATION examples)