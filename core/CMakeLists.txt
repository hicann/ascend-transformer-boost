file(GLOB_RECURSE SOURCE_FILES "${CMAKE_CURRENT_LIST_DIR}/*.cpp")

add_library(acltransformer_static STATIC ${SOURCE_FILES})
target_link_libraries(acltransformer_static PRIVATE dl asdops torch c10 torch_cpu torch_python torch_npu)
add_library(acltransformer SHARED ${SOURCE_FILES})
target_link_libraries(acltransformer PRIVATE dl asdops torch c10 torch_cpu torch_python torch_npu)

install(TARGETS acltransformer_static DESTINATION lib)
install(TARGETS acltransformer DESTINATION lib)