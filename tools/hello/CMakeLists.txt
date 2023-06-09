file(GLOB_RECURSE SOURCE_FILES "${CMAKE_CURRENT_LIST_DIR}/*.cpp")
add_executable(acltransformer_hello ${SOURCE_FILES})
target_link_libraries(acltransformer_hello PRIVATE acltransformer dl)
install(TARGETS acltransformer_hello DESTINATION bin)