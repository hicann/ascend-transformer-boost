file(GLOB_RECURSE SOURCE_FILES "${CMAKE_CURRENT_LIST_DIR}/*.cpp")
add_library(acltransformer_models SHARED ${SOURCE_FILES})
target_link_libraries(acltransformer_models PRIVATE
    acltransformer)
install(TARGETS acltransformer_models DESTINATION lib)