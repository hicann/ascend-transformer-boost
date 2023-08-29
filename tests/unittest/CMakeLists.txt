file(GLOB_RECURSE SOURCE_FILES "${CMAKE_CURRENT_LIST_DIR}/*.cpp")

include_directories(
    ${PROJECT_SOURCE_DIR}/3rdparty/half/include
)
set(CMAKE_FIND_LIBRARY_SUFFIXES .a ${CMAKE_FIND_LIBRARY_SUFFIXES})
add_executable(acltransformer_unittest ${SOURCE_FILES})
target_link_libraries(acltransformer_unittest PRIVATE acltransformer asdops gtest gmock pthread dl torch c10 torch_cpu torch_python torch_npu ascendcl)
if(NOT ${LINK_PYTHON} STREQUAL "")
    target_link_libraries(acltransformer_unittest PRIVATE ${LINK_PYTHON})
endif()
install(TARGETS acltransformer_unittest DESTINATION bin)
install(FILES ${PROJECT_SOURCE_DIR}/3rdparty/googletest/lib/libgtest.so.1.13.0
        DESTINATION lib)
install(FILES ${PROJECT_SOURCE_DIR}/3rdparty/googletest/lib/libgmock.so.1.13.0
        DESTINATION lib)
