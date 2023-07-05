add_subdirectory(utils)
add_subdirectory(base)
add_subdirectory(context)
add_subdirectory(runner)
if(USE_TORCH_RUNNER)
    add_subdirectory(torch)
endif()
add_subdirectory(ops)


add_library(acltransformer_static STATIC ${SOURCE_FILES})

if(USE_TORCH_RUNNER)
    target_link_libraries(acltransformer_static PRIVATE
    core_utils core_base core_runner core_context core_ops
    dl asdops_static core_torch torch c10 torch_cpu torch_python
    torch_npu python3)
else()
    target_link_libraries(acltransformer_static PRIVATE
    core_utils core_base core_runner core_context core_ops
    dl asdops_static)
endif()

add_library(acltransformer SHARED ${SOURCE_FILES})

if(USE_TORCH_RUNNER)
    target_link_libraries(acltransformer PRIVATE
    core_utils core_base core_runner core_context core_ops
    dl asdops_static core_torch torch c10 torch_cpu
    torch_python torch_npu python3)
else()
    target_link_libraries(acltransformer PRIVATE
    core_utils core_base core_runner core_context
    core_ops dl asdops_static)
endif()

install(TARGETS acltransformer_static DESTINATION lib)
install(TARGETS acltransformer DESTINATION lib)