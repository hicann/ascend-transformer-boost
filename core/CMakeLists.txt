add_subdirectory(utils)
add_subdirectory(base)
if(USE_TORCH_RUNNER)
    add_subdirectory(torch)
endif()
add_subdirectory(ops)


add_library(acltransformer_static STATIC ${SOURCE_FILES})

target_link_libraries(acltransformer_static PRIVATE core_utils
                      core_base core_ops dl asdops_static)

if(USE_TORCH_RUNNER)
target_link_libraries(acltransformer_static PRIVATE
                      core_torch torch c10 torch_cpu
                      torch_python torch_npu python3)
endif()

add_library(acltransformer SHARED ${SOURCE_FILES})


target_link_libraries(acltransformer PRIVATE core_utils core_base
                      core_ops dl asdops_static)

if(USE_TORCH_RUNNER)
target_link_libraries(acltransformer PRIVATE core_torch torch
                      c10 torch_cpu torch_python torch_npu python3)
endif()

target_link_libraries(acltransformer PRIVATE profapi)
target_link_libraries(acltransformer_static PRIVATE profapi)

install(TARGETS acltransformer_static DESTINATION lib)
install(TARGETS acltransformer DESTINATION lib)