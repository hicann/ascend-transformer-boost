add_subdirectory(utils)
add_subdirectory(base)
add_subdirectory(context)
if(USE_TORCH_RUNNER)
    add_subdirectory(torch)
endif()
add_subdirectory(ops)


add_library(acltransformer_static STATIC ${SOURCE_FILES})

target_link_libraries(acltransformer_static PRIVATE core_utils
                      core_context core_base core_ops dl asdops_static pthread ascendcl)

if(USE_TORCH_RUNNER)
target_link_libraries(acltransformer_static PRIVATE
                      core_torch torch c10 torch_cpu
                      torch_python torch_npu python3)
endif()

if(USE_LCCL_RUNNER)
target_link_libraries(acltransformer_static PRIVATE lccl ascendcl)
endif()

if(USE_HCCL_RUNNER)
target_link_libraries(acltransformer_static PRIVATE hccl pthread)
endif()

add_library(acltransformer SHARED ${SOURCE_FILES})


target_link_libraries(acltransformer PRIVATE core_utils core_base
                      core_context core_ops dl asdops_static ascendcl)

if(USE_TORCH_RUNNER)
target_link_libraries(acltransformer PRIVATE core_torch torch
                      c10 torch_cpu torch_python torch_npu python3)
endif()

if(USE_LCCL_RUNNER)
target_link_libraries(acltransformer PRIVATE lccl ascendcl)
endif()

if(USE_HCCL_RUNNER)
target_link_libraries(acltransformer PRIVATE hccl pthread)
endif()

if(USE_TILING_STREAM)
target_link_libraries(acltransformer_static PRIVATE ascendcl)
target_link_libraries(acltransformer PRIVATE ascendcl)
endif()

target_link_libraries(acltransformer PRIVATE profapi)
target_link_libraries(acltransformer_static PRIVATE profapi)

install(TARGETS acltransformer_static DESTINATION lib)
install(TARGETS acltransformer DESTINATION lib)