set(sort_srcs
    ${CMAKE_CURRENT_LIST_DIR}/sort_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tiling/sort_tiling.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tiling/sort_tiling_v2.cpp
    ${CMAKE_CURRENT_LIST_DIR}/top_k/top_k_desc_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/top_k/top_k_desc_kernel_v2.cpp
)

add_operation(SortOperation "${sort_srcs}")