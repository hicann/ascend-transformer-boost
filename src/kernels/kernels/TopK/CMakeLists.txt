set(topk_srcs
    ${CMAKE_CURRENT_LIST_DIR}/topk_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tiling/topk_tiling_v2.cpp
    ${CMAKE_CURRENT_LIST_DIR}/top_k/top_k_desc_kernel_v2.cpp
)

add_operation(TopKOperation "${topk_srcs}")