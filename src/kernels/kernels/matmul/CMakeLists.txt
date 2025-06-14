set(matmul_srcs
    ${CMAKE_CURRENT_LIST_DIR}/matmul_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/batch_matmul_kernel/batch_matmul_nd_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/batch_matmul_kernel/batch_matmul_nz_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/matmul_kernel/matmul_nd_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/matmul_kernel/matmul_nz_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/pp_matmul_f16_mix_kernel/pp_matmul_f16_mix_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/pp_matmul_i8_kernel/pp_matmul_i8_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/pp_matmul_f16_m300_kernel/pp_matmul_f16_m300_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/pp_matmul_i8_nz_kernel/pp_matmul_i8_nz_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/pp_matmul_i8_nz_compress_kernel/pp_matmul_i8_nz_compress_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tiling/matmul_nd_tiling.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tiling/matmul_nz_tiling.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tiling/pp_matmul_mix_tiling.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tiling/pp_matmul_tiling.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tiling/pp_matmul_i8_nz_tiling.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tiling/pp_matmul_nz_tiling.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tiling/pp_matmul_nd_tiling.cpp
    ${CMAKE_CURRENT_LIST_DIR}/pp_mat_mul_bf16nd_bf16nd_bf16nd_kernel/pp_mat_mul_bf16nd_bf16nd_bf16nd_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/pp_mat_mul_bf16nd_bf16nd_f32nd_kernel/pp_mat_mul_bf16nd_bf16nd_f32nd_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/pp_mat_mul_bf16nd_bf16nz_bf16nd_kernel/pp_mat_mul_bf16nd_bf16nz_bf16nd_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/pp_mat_mul_f16nd_f16nd_f16nd_kernel/pp_mat_mul_f16nd_f16nd_f16nd_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/pp_mat_mul_f16nd_f16nd_f32nd_kernel/pp_mat_mul_f16nd_f16nd_f32nd_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/pp_mat_mul_f16nd_f16nz_f16nd_kernel/pp_mat_mul_f16nd_f16nz_f16nd_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/pp_mat_mul_f16nz_f16nz_f16nz_kernel/pp_mat_mul_f16nz_f16nz_f16nz_kernel.cpp
)

add_operation(MatMulOperation "${matmul_srcs}")
