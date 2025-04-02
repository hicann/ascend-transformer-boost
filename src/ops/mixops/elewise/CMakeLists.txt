set(elewise_srcs
    ${CMAKE_CURRENT_LIST_DIR}/cast/cast_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/elewise_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/dynamic_quant/dynamic_quant_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/dynamic_quant/dynamic_quant_tiling/dynamic_quant_tiling.cpp
    ${CMAKE_CURRENT_LIST_DIR}/simple_broadcast/tiling/simple_broadcast_tiling.cpp
    ${CMAKE_CURRENT_LIST_DIR}/simple_broadcast/simple_broadcast_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/cast/tiling/cast_tiling.cpp
)

add_operation(ElewiseOperation "${elewise_srcs}")

add_kernel(dynamic_quant ascend910b vector
    dynamic_quant/op_kernel/dynamic_quant.cpp
    DynamicQuantKernel)

add_kernel(dynamic_quant ascend310p vector
    dynamic_quant/op_kernel/dynamic_quant.cpp
    DynamicQuantKernel)

add_kernel(cast_wide ascend910b vector
    cast/op_kernel/cast_wide_nd.cpp
    CastWideKernel)

add_kernel(cast_wide ascend310p vector
    cast/op_kernel/cast_wide_nd.cpp
    CastWideKernel)

add_kernel(cast_wide ascend910 vector
    cast/op_kernel/cast_wide_nd.cpp
    CastWideKernel)

add_kernel(quant_per_channel ascend910b vector
    simple_broadcast/kernel/quant_per_channel.cpp
    QuantPerChannelKernel)

add_kernel(dequant_per_channel ascend910b vector
    simple_broadcast/kernel/dequant_per_channel.cpp
    DequantPerChannelKernel)

add_kernel(quant_per_channel ascend310p vector
    simple_broadcast/kernel/quant_per_channel.cpp
    QuantPerChannelKernel)

add_kernel(quant_per_channel ascend910 vector
    simple_broadcast/kernel/quant_per_channel.cpp
    QuantPerChannelKernel)

add_kernel(quant_per_channel ascend310b vector
    simple_broadcast/kernel/quant_per_channel.cpp
    QuantPerChannelKernel)
