set(add_src
    ${CMAKE_CURRENT_LIST_DIR}/op_kernel/example.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tiling/example_tiling.cpp
)

add_operation(AddOperation "${add_src}")