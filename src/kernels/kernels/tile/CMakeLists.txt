set(Tile_srcs
    ${CMAKE_CURRENT_LIST_DIR}/tile_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/TileKernel/tile_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tiling/tile_tiling.cpp
)

add_operation(TileOperation "${Tile_srcs}")