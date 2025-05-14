file(GLOB_RECURSE USR_COMMON_SRC "${CMAKE_CURRENT_LIST_DIR}/common/*.cpp")
file(GLOB_RECURSE USR_OPS_SRC "${CMAKE_CURRENT_LIST_DIR}ops/*/*.cpp")

add_library(atb_usrdefine SHARED
    ${USR_COMMON_SRC}
    ${USR_OPS_SRC}
)

add_library(atb_usrdefine_static STATIC
    ${USR_COMMON_SRC}
    ${USR_OPS_SRC}
)

include_directories(
    ${CMAKE_CURRENT_LIST_DIR}/include
    ${CMAKE_CURRENT_LIST_DIR}/common
)

target_link_libraries(atb_usrdefine PUBLIC dl mki asdops atb_mixops ascendcl profapi lcal hccl pthread acl_op_compiler)

install(TARGETS atb_usrdefine atb_usrdefine_static DESTINATION lib)
install(DIRECTORY include/ DESTINATION include/usrdefine)