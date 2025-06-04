include_directories(
    $ENV{ATB_HOME_PATH}/include
    $ENV{ASCEND_HOME_PATH}/include
    $ENV{ASCEND_HOME_PATH}/include/experiment/msprof)

link_directories(
    $ENV{ATB_HOME_PATH}/lib
    $ENV{ASCEND_HOME_PATH}/lib64
    )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -D_GLIBCXX_USE_CXX11_ABI=0")
add_executable(multiStreamDemo multiStream_demo.cpp)
target_link_libraries(multiStreamDemo atb ascendcl runtime)