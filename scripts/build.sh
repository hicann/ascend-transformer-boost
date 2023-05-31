#!/bin/bash
# Copyright(C) 2023. Huawei Technologies Co.,Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e
SCRIPT_DIR=$(cd $(dirname $0); pwd)
CURRENT_DIR=$(pwd)
cd $SCRIPT_DIR
cd ..
export CODE_ROOT=`pwd`
export CACHE_DIR=$CODE_ROOT/build
export OUTPUT_DIR=$CODE_ROOT/output
THIRD_PARTY_DIR=$CODE_ROOT/3rdparty
COMPILE_OPTIONS=""
INCREMENTAL_SWITCH=OFF
HOST_CODE_PACK_SWITCH=ON
DEVICE_CODE_PACK_SWITCH=ON
BUILD_OPTION_LIST="3rdparty unittest unittest_and_run pythontest pythontest_and_run debug release help examples"
BUILD_CONFIGURE_LIST=("--output=.*" "--cache=.*" "--incremental" "--gcov" "--no_hostbin" "--no_devicebin" "--use_cxx11_abi=0" "--use_cxx11_abi=1" "--build_config=.*")

function fn_build_googltest()
{
    cd $CACHE_DIR
    rm -rf v1.13.0.tar.gz
    if [ -f "$CODE_ROOT/3rdparty/googletest-1.13.0.tar.gz" ];then
        cp $CODE_ROOT/3rdparty/googletest-1.13.0.tar.gz ./v1.13.0.tar.gz
    else
        wget --no-check-certificate https://github.com/google/googletest/archive/refs/tags/v1.13.0.tar.gz
    fi

    tar -xvf v1.13.0.tar.gz
    cd googletest-1.13.0
    mkdir build
    cd build
    cmake -DCMAKE_INSTALL_PREFIX=$THIRD_PARTY_DIR/googletest ../
    make -j
    make install
    cd ../../
}

function fn_build_half()
{
    cd $CACHE_DIR
    rm -rf half-2.2.0.zip
    if [ -f "$CODE_ROOT/3rdparty/half-2.2.0.zip" ];then
        cp $CODE_ROOT/3rdparty/half-2.2.0.zip ./
    else
        wget --no-check-certificate https://netcologne.dl.sourceforge.net/project/half/half/2.2.0/half-2.2.0.zip
    fi

    unzip half-2.2.0.zip
    mkdir -p $THIRD_PARTY_DIR/half/include
    cp include/half.hpp $THIRD_PARTY_DIR/half/include
}

function fn_build_jsoncpp()
{
    cd $CACHE_DIR
    rm -rf jsoncpp*
    if [ -f "$CODE_ROOT/3rdparty/jsoncpp-1.9.5.tar.gz" ];then
        cp $CODE_ROOT/3rdparty/jsoncpp-1.9.5.tar.gz ./
        tar -xvf jsoncpp-1.9.5.tar.gz
    else
        wget --no-check-certificate https://codeload.github.com/open-source-parsers/jsoncpp/tar.gz/refs/tags/1.9.5
        tar -xvf 1.9.5
    fi
    cd jsoncpp-1.9.5
    mkdir build
    cd build
    cmake -DCMAKE_INSTALL_PREFIX=$THIRD_PARTY_DIR/jsoncpp ../
    make -j
    make install
    cd ../../
    if [ -d "$THIRD_PARTY_DIR/jsoncpp/lib64" ]; then
        if [ -d "$THIRD_PARTY_DIR/jsoncpp/lib" ]; then
            rm -rf $THIRD_PARTY_DIR/jsoncpp/lib64
        else
            mv $THIRD_PARTY_DIR/jsoncpp/lib64 $THIRD_PARTY_DIR/jsoncpp/lib
        fi
    fi
    if [ -f "$THIRD_PARTY_DIR/jsoncpp/lib/libjsoncpp.a" ]; then
        mv $THIRD_PARTY_DIR/jsoncpp/lib/libjsoncpp.a $THIRD_PARTY_DIR/jsoncpp/lib/libjsoncpp_static.a
    fi
}

function fn_build_asdops()
{
    git clone https://gitee.com/ascend/ascend-op-common-lib.git
    cd ascend-op-common-lib
    ASD_OPP_PATH=/usr/local/Ascend/ascend-toolkit/6.0.2/opp
    if [ -d "$ASD_OPP_PATH/built_in/op_impl/ai_core/tbe/kernel" ]; then
        bash scripts/build.sh release --ascend_opp_path=$ASD_OPP_PATH/6.0.2/opp --output=$THIRD_PARTY_DIR --build_config=$CODE_ROOT/scripts/asdops_build_config.json 
    else
        bash scripts/build.sh release --output=$THIRD_PARTY_DIR --build_config=$CODE_ROOT/scripts/asdops_build_config.json 
    fi
    cd $THIRD_PARTY_DIR
    tar -xvf asdops.tar.gz
    cd ..
}

function fn_build_hacl()
{
    echo "dd"
}

function fn_build_3rdparty()
{
    rm -rf $CACHE_DIR
    mkdir $CACHE_DIR
    cd $CACHE_DIR
    fn_build_googltest
    fn_build_jsoncpp
    fn_build_asdops
    fn_build_hacl
    cd ..
}

function fn_init_pytorch_env()
{
    export PYTHON_INCLUDE_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
    export PYTHON_LIB_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
    export PYTORCH_INSTALL_PATH="$(python3 -c 'import torch, os; print(os.path.dirname(os.path.abspath(torch.__file__)))')"
    if [ -z "${PYTORCH_NPU_INSTALL_PATH}" ];then
        export PYTORCH_NPU_INSTALL_PATH="$(python3 -c 'import torch, torch_npu, os; print(os.path.dirname(os.path.abspath(torch_npu.__file__)))')"
    fi
    echo "PYTHON_INCLUDE_PATH=$PYTHON_INCLUDE_PATH"
    echo "PYTHON_LIB_PATH=$PYTHON_LIB_PATH"
    echo "PYTORCH_INSTALL_PATH=$PYTORCH_INSTALL_PATH"
    echo "PYTORCH_NPU_INSTALL_PATH=$PYTORCH_NPU_INSTALL_PATH"
}

function fn_build()
{
    if [ ! -d "$OUTPUT_DIR" ];then
        mkdir -p $OUTPUT_DIR
    fi
    if [ "$INCREMENTAL_SWITCH" == "OFF" ];then
        rm -rf $CACHE_DIR
    fi
    if [ ! -d "$CACHE_DIR" ];then
        mkdir $CACHE_DIR
    fi
    cd $CACHE_DIR
    COMPILE_OPTIONS="${COMPILE_OPTIONS} -DCMAKE_INSTALL_PREFIX=$OUTPUT_DIR/acltransformer"
    echo "COMPILE_OPTIONS:$COMPILE_OPTIONS"
    cmake $CODE_ROOT $COMPILE_OPTIONS
    if [ "$INCREMENTAL_SWITCH" == "OFF" ];then
        make clean
    fi
    make -j VERBOS=1
    make install
    chmod +x $OUTPUT_DIR/acltransformer/bin/*
    fn_generate_doxygen
}

function fn_run_unittest()
{
    cd $OUTPUT_DIR/acltransformer
    source set_env.sh
    echo "run $OUTPUT_DIR/acltransformer/bin/acltransformer_unittest"
    $OUTPUT_DIR/acltransformer/bin/acltransformer_unittest
}

function fn_build_coverage()
{
    GCOV_DIR=$OUTPUT_DIR/acltransformer/gcov
    GCOV_CACHE_DIR=$OUTPUT_DIR/acltransformer/gcov/cache
    GCOV_INFO_DIR=$OUTPUT_DIR/acltransformer/gcov/cov_info
    UNIT_TEST_DIR=$CODE_ROOT/tests/unittest
    LCOV_PATH=/usr/local/lcov/bin/lcov
    FIND_IGNORE_PATH=$CACHE_DIR/core/CMakeFiles/acltransformer_static.dir/*

    rm -rf $GCOV_DIR
    mkdir $GCOV_DIR
    mkdir $GCOV_CACHE_DIR
    mkdir $GCOV_INFO_DIR

    $LCOV_PATH -d $GCOV_CACHE_DIR --zerocounters >> $GCOV_DIR/log.txt

    find $CACHE_DIR -not -path "$FIND_IGNORE_PATH" -name "*.gcno" | xargs -i cp {} $GCOV_CACHE_DIR
    find $UNIT_TEST_DIR -name "*.cpp" | xargs -i cp {} $GCOV_CACHE_DIR
    $LCOV_PATH -c -i -d $GCOV_CACHE_DIR -o $GCOV_INFO_DIR/init.info >> $GCOV_DIR/log.txt

    fn_run_unittest

    find $CACHE_DIR -name "*.gcda" | xargs -i cp {} $GCOV_CACHE_DIR
    cd $GCOV_CACHE_DIR
    find . -name "*.cpp" | xargs -i gcov {} >> $GCOV_DIR/log.txt
    cd ..
    $LCOV_PATH -c -d $GCOV_CACHE_DIR -o $GCOV_INFO_DIR/cover.info >> $GCOV_DIR/log.txt
    $LCOV_PATH -a $GCOV_INFO_DIR/init.info -a $GCOV_INFO_DIR/cover.info -o $GCOV_INFO_DIR/total.info >> $GCOV_DIR/log.txt
    $LCOV_PATH --remove $GCOV_INFO_DIR/total.info '*/3rdparty/*' '*torch/*' '*c10/*' '*ATen/*' '*/c++/7*' '*tests/*' '*tools/*' '/usr/*' -o $GCOV_INFO_DIR/final.info >> $GCOV_DIR/log.txt
    /usr/local/lcov/bin/genhtml -o cover_result $GCOV_INFO_DIR/final.info >> $GCOV_DIR/log.txt
    tail -n 4 $GCOV_DIR/log.txt
    cd $OUTPUT_DIR/acltransformer
    tar -czf gcov.tar.gz gcov
    rm -rf gcov
}

function fn_generate_doxygen()
{
    rm -rf $OUTPUT_DIR/acltransformer/doc
    mkdir $OUTPUT_DIR/acltransformer/doc

    cd $CODE_ROOT
    if [[ $OUTPUT_DIR != $CODE_ROOT/output ]];then
        sed -i 's|OUTPUT_DIRECTORY       =.\/output\/acltransformer\/doc|OUTPUT_DIRECTORY       ='"$OUTPUT_DIR"'\/acltransformer\/doc|g' $CODE_ROOT/Doxyfile
    fi
    /usr/local/doxygen/bin/doxygen $CODE_ROOT/Doxyfile >/dev/null 2>&1
}

function fn_build_torch_extension()
{
    cd $CODE_ROOT/torch_extension
    rm -rf cache
    rm -rf dist
    rm -rf build
    rm -rf *egg*
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CODE_ROOT/3rdparty/jsoncpp/lib
    cd $OUTPUT_DIR/acltransformer
    source set_env.sh
    cd $CODE_ROOT/torch_extension
    python3 op_desc_json_auto.py
    python3 setup.py install
}

function fn_run_pythontest()
{
    cd $CODE_ROOT/scripts
    rm -rf ./kernel_meta*
    rm -rf __pycache__
    cd $CODE_ROOT/tests/pythontest/
    bash pythontest.sh
}

function fn_make_tar_package()
{
    cd $OUTPUT_DIR
    tar -czf acltransformer.tar.gz acltransformer
    rm -rf acltransformer
}

function fn_main()
{
    if [ -z $ASCEND_HOME_PATH ];then
        echo "env ASCEND_HOME_PATH not exists, build fail"
        exit -1
    fi

    PYTORCH_VERSION="$(python3 -c 'import torch; print(torch.__version__)')"
    if [ ${PYTORCH_VERSION:0:5} == "1.8.0" ];then
        COMPILE_OPTIONS="${COMPILE_OPTIONS} -DTORCH_18=ON"
    fi

    if [[ "$BUILD_OPTION_LIST" =~ "$1" ]];then
        if [[ -z "$1" ]];then
            arg1="release"
        else
            arg1=$1
            shift
        fi
    else
        cfg_flag=0
        for item in ${BUILD_CONFIGURE_LIST[*]};do
            if [[ $1 =~ $item ]];then
                cfg_flag=1
                break 1
            fi
        done
        if [[ $cfg_flag == 1 ]];then
            arg1="release"
        else
            echo "argument $1 is unknown, please type build.sh help for more imformation"
            exit -1
        fi
    fi

    until [[ -z "$1" ]]
    do {
        arg2=$1
        case "${arg2}" in
        --output=*)
            arg2=${arg2#*=}
            if [ -z $arg2 ];then
                echo "the output directory is not set. This should be set like --output=<outputDir>"
            else
                cd $CURRENT_DIR
                if [ ! -d "$arg2" ];then
                    mkdir -p $arg2
                fi
                export OUTPUT_DIR=$(cd $arg2; pwd)
            fi
            ;;
        --cache=*)
            arg2=${arg2#*=}
            if [ -z $arg2 ];then
                echo "the cache directory is not set. This should be set like --cache=<cacheDir>"
            else
                cd $CURRENT_DIR
                if [ ! -d "$arg2" ];then
                    mkdir -p $arg2
                fi
                export CACHE_DIR=$(cd $arg2; pwd)
            fi
            ;;
        "--use_cxx11_abi=1")
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -D_GLIBCXX_USE_CXX11_ABI=1"
            ;;
        "--use_cxx11_abi=0")
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -D_GLIBCXX_USE_CXX11_ABI=0"
            ;;
        "--gcov")
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DUSE_GCOV=ON"
            ;;
        "--incremental")
            INCREMENTAL_SWITCH=ON
            ;;
        "--no_hostbin")
            HOST_CODE_PACK_SWITCH=OFF
            ;;
        "--no_devicebin")
            DEVICE_CODE_PACK_SWITCH=OFF
            ;;
        --link_python=*)
            arg2=${arg2#*=}
            if [ -z $arg2 ];then
                echo "the python version is not set. This should be set like --link_python=python3.7|python3.8|python3.9"
            else
                COMPILE_OPTIONS="${COMPILE_OPTIONS} -DLINK_PYTHON=$arg2"
            fi
            ;;
        --build_config=*)
            arg2=${arg2#*=}
            if [ -z $arg2 ];then
                echo "the config file is not set. This should be set like --build_config=<configFilePath>"
            else
                firstChar=${arg2: 0: 1}
                if [[ "$firstChar" == "/" ]];then
                    export BUILD_CONFIG_FILE=$arg2
                else
                    export BUILD_CONFIG_FILE=$CURRENT_DIR"/"$arg2
                fi
            fi
            ;;
        esac
        shift
    }
    done

    fn_init_pytorch_env
    case "${arg1}" in
        "3rdparty")
            fn_build_3rdparty
            ;;
        "examples")
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DUSE_EXAMPLES=ON"
            fn_build
            ;;
        "unittest")
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DUSE_UT_TEST=ON"
            fn_build
            ;;
        "unittest_and_run")
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DUSE_UT_TEST=ON"
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DUSE_GCOV=ON"
            fn_build
            fn_build_coverage
            ;;
        "pythontest")
            fn_build
            fn_build_torch_extension
            ;;
        "pythontest_and_run")
            fn_build
            fn_build_torch_extension
            fn_run_pythontest
            ;;
        "debug")
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DCMAKE_BUILD_TYPE=Debug"
            fn_build
            fn_make_tar_package
            ;;
        "release")
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DCMAKE_BUILD_TYPE=Release"
            fn_build
            fn_make_tar_package
            ;;
        "help")
            echo "build.sh 3rdparty|unittest|unittest_and_run|pythontest|pythontest_and_run|debug|release --incremental|--gcov|--no_hostbin|--no_devicebin|--output=<dir>|--cache=<dir>|--use_cxx11_abi=0|--use_cxx11_abi=1|--build_config=<path>"
            ;;
        *)
            echo "unknown build type:${arg1}";
            ;;
    esac
}

fn_main "$@"
