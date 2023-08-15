#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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

SCRIPT_DIR=$(cd $(dirname $0); pwd)
CURRENT_DIR=$(pwd)
cd $SCRIPT_DIR
cd ..
export CODE_ROOT=`pwd`
export CACHE_DIR=$CODE_ROOT/build
DEVICE_ID=0
BUILD_OPTION_LIST="pull_image run_container"
BUILD_CONFIGURE_LIST=("--container_name=*" "--devid=*")


function fn_pull_image()
{
    DOCKER_JSON=`docker inspect acltransformer:v1`
    if [ $? == "0" ];then
        echo "chatglm & llama image has already existed."
        return 0
    fi
    cd $CACHE_DIR
    if [ ! -d "$CACHE_DIR/docker" ];then
        mkdir $CACHE_DIR/docker
    fi
    cd docker
    wget https://ascend-transformer-acceleration.obs.cn-north-4.myhuaweicloud.com/docker_file/acltransformer.tar --no-check-certificate
    docker load -i acltransformer.tar
    rm -f acltransformer.tar
}

function fn_run_container()
{
    if [ -z $CONTAINER_NAME ];then
        CONTAINER_NAME="acltransformer"
    fi
    if [ -z $DEVICE_ID ];then
        DEVICE_ID="0"
    fi
    docker run --name $CONTAINER_NAME -it -d --net=host --shm-size=500g --device=/dev/davinci$DEVICE_ID \
        --privileged=true \
        -w /home \
        --device=/dev/davinci_manager \
        --device=/dev/hisi_hdc \
        --device=/dev/devmm_svm \
        --entrypoint=bash \
        -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
        -v /usr/local/dcmi:/usr/local/dcmi \
        -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
        -v /usr/local/sbin/:/usr/local/sbin \
        -v /home:/home \
        -v /usr/local/asdops:/usr/local/asdops \
        -v /data/acltransformer_testdata:/data/acltransformer_testdata \
        acltransformer:v1
}

function fn_main()
{
    if [ ! -d "$CACHE_DIR" ];then
        mkdir $CACHE_DIR
    fi

    if [[ "$BUILD_OPTION_LIST" =~ "$1" ]];then
        if [[ -z "$1" ]];then
            echo "need option param!"
        else
            arg1=$1
            shift
        fi
    else
        echo "option param is unknown!"
        exit -1
    fi

    until [[ -z "$1" ]]
    do {
        arg2=$1
        case "${arg2}" in
            --container_name=*)
                CONTAINER_NAME=${arg2#*=}
                echo "CONTAINER_NAME=$CONTAINER_NAME"
                ;;
            --devid=*)
                DEVICE_ID=${arg2#*=}
                echo "DEVICE_ID=$DEVICE_ID"
                ;;
        esac
        shift
    }
    done

    case "${arg1}" in
        "pull_image")
            fn_pull_image
            ;;
        "run_container")
            fn_run_container
            ;;
    esac
}

fn_main "$@"