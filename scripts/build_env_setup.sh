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

INSTALL_DIR=/usr/local

function fn_setup_lcov()
{
    if [ -f "lcov-v1.16.tar.gz" ];then
        tar -xvf lcov-v1.16.tar.gz
    else
        wget --no-check-certificate https://github.com/linux-test-project/lcov/archive/refs/tags/v1.16.tar.gz
        tar -xvf v1.16.tar.gz
    fi 
    if[ `cat /etc/redhat-release`== "Ubuntu" ];then
        apt-get install perl-Digest-MD5
        apt-get install perl*
        apt-get install cpan
    else
        yum install perl-Digest-MD5
        yum install perl*
        yum install cpan

    cd lcov-1.16
    make -j
    make PREFIX=$INSTALL_DIR/lcov install
}

function fn_setup_doxygen()
{
    if [ -f "doxygen.tar.gz" ];then
        tar -xvf doxygen.tar.gz
    else
        wget --no-check-certificate https://github.com/doxygen/doxygen/archive/refs/heads/master.tar.gz
        tar -xvf master.tar.gz
    fi
    if[ `cat /etc/redhat-release`== "Ubuntu" ];then
        apt-get install flex
        apt-get install bison
    else
        yum install flex
        yum install bison


    cd doxygen-master
    mkdir build
    cd build
    cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR/doxygen ../
    make -j
    make install
}

function fn_main()
{
    fn_setup_lcov
    fn_setup_doxygen
}

fn_main "$@"