SCRIPT_DIR=$(cd $(dirname $0); pwd)
CURRENT_DIR=$(pwd)
TESTS_DIR=$(cd $SCRIPT_DIR/../; pwd)

BUILD_OPTION_LIST="all performance help"

function fn_pre_check()
{
    if [ -z $ASCEND_HOME_PATH ];then
        echo "env ASCEND_HOME_PATH not exists, fail"
        exit -1
    fi
}

function fn_performance_test()
{
    PERFORMANCE_TEST_DIR=$TESTS_DIR/performancetest
    rm -rf $PERFORMANCE_TEST_DIR/results
    cd $PERFORMANCE_TEST_DIR
    python3 model_test.py
    for file in $(ls $PERFORMANCE_TEST_DIR);
    do
        if [[ $file != "model_test.py" ]];then
            python3 $file
        fi
    done
}

function fn_main()
{
    fn_pre_check

    if [[ "$BUILD_OPTION_LIST" =~ "$1" ]];then
        if [[ -z "$1" ]];then
            arg1="all"
        else
            arg1=$1
        fi
    else
        echo "argument $1 is unknown, please type dailytest.sh help for more imformation"
        exit -1
    fi

    case "${arg1}" in
        "all")
            fn_performance_test
            ;;
        "performance")
            fn_performance_test
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