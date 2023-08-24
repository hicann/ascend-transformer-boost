SCRIPT_DIR=$(cd $(dirname $0); pwd)
CURRENT_DIR=$(pwd)
TESTS_DIR=$(cd $SCRIPT_DIR/../; pwd)

BUILD_OPTION_LIST="all performance precision unittest pythontest help"

function fn_pre_check()
{
    if [ -z $ASCEND_HOME_PATH ];then
        echo "env ASCEND_HOME_PATH not exists, fail"
        exit -1
    fi
    if [ -z $ACLTRANSFORMER_HOME_PATH ];then
        echo "env ACLTRANSFORMER_HOME_PATH not exists, fail"
        exit -1
    fi
}

function fn_prepare()
{
    DAILY_TEST_DIR=$TESTS_DIR/dailytest
    rm -rf $DAILY_TEST_DIR/results
    python3 $DAILY_TEST_DIR/model_test.py
}

function fn_performance_test()
{
    PERFORMANCE_TEST_DIR=$TESTS_DIR/performancetest
    cd $PERFORMANCE_TEST_DIR
    for file in $(ls $PERFORMANCE_TEST_DIR);
    do
        if [[ "$file" =~ py$ ]];then
            python3 $file
        fi
    done
}

function fn_precision_test()
{
    PRECISION_TEST_DIR=$TESTS_DIR/precisiontest 
    cd $PRECISION_TEST_DIR
    for file in $(ls $PRECISION_TEST_DIR);
    do
        if [[ "$file" =~ py$ ]];then
            python3 $file
        fi
    done
}

function fn_unit_test()
{
    export ACLTRANSFORMER_OPSRUNNER_KERNEL_CACHE_ENABLE=0
    echo "run $ACLTRANSFORMER_HOME_PATH/bin/acltransformer_unittest"
    $ACLTRANSFORMER_HOME_PATH/bin/acltransformer_unittest
}

function fn_python_test()
{
    cd $ACLTRANSFORMER_HOME_PATH/../../tests/pythontest
    rm -rf ./kernel_meta*
    bash pythontest.sh
}

function fn_main()
{
    fn_pre_check

    fn_prepare

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
            fn_precision_test
            fn_unit_test
            fn_python_test
            ;;
        "performance")
            fn_performance_test
            ;;
        "precision")
            fn_precision_test
            ;;
        "pythontest")
            fn_python_test
            ;;
        "unittest")
            fn_unit_test
            ;;
        "help")
            echo "build.sh all|performance|precision|pythontest|unittest"
            ;;
        *)
            echo "unknown build type:${arg1}";
            ;;
    esac
}

fn_main "$@"