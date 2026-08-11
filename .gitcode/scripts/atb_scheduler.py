import logging
import os
import re
import argparse
import importlib.util

from atb_common import (
    deepseek_op_list,
    deepseek_op_dict,
    torch_atb_op_test_dict,
    SINGLE_GROUP_DEVICES,
    capitalize_first_letter,
    read_diff_file_list,
    get_op_from_header_file,
    build_ops_dict,
    run_test_cmd,
    shell_source,
    install_torch_atb,
    run_group_serial,
    run_two_groups,
)


def is_aarch64():
    import platform

    return platform.machine() == 'aarch64'


def analyze_diff(args):
    diff_file_list = read_diff_file_list(args)
    run_all_test = False
    run_all_torch_atb_test = False
    run_all_csvtest = False
    run_all_pythontest = False
    no_need_run_test = True
    run_mixops_pythontest = False
    run_unit_test = False

    ops = set()
    torch_atb_ops_to_test = set()
    op_dict = build_ops_dict()
    diff_python_layers_test_files = []
    diff_torch_atb_op_test_files = []
    diff_torch_atb_graph_test_files = []

    if diff_file_list:
        diff_file_list = diff_file_list.split('\n')
        for file in diff_file_list:
            if (
                file.startswith("include/atb/infer_op_params.h")
                or file.startswith("include/atb/train_op_params.h")
                or file.startswith("src/include/asdops")
                or file.startswith("src/include/atbops")
            ):
                ops = ops.union(get_op_from_header_file(file))
                no_need_run_test = False
                continue
            elif (
                re.search(r'/param_to_json\.\w+', file)
                or re.search(r'/param_compare\.\w+', file)
                or re.search(r'/runner_type\.\w+', file)
                or re.search(r'/testcase_example.csv', file)
                or re.search(r'\.md$', file)
            ):
                continue
            elif file.startswith("src/kernels/include/lcal") or file.startswith("comm/lcal"):
                no_need_run_test = False
                run_all_test = False
                ops.add("allgather")
                ops.add("allreduce")
                ops.add("broadcast")
                ops.add("reducescatter")
                ops.add("linearparallel")
                ops.add("alltoallv")
                ops.add("alltoallvv2")
                ops.add("alltoall")
            elif (
                file.startswith("include/atb/")
                or file.startswith("scripts/set_env")
                or file.startswith("src/atb/")
                or file.startswith("src/include/atb/")
                or file.startswith("src/kernels/tbe_adapter/")
                or file.startswith("src/kernels/CMakeLists.txt")
            ):
                run_all_test = True
                no_need_run_test = False
                break
            elif file.startswith("src/ops/ops_infer") or file.startswith("ops/src/ops_train"):
                op = file.split("/")[3].replace('_', '').lower()
                if op not in ops:
                    ops.add(op)
                    logging.info(f"ops from src/ops/ops_infer/ or src/ops/ops_train/ files are: {op}")
                if op in torch_atb_op_test_dict:
                    torch_atb_ops_to_test.add(op)
                    logging.info(f"op {op} has torch_atb binding, will run torch_atb test")
                no_need_run_test = False
                continue
            elif file.startswith("src/torch_atb"):
                run_all_torch_atb_test = True
                no_need_run_test = False
                continue
            elif file.startswith("tests/unittest/"):
                no_need_run_test = False
                continue
            elif file.startswith("tests/cinterface/"):
                no_need_run_test = False
                run_unit_test = True
                continue
            elif file.startswith("tests/framework/python/CsvOpsTestTool/"):
                if file.endswith("data_generation.py"):
                    continue
                else:
                    run_all_csvtest = True
                    no_need_run_test = False
                    continue
            elif file.startswith("tests/apitest/opstest/csv/"):
                op = file.split('/')[4].replace('_', '').split('.')[0].lower()
                raw_op_name = file.split('/')[4]
                if op not in op_dict.keys():
                    logging.error(f"file name {raw_op_name} should be the same as corresponding op folder name")
                    exit(1)
                if op not in ops:
                    ops.add(op)
                    logging.info(f"ops from tests/apitest/opstest/csv are: {op_dict[op]}")
                no_need_run_test = False
                continue
            elif file.startswith("tests/apitest/opstest/python/operations/"):
                if file.endswith('operation_test.py'):
                    run_all_pythontest = True
                    no_need_run_test = False
                    continue
                op = file.split('/')[5].replace('_', '').lower()
                raw_op_name = file.split('/')[5]
                if op not in op_dict.keys():
                    logging.error(f"folder name {raw_op_name} should be the same as corresponding op folder name")
                    exit(1)
                if op not in ops:
                    ops.add(op)
                    logging.info(f"ops from python/operations are: {op_dict[op]}")
                no_need_run_test = False
                continue
            elif file.startswith("tests/apitest/opstest/python/layers/"):
                if file.split('/')[5] not in diff_python_layers_test_files and file.split('/')[5] != 'chatglm6b':
                    diff_python_layers_test_files.append(file)
                    logging.info(f"ops from python/layers are: {file.split('/')[5]}")
                no_need_run_test = False
                continue
            elif file.startswith("tests/apitest/torch_atb_test/op_test/"):
                diff_torch_atb_op_test_files.append(file)
                op_test_dir = file.split('/')[5]
                torch_atb_ops_to_test.add(op_test_dir)
                logging.info(f"torch atb op test dir: {op_test_dir}")
                no_need_run_test = False
                continue
            elif file.startswith("tests/apitest/torch_atb_test/graph_test/"):
                diff_torch_atb_graph_test_files.append(file)
                run_all_torch_atb_test = True
                logging.info("torch atb graph test changed, will run all torch_atb tests")
                no_need_run_test = False
                continue
            elif file.startswith("src/kernels/mixkernels/"):
                op = file.split("/")[3].replace('_', '').lower()
                if op in ["multilatentattention"]:
                    ops.add("mla")
                    logging.info(f"mixops from mixops files are: {op}")
                mixops = {"kvcache", "pagedattention", "rope", "unpadflashattention", "unpadflashattentionnz"}
                if op in mixops:
                    if op == "unpadflashattention" and "fa_common.cce" not in file:
                        ops.add("flashattention")
                        ops.add("attention")
                    elif op == "unpadflashattentionnz":
                        ops.add("flashattentionnz")
                        ops.add("attention")
                    elif op == "kvcache":
                        ops.add("KVCache")
                    else:
                        ops.add(op)
                    logging.info(f"mixops from mixops files are: {op}")
                    run_unit_test = True
                if op not in ops:
                    ops.add(op)
                    logging.info(f"mixops from mixops files are: {op}")
                if op in torch_atb_op_test_dict:
                    torch_atb_ops_to_test.add(op)
                    logging.info(f"op {op} has torch_atb binding, will run torch_atb test")
                no_need_run_test = False
                run_mixops_pythontest = True
                continue
            elif file.startswith("src/kernels/kernels/"):
                op = file.split("/")[3].replace('_', '').lower()
                if op not in ops:
                    ops.add(op)
                    logging.info(f"ops from mixops files are: {op}")
                if op in torch_atb_op_test_dict:
                    torch_atb_ops_to_test.add(op)
                    logging.info(f"op {op} has torch_atb binding, will run torch_atb test")
                run_unit_test = True
                no_need_run_test = False
                continue
            elif file.startswith("tests/unittest/kernels"):
                op = file.split("/")[3].replace('_', '').lower()
                if op not in ops and op != "mix":
                    ops.add(op)
                    logging.info(f"mixops from mixops files are: {op}")
                if op not in ops and op == "mix":
                    op = file.split("/")[3].replace('_', '').lower()
                    if op == "attention" and op == "attentionnz" and op == "pagedattention":
                        ops.add("attention")
                    else:
                        ops.add(op)
                    logging.info(f"mixops from mixops files are: {op}")
                run_unit_test = True
                no_need_run_test = False
                continue
            elif file.startswith("tests/apitest/kerneltest/mix"):
                raw_op_name = file.split('/')[4]
                op = raw_op_name[len("test_") : -len(".py")]
                op = op.replace('_', '').lower()
                if op not in ops:
                    ops.add(op)
                    logging.info(f"ops from python/operations are: {op_dict[op]}")
                run_mixops_pythontest = True
                no_need_run_test = False
                continue
            elif (
                file.startswith("tests/apitest/kerneltest/")
                and not file.startswith("tests/apitest/kerneltest/optest.sh")
                and not file.startswith("tests/apitest/kerneltest/tensor_file.py")
            ):
                op = file.split('/')[3].replace('_', '').lower()
                raw_op_name = file.split('/')[3]
                if op not in op_dict.keys() and op != "mix":
                    logging.error(f"folder name {raw_op_name} should be the same as corresponding op folder name")
                    exit(1)
                if op not in ops:
                    ops.add(op)
                    logging.info(f"ops from python/operations are: {op_dict[op]}")
                no_need_run_test = False
                continue

    logging.info("--------------------------------------------------------------------------------------------------")
    logging.info(f"ops to run tests are: {ops}")
    logging.info(f"ops to run torch_atb test are: {torch_atb_ops_to_test}")
    logging.info(f"run_all_torch_atb_test: {run_all_torch_atb_test}")
    csv_info = "\n"
    python_info = "\n"
    pythonl_info = "\n"
    torch_atb_info = "\n"
    for op in ops:
        if op in op_dict.keys():
            csv_path = f"{os.getcwd()}/tests/apitest/opstest/csv/{op_dict[op]}.csv"
            if os.path.exists(csv_path):
                csv_info += f"      {op_dict[op]}\n"
            python_path = f"{os.getcwd()}/tests/apitest/opstest/python/operations/{op_dict[op]}"
            if os.path.exists(python_path):
                python_info += f"      {op_dict[op]}\n"
    for x in diff_python_layers_test_files:
        pythonl_info += f"      {x}\n"
    for op in torch_atb_ops_to_test:
        if op in torch_atb_op_test_dict:
            torch_atb_info += f"      {op} -> {torch_atb_op_test_dict[op]}\n"
    logging.info(f"op needs to run csv test is: {csv_info}")
    logging.info(f"op needs to run python operations test is: {python_info}")
    logging.info(f"op needs to run python layers test is: {pythonl_info}")
    logging.info(f"op needs to run torch_atb test is: {torch_atb_info}")

    return {
        "run_all_test": run_all_test,
        "run_all_torch_atb_test": run_all_torch_atb_test,
        "run_all_csvtest": run_all_csvtest,
        "run_all_pythontest": run_all_pythontest,
        "no_need_run_test": no_need_run_test,
        "run_mixops_pythontest": run_mixops_pythontest,
        "run_unit_test": run_unit_test,
        "ops": ops,
        "torch_atb_ops_to_test": torch_atb_ops_to_test,
        "op_dict": op_dict,
        "diff_python_layers_test_files": diff_python_layers_test_files,
        "diff_torch_atb_graph_test_files": diff_torch_atb_graph_test_files,
    }


def build_csvtest_all_cmd():
    cmd = [
        "bash",
        f"{os.getcwd()}/scripts/build.sh",
        "csvopstest",
        "--skip_build",
        f"--csvopstest_options=-i {os.getcwd()}/tests/apitest/opstest/csv/ -o {os.getcwd()}/output/atb/csvopstest -tt Function",
    ]
    return (cmd, "all csvtest")


def build_csvtest_ops_cmd(ops, op_dict):
    cmds = []
    for op in ops:
        if op in op_dict.keys():
            csv_path = f"{os.getcwd()}/tests/apitest/opstest/csv/{op_dict[op]}.csv"
            if os.path.exists(csv_path):
                cmd = [
                    "bash",
                    f"{os.getcwd()}/scripts/build.sh",
                    "csvopstest",
                    f"--csvopstest_options= -i {csv_path}",
                    "--skip_build",
                ]
                cmds.append((cmd, f"csv test for {op_dict[op]}"))
            if op in deepseek_op_dict:
                for _, _, files in os.walk(f"{os.getcwd()}/tests/high_level_test/{deepseek_op_dict[op]}/Smoke/"):
                    for filename in files:
                        if filename.endswith('.csv'):
                            smoke_csv_path = (
                                f"{os.getcwd()}/tests/high_level_test/{deepseek_op_dict[op]}/Smoke/{filename}"
                            )
                            cmd = [
                                "bash",
                                f"{os.getcwd()}/scripts/build.sh",
                                "csvopstest",
                                f"--csvopstest_options= -i {smoke_csv_path}",
                                "--skip_build",
                            ]
                            cmds.append((cmd, f"csv smoke test for {deepseek_op_dict[op]}"))
    return cmds


def build_pythontest_all_cmd():
    cmd = ["bash", f"{os.getcwd()}/scripts/build.sh", "pythontest", "--skip_build"]
    return (cmd, "all pythontest")


def build_pythontest_ops_cmd(ops, op_dict):
    cmds = []
    for op in ops:
        if op in op_dict.keys():
            python_path = f"{os.getcwd()}/tests/apitest/opstest/python/operations/{op_dict[op]}"
            if os.path.exists(python_path):
                cmd = ["python3", "-m", "unittest", "discover", "-s", python_path, "-p", "test*.py"]
                cmds.append((cmd, f"pythontest for {op_dict[op]}"))
    return cmds


def build_torchatbtest_all_cmd():
    cmd = ["bash", f"{os.getcwd()}/scripts/build.sh", "torchatbtest", "--torch_atb_gcc_path=/usr/bin"]
    return (cmd, "all torchatbtest")


def build_torchatbtest_ops_cmd(torch_atb_ops_to_test):
    cmds = []
    for op in torch_atb_ops_to_test:
        if op in torch_atb_op_test_dict:
            test_dir = torch_atb_op_test_dict[op]
            test_path = f"{os.getcwd()}/tests/apitest/torch_atb_test/op_test/{test_dir}"
            if os.path.exists(test_path):
                cmd = ["python3", "-m", "unittest", "discover", "-s", test_path, "-p", "test*.py"]
                cmds.append((cmd, f"torch_atb test for {op}"))
    return cmds


def build_smoke_test_cmds():
    cmds = []
    for deepseek_op in deepseek_op_list:
        smoke_testcase_path = f"{os.getcwd()}/tests/high_level_test/{deepseek_op}/Smoke/"
        if os.path.exists(smoke_testcase_path):
            cmd = ["python3", "-m", "unittest", "discover", "-s", smoke_testcase_path, "-p", "test*.py"]
            cmds.append((cmd, f"smoke pythontest for {deepseek_op}"))
            for _, _, files in os.walk(smoke_testcase_path):
                for filename in files:
                    if filename.endswith('.csv'):
                        csv_path = os.path.join(smoke_testcase_path, filename)
                        cmd = [
                            "bash",
                            f"{os.getcwd()}/scripts/build.sh",
                            "csvopstest",
                            f"--csvopstest_options= -i {csv_path}",
                            "--skip_build",
                        ]
                        cmds.append((cmd, f"smoke csvtest for {deepseek_op}"))
    return cmds


def build_python_layers_cmds(diff_python_layers_test_files, op_dict, run_mixops_pythontest):
    cmds = []
    for pytest_file in diff_python_layers_test_files:
        cmd = ["python", f"{os.getcwd()}/{pytest_file}"]
        cmds.append((cmd, f"pythontest for {pytest_file}"))
    return cmds


def build_commonlib_python_cmds(ops, op_dict, run_mixops_pythontest):
    cmds = []
    kernelstest_path = os.path.abspath(f"{os.getcwd()}/tests/apitest/kernelstest")
    if not os.path.isdir(kernelstest_path):
        return cmds

    mix_path = os.path.join(kernelstest_path, "mix")
    for op in ops:
        if op not in op_dict:
            continue
        if run_mixops_pythontest and os.path.isdir(mix_path):
            for file in os.listdir(mix_path):
                if file.startswith("test_") and file.endswith(".py"):
                    op_name = file[len("test_") : -len(".py")]
                    op_name = op_name.split('.')[0].replace('_', '').lower()
                    if op in op_name:
                        cmd = ["python3", os.path.join(mix_path, file)]
                        cmds.append((cmd, f"commonlib pythontest for {op_dict[op]}", kernelstest_path))
        op_subdir = os.path.join(kernelstest_path, op_dict[op])
        if os.path.isdir(op_subdir):
            cmd = ["python3", "-m", "unittest", "discover", "-s", op_dict[op], "-p", "test*.py"]
            cmds.append((cmd, f"commonlib pythontest for {op_dict[op]}", kernelstest_path))
    return cmds


def schedule_all_test(config):
    logging.info(
        "---------------------------------------- Starting Test: All Tests (Parallel) ----------------------------------------"
    )

    cmd = ["bash", f"{os.getcwd()}/scripts/build.sh", "testframework", "--no_werror"]
    if not run_test_cmd(cmd, test_mode="build for testframework"):
        return

    if not shell_source(f"{os.getcwd()}/output/atb/set_env.sh"):
        logging.error("source output/atb/set_env.sh failed!")
        exit(1)

    install_torch_atb()

    group1_cmds = []
    group2_cmds = []

    if is_aarch64():
        group1_cmds.extend(build_smoke_test_cmds())
        group1_cmds.append(build_csvtest_all_cmd())
        group2_cmds.append(build_pythontest_all_cmd())
        group2_cmds.append(build_torchatbtest_all_cmd())
    else:
        group1_cmds.append(build_csvtest_all_cmd())
        group1_cmds.append(build_torchatbtest_all_cmd())
        group2_cmds.append(build_pythontest_all_cmd())

    run_two_groups(group1_cmds, group2_cmds)
    logging.info(
        "---------------------------------------- Run All Tests has completed! ----------------------------------------"
    )


def schedule_precise_test(config):
    ops = config["ops"]
    op_dict = config["op_dict"]

    cmd = ["bash", f"{os.getcwd()}/scripts/build.sh", "testframework", "--no_werror"]
    if not run_test_cmd(cmd, test_mode="scripts/build.sh for testframework"):
        return

    if not shell_source(f"{os.getcwd()}/output/atb/set_env.sh"):
        logging.error("source output/atb/set_env.sh failed!")
        exit(1)

    logging.info(f"ATB_HOME_PATH: {os.environ.get('ATB_HOME_PATH')}")

    need_torch_atb = config["run_all_torch_atb_test"] or config["torch_atb_ops_to_test"]
    if need_torch_atb:
        install_torch_atb()

    group1_cmds = []
    group2_cmds = []

    group1_cmds.append((["bash", f"{os.getcwd()}/scripts/build.sh", "unittest"], "unit-test"))

    if config["run_all_csvtest"]:
        group1_cmds.append(build_csvtest_all_cmd())
    elif ops:
        group1_cmds.extend(build_csvtest_ops_cmd(ops, op_dict))

    if config["run_all_torch_atb_test"]:
        group1_cmds.append(build_torchatbtest_all_cmd())
    elif config["torch_atb_ops_to_test"]:
        group1_cmds.extend(build_torchatbtest_ops_cmd(config["torch_atb_ops_to_test"]))

    if config["run_all_pythontest"]:
        group2_cmds.append(build_pythontest_all_cmd())
    elif ops:
        group2_cmds.extend(build_pythontest_ops_cmd(ops, op_dict))

    group2_cmds.extend(
        build_python_layers_cmds(config["diff_python_layers_test_files"], op_dict, config["run_mixops_pythontest"])
    )
    group2_cmds.extend(build_commonlib_python_cmds(ops, op_dict, config["run_mixops_pythontest"]))

    if group1_cmds and group2_cmds:
        run_two_groups(group1_cmds, group2_cmds)
    elif group1_cmds:
        rc, _ = run_group_serial(group1_cmds, SINGLE_GROUP_DEVICES)
        if not rc:
            exit(1)
    elif group2_cmds:
        rc, _ = run_group_serial(group2_cmds, SINGLE_GROUP_DEVICES)
        if not rc:
            exit(1)
    else:
        logging.info(
            "---------------------------------------- No test tasks to run ----------------------------------------"
        )

    if config["run_unit_test"]:
        if not ops:
            logging.info(
                "---------------------------------------- Starting Run-Common-lib-Unit-Test ----------------------------------------"
            )
            cmd = ["bash", f"{os.getcwd()}/scripts/build.sh", "kernelunittest"]
            run_test_cmd(cmd, test_mode="scripts/build.sh for kernel-unit-test")
        else:
            logging.info(
                "---------------------------------------- Starting Run-Partial-Unit-Test ----------------------------------------"
            )
            torch_spec = importlib.util.find_spec("torch")
            torch_npu_spec = importlib.util.find_spec("torch_npu")
            ld_lib_path = os.environ.get('LD_LIBRARY_PATH')
            ld_lib_path = ":".join(
                (
                    torch_npu_spec.submodule_search_locations[0] + "/lib",
                    torch_spec.submodule_search_locations[0] + "/lib",
                    ld_lib_path,
                )
            )
            os.environ.update({'LD_LIBRARY_PATH': ld_lib_path})
            for op in ops:
                capitalized_op = capitalize_first_letter(op)
                filter_pattern = f"*{capitalized_op}*"
                print(f"--gtest_filter={filter_pattern}")
                atb_home = os.environ.get("ATB_HOME_PATH")
                if not atb_home:
                    logging.error("ATB_HOME_PATH is not set after sourcing set_env.sh")
                    exit(1)
                cmd = [f"{atb_home}/bin/kernels_unittest", f"--gtest_filter={filter_pattern}"]
                run_test_cmd(cmd, test_mode=f"kernels_unittest --gtest_filter={filter_pattern}")


def fn_gen_test_script(args):
    config = analyze_diff(args)

    if config["no_need_run_test"]:
        logging.info(
            "---------------------------------------- No Need To Run Test ----------------------------------------"
        )
        exit(0)

    if config["run_all_test"]:
        schedule_all_test(config)
    else:
        schedule_precise_test(config)

    logging.info("---------------------------------------- ALL DONE ! ----------------------------------------")
    exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--branch', default='master')
    args = parser.parse_args()
    fn_gen_test_script(args)
