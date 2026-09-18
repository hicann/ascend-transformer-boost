import logging
import os
import re
import argparse
import importlib.util
import shutil

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
    run_legacy_two_groups,
    run_n_groups,
)


def is_aarch64():
    import platform

    return platform.machine() == 'aarch64'


COMM_OPS = {
    "all_gather",
    "all_gatherv",
    "all_reduce",
    "all_to_all",
    "all_to_allv",
    "all_to_allvv2",
    "broadcast",
    "linear_parallel",
    "recv",
    "reduce_scatter",
    "reduce_scatterv",
    "send",
}


def _split_round_robin(tasks, n):
    groups = [[] for _ in range(n)]
    for i, task in enumerate(tasks):
        groups[i % n].append(task)
    return groups


def _split_balanced(tasks, n, weight_of):
    groups = [[] for _ in range(n)]
    loads = [0] * n
    for task in sorted(tasks, key=lambda t: -weight_of(t)):
        i = loads.index(min(loads))
        groups[i].append(task)
        loads[i] += weight_of(task)
    return groups


def _split_pool_to_buckets(pool, caps):
    buckets = [[] for _ in caps]
    loads = [0.0] * len(caps)
    for weight, kind, item in sorted(pool, key=lambda x: -x[0]):
        i = min(range(len(caps)), key=lambda k: loads[k] / caps[k] if caps[k] > 0 else float("inf"))
        buckets[i].append((weight, kind, item))
        loads[i] += weight
    return buckets


DUAL_STREAM_DEVICES = "0,1"
SINGLE_STREAM_DEVICES = ["2", "3"]
USE_THREE_STREAM_SCHEDULER = True


_SMOKE_WEIGHT = {
    ("pythontest", "MultiLatentAttentionOperation"): 4.5,
    ("pythontest", "PagedAttentionOperation"): 2,
    ("pythontest", "LinearOperation"): 0.5,
    ("pythontest", "FusedAddTopkDivOperation"): 0.3,
    ("pythontest", "GatingOperation"): 0.3,
    ("pythontest", "RopeQConcatOperation"): 0.3,
}


def _smoke_task_weight(entry):
    label = entry[1]
    for (kind, op), weight in _SMOKE_WEIGHT.items():
        if f"smoke {kind} for {op}" == label:
            return weight
    return 0.2


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


def build_legacy_csvtest_all_cmd():
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


def build_legacy_pythontest_all_cmd():
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


def build_legacy_torchatbtest_all_cmd():
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


def build_legacy_smoke_test_cmds():
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


def build_smoke_pytest_cmds():
    cmds = []
    for deepseek_op in deepseek_op_list:
        smoke_testcase_path = f"{os.getcwd()}/tests/high_level_test/{deepseek_op}/Smoke/"
        if os.path.exists(smoke_testcase_path):
            cmd = ["python3", "-m", "unittest", "discover", "-s", smoke_testcase_path, "-p", "test*.py"]
            cmds.append((cmd, f"smoke pythontest for {deepseek_op}"))
    return cmds


def list_csv_files():
    csv_dir = f"{os.getcwd()}/tests/apitest/opstest/csv"
    files = []
    for filename in sorted(os.listdir(csv_dir)):
        if not filename.endswith('.csv'):
            continue
        if '_csvopstest_' in filename or 'tmp' in filename:
            continue
        op = filename[:-4]
        files.append((os.path.join(csv_dir, filename), f"csvtest for {op}", op in COMM_OPS))
    return files


def list_smoke_csv_files():
    files = []
    for deepseek_op in deepseek_op_list:
        smoke_testcase_path = f"{os.getcwd()}/tests/high_level_test/{deepseek_op}/Smoke/"
        if not os.path.exists(smoke_testcase_path):
            continue
        for _, _, fs in os.walk(smoke_testcase_path):
            for filename in sorted(fs):
                if not filename.endswith('.csv'):
                    continue
                if '_csvopstest_' in filename or 'tmp' in filename:
                    continue
                files.append((os.path.join(smoke_testcase_path, filename), f"smoke csvtest for {deepseek_op}", False))
    return files


def list_python_op_dirs():
    ops_root = f"{os.getcwd()}/tests/apitest/opstest/python/operations"
    dirs = []
    for op_dir in sorted(os.listdir(ops_root)):
        dir_path = os.path.join(ops_root, op_dir)
        if not os.path.isdir(dir_path):
            continue
        if not os.path.isfile(os.path.join(dir_path, "__init__.py")):
            continue
        dirs.append((dir_path, f"pythontest for {op_dir}", op_dir in COMM_OPS))
    return dirs


def _reset_shard_dir(root, name):
    shard_dir = os.path.join(root, name)
    shutil.rmtree(shard_dir, ignore_errors=True)
    os.makedirs(shard_dir, exist_ok=True)
    return shard_dir


def _op_dir_weight(item):
    label = item[1]
    op = label.rsplit(" for ", 1)[-1]
    return {"self_attention": 28.3, "paged_attention": 11}.get(op, 0.5)


def _single_csv_weight(item):
    label = item[1]
    op = label.rsplit(" for ", 1)[-1]
    return {
        "linear": 16.7,
        "paged_cache_load": 8.3,
        "gather": 4.0,
        "grouped_matmul_with_routing": 2.4,
        "self_attention": 1.9,
        "activation": 1.9,
        "dynamic_ntk": 1.5,
        "reshape_and_cache_omni": 1.7,
        "reshape_and_cache": 1.4,
        "rms_norm": 1.4,
        "laser_attention": 1.0,
        "gmm_deq_swiglu_quant_gmm_deq": 0.9,
        "mm_deq_swiglu_quant_mm_deq": 0.7,
        "laser_attention_grad": 0.6,
        "scatter_elements_v2": 0.6,
        "index_add": 0.4,
        "linear_sparse": 0.4,
        "paged_attention": 0.3,
        "elewise": 0.3,
        "kv_cache": 0.3,
        "relay_attention": 0.3,
        "layer_norm": 0.2,
        "rope_q_concat": 0.2,
        "cumsum": 0.2,
    }.get(op, 0.2)


def _smoke_csv_weight(item):
    name = os.path.basename(item[0])
    return {
        "LinearOperation_TestCase_DeepSeek.csv": 10,
        "PagedAttenstionOperation_TestCase.csv": 0.7,
        "LinearOperation_TestCase.csv": 0.4,
    }.get(name, 0.1)


def _copy_files_to_shards(root, shard_names, items, weight_of=None):
    if weight_of:
        groups = _split_balanced(items, len(shard_names), weight_of)
    else:
        groups = _split_round_robin(items, len(shard_names))
    dirs = []
    for name, group in zip(shard_names, groups):
        if not group:
            continue
        shard_dir = _reset_shard_dir(root, name)
        for src, _ in group:
            shutil.copy2(src, os.path.join(shard_dir, os.path.basename(src)))
        dirs.append(shard_dir)
    return dirs


def _copy_op_dirs_to_shards(root, shard_names, items, weight_of=None):
    if weight_of:
        groups = _split_balanced(items, len(shard_names), weight_of)
    else:
        groups = _split_round_robin(items, len(shard_names))
    ops_root = os.path.join(root, "operations")
    dirs = []
    for name, group in zip(shard_names, groups):
        if not group:
            continue
        shard_dir = _reset_shard_dir(root, name)
        with open(os.path.join(shard_dir, "__init__.py"), 'w', encoding='utf-8'):
            pass
        for src, _ in group:
            shutil.copytree(src, os.path.join(shard_dir, os.path.basename(src)))
        for common in ("operation_test.py", "precision_calcu.py"):
            path = os.path.join(ops_root, common)
            if os.path.isfile(path):
                shutil.copy2(path, shard_dir)
        dirs.append(shard_dir)
    return dirs


def build_torchatbtest_all_tasks():
    torch_atb_root = f"{os.getcwd()}/tests/apitest/torch_atb_test"
    apitest_root = f"{os.getcwd()}/tests/apitest"
    dual, single = [], []
    for testdir in sorted(os.listdir(torch_atb_root)):
        dir_path = os.path.join(torch_atb_root, testdir)
        if not testdir.endswith('_test') or not os.path.isdir(dir_path):
            continue
        for root, _, files in os.walk(dir_path):
            for filename in sorted(files):
                if not filename.endswith('_test.py'):
                    continue
                op_dir = os.path.basename(root)
                cmd = ["python3", "-m", "unittest", "discover", "-s", apitest_root, "-p", filename, "-v"]
                entry = (cmd, f"torchatbtest for {filename}", torch_atb_root)
                if op_dir in COMM_OPS or "multirank" in filename:
                    dual.append(entry)
                else:
                    single.append(entry)
    return dual, single


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


def _run_legacy_all_test():
    if is_aarch64():
        group1_cmds = build_legacy_smoke_test_cmds()
        group1_cmds.append(build_legacy_csvtest_all_cmd())
        group2_cmds = [build_legacy_pythontest_all_cmd(), build_legacy_torchatbtest_all_cmd()]
    else:
        group1_cmds = [build_legacy_csvtest_all_cmd(), build_legacy_torchatbtest_all_cmd()]
        group2_cmds = [build_legacy_pythontest_all_cmd()]
    run_legacy_two_groups(group1_cmds, group2_cmds)


def schedule_all_test(config):
    logging.info(
        "---------------------------------------- Starting Test: All Tests (Parallel) ----------------------------------------"
    )

    cmd = [
        "bash",
        f"{os.getcwd()}/scripts/build.sh",
        "testframework",
        "--no_werror",
        "--torch_atb",
        "--torch_atb_gcc_path=/usr/bin",
    ]
    if not run_test_cmd(cmd, test_mode="build for testframework"):
        return

    if not shell_source(f"{os.getcwd()}/output/atb/set_env.sh"):
        logging.error("source output/atb/set_env.sh failed!")
        exit(1)

    if not install_torch_atb():
        exit(1)

    if not USE_THREE_STREAM_SCHEDULER:
        _run_legacy_all_test()
        logging.info(
            "---------------------------------------- Run All Tests has completed! ----------------------------------------"
        )
        return

    smoke_py_tasks = build_smoke_pytest_cmds() if is_aarch64() else []
    csv_files = list_csv_files()
    smoke_csv_files = list_smoke_csv_files() if is_aarch64() else []
    op_dirs = list_python_op_dirs()
    ta_dual, ta_single = build_torchatbtest_all_tasks()

    csv_dual = [file for file in csv_files if file[2]]
    csv_single = [file for file in csv_files if not file[2]]
    op_dual = [item for item in op_dirs if item[2]]
    op_single = [item for item in op_dirs if not item[2]]

    logging.info(f"dual-card: pytest dirs={len(op_dual)} csv files={len(csv_dual)} torchatbtest={len(ta_dual)}")
    logging.info(
        f"single-card: smoke py={len(smoke_py_tasks)} smoke csv={len(smoke_csv_files)} "
        f"pytest dirs={len(op_single)} csv files={len(csv_single)} torchatbtest={len(ta_single)}"
    )

    opstest_root = f"{os.getcwd()}/tests/apitest/opstest"
    python_root = f"{opstest_root}/python"
    out_dir = f"{os.getcwd()}/output/atb/csvopstest"

    def _csv_shard_task(shard_dir, name):
        return (
            [
                "python3",
                f"{os.getcwd()}/tests/framework/python/CsvOpsTestTool/atb_csv_ops_test.py",
                "-i",
                shard_dir,
                "-o",
                out_dir,
                "-tt",
                "Function",
            ],
            f"csvtest for {name}",
        )

    def _ops_shard_task(shard_dir, name):
        return (
            ["python3", "-m", "unittest", "discover", "-s", name, "-p", "*test*.py", "-t", "."],
            f"pythontest for {name}",
            python_root,
        )

    dual_csv_dirs = _copy_files_to_shards(opstest_root, ["csv_dual_1"], [(path, label) for path, label, _ in csv_dual])
    dual_ops_dirs = _copy_op_dirs_to_shards(
        python_root, ["operations_dual_1"], [(path, label) for path, label, _ in op_dual]
    )
    dual_stream = []
    if dual_csv_dirs:
        dual_stream.append(_csv_shard_task(dual_csv_dirs[0], "csv_dual_1"))
    if dual_ops_dirs:
        dual_stream.append(_ops_shard_task(dual_ops_dirs[0], "operations_dual_1"))
    dual_stream.extend(ta_dual)

    def _dual_csv_dur(item):
        op = item[1].rsplit(" for ", 1)[-1]
        return {"all_reduce": 6, "broadcast": 1.6, "linear_parallel": 1.4}.get(op, 0.65)

    def _dual_ops_dur(item):
        op = item[1].rsplit(" for ", 1)[-1]
        return {"all_reduce": 1.7, "linear_parallel": 1.2, "all_to_all": 0.9}.get(op, 0.3)

    dual_weight = sum(_dual_csv_dur(file) for file in csv_dual)
    dual_weight += sum(_dual_ops_dur(item) for item in op_dual)
    dual_weight += 0.5 * len(ta_dual)

    pool = []
    for task in smoke_py_tasks:
        pool.append((_smoke_task_weight(task), "smoke_py", task))
    for file in smoke_csv_files:
        pool.append((_smoke_csv_weight(file), "smoke_csv", file))
    for file in csv_single:
        pool.append((_single_csv_weight(file), "csv", file))
    for item in op_single:
        pool.append((_op_dir_weight(item), "ops", item))
    for task in ta_single:
        pool.append((0.2, "torchatb", task))

    single_weight = sum(weight for weight, _, _ in pool)
    n_single = len(SINGLE_STREAM_DEVICES)
    comp_weight = max((single_weight - n_single * dual_weight) / (n_single + 1), 0.0)
    stream_cap = (single_weight - comp_weight) / n_single
    buckets = _split_pool_to_buckets(pool, [stream_cap] * n_single + [comp_weight])
    comp_bucket = buckets[-1]

    logging.info(
        "---------------------------------------- Phase NEW: 1 dual-card stream + %d single-card streams ----------------------------------------"
        % n_single
    )
    logging.info(
        f"weight summary: dual={dual_weight:.1f} single={single_weight:.1f} comp={comp_weight:.1f} "
        f"stream_cap={stream_cap:.1f} target_T={dual_weight + comp_weight:.1f}"
    )
    for i in range(n_single):
        logging.info(
            f"single bucket {i + 1}: weight={sum(weight for weight, _, _ in buckets[i]):.1f} items={len(buckets[i])}"
        )
    logging.info(f"comp bucket: weight={sum(weight for weight, _, _ in comp_bucket):.1f} items={len(comp_bucket)}")

    def _build_stream_tasks(bucket, idx):
        smoke_py = []
        smoke_csv = []
        csv_items = []
        ops_items = []
        torchatb = []
        for _, kind, item in bucket:
            if kind == "smoke_py":
                smoke_py.append(item)
            elif kind == "smoke_csv":
                smoke_csv.append((item[0], item[1]))
            elif kind == "csv":
                csv_items.append((item[0], item[1]))
            elif kind == "ops":
                ops_items.append((item[0], item[1]))
            else:
                torchatb.append(item)

        tasks = list(smoke_py)
        if smoke_csv:
            dirs = _copy_files_to_shards(opstest_root, [f"smoke_csv_{idx}"], smoke_csv)
            tasks.append(_csv_shard_task(dirs[0], f"smoke_csv_{idx}"))
        if ops_items:
            dirs = _copy_op_dirs_to_shards(python_root, [f"operations_{idx}"], ops_items)
            tasks.append(_ops_shard_task(dirs[0], f"operations_{idx}"))
        if csv_items:
            dirs = _copy_files_to_shards(opstest_root, [f"csv_{idx}"], csv_items)
            tasks.append(_csv_shard_task(dirs[0], f"csv_{idx}"))
        tasks.extend(torchatb)
        return tasks

    dual_stream.extend(_build_stream_tasks(comp_bucket, 5))
    streams = [(dual_stream, DUAL_STREAM_DEVICES, "dual-ab")]
    for i, device in enumerate(SINGLE_STREAM_DEVICES):
        streams.append((_build_stream_tasks(buckets[i], i + 1), device, f"single-{device}"))

    run_n_groups(streams)
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
        group1_cmds.append(build_legacy_csvtest_all_cmd())
    elif ops:
        group1_cmds.extend(build_csvtest_ops_cmd(ops, op_dict))

    if config["run_all_torch_atb_test"]:
        group1_cmds.append(build_legacy_torchatbtest_all_cmd())
    elif config["torch_atb_ops_to_test"]:
        group1_cmds.extend(build_torchatbtest_ops_cmd(config["torch_atb_ops_to_test"]))

    if config["run_all_pythontest"]:
        group2_cmds.append(build_legacy_pythontest_all_cmd())
    elif ops:
        group2_cmds.extend(build_pythontest_ops_cmd(ops, op_dict))

    group2_cmds.extend(
        build_python_layers_cmds(config["diff_python_layers_test_files"], op_dict, config["run_mixops_pythontest"])
    )
    group2_cmds.extend(build_commonlib_python_cmds(ops, op_dict, config["run_mixops_pythontest"]))

    if group1_cmds and group2_cmds:
        run_legacy_two_groups(group1_cmds, group2_cmds)
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
