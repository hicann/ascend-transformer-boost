import logging
import os
import re
import subprocess  # nosec B404
import sys
import threading
import time

logging.getLogger().setLevel(logging.INFO)

deepseek_op_list = [
    "FusedAddTopkDivOperation",
    "MlaPreprocessOperation",
    "MultiLatentAttentionOperation",
    "FaUpdateOperation",
    "TopkToppSamplingOperation",
    "ReshapeAndCacheOperation",
    "SwigluQuantOperation",
    "LinearOperation",
    "GatingOperation",
    "RopeQConcatOperation",
    "NormRopeReshapeOperation",
    "GatherPreRmsNormOperation",
    "PagedAttentionOperation",
]
deepseek_op_dict = {}
for op in deepseek_op_list:
    op_key = op.replace('Operation', '').lower()
    deepseek_op_dict[op_key] = op

torch_atb_op_test_dict = {
    "layernorm": "layernorm",
    "elewise": "elewise",
    "linear": "linear",
    "softmax": "softmax",
    "selfattention": "self_attention",
    "pagedattention": "paged_attention",
    "rope": "rope",
    "split": "split",
    "gather": "gather",
    "activation": "activation",
    "rmsnorm": "rms_norm",
    "allgather": "all_gather",
    "asstrided": "as_strided",
    "cumsum": "cumsum",
    "dynamicntk": "dynamic_NTK",
    "multinomial": "multinomial",
    "concat": "concat",
    "reshapeandcache": "reshape_and_cache",
    "razorfusionattention": "razor_fusion_attention",
    "faupdate": "faupdate",
    "linearparallel": "linear_parallel",
    "topktoppsampling": "topk_topp_sampling",
    "alltoall": "all_to_all",
}

GROUP1_DEVICES = "0,1"
GROUP2_DEVICES = "2,3"
SINGLE_GROUP_DEVICES = "0,1,2,3"

LOG_DIR_NAME = "atb_test_logs"
_error_event = threading.Event()
_active_processes = []
_active_processes_lock = threading.Lock()


def _get_log_dir():
    log_dir = os.path.join(os.getcwd(), LOG_DIR_NAME)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def _sanitize_filename(name):
    return re.sub(r'[^\w\-.]', '_', name)


def _get_log_path(test_mode, log_dir=None):
    ts = time.strftime("%Y%m%d_%H%M%S")
    safe_name = _sanitize_filename(test_mode) if test_mode else "unknown"
    if log_dir is None:
        log_dir = _get_log_dir()
    return os.path.join(log_dir, f"{ts}_{safe_name}.log")


def _cat_log(log_path):
    if os.path.exists(log_path):
        logging.info(f"========== LOG START: {os.path.basename(log_path)} ==========")
        with open(log_path, 'r', encoding='utf-8') as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                sys.stdout.write(chunk)
                sys.stdout.flush()
        logging.info(f"========== LOG END: {os.path.basename(log_path)} ==========")
    else:
        logging.warning(f"Log file not found: {log_path}")


def _terminate_active_processes():
    with _active_processes_lock:
        for proc in _active_processes:
            if proc.poll() is None:
                try:
                    proc.terminate()
                except Exception:  # nosec B110
                    pass
        _active_processes.clear()


def _register_process(proc):
    with _active_processes_lock:
        _active_processes.append(proc)


def _unregister_process(proc):
    with _active_processes_lock:
        if proc in _active_processes:
            _active_processes.remove(proc)


def capitalize_first_letter(s):
    if not s:
        return s
    return s[0].upper() + s[1:]


def read_diff_file_list(args):
    diff_file_path = os.path.join(os.getcwd(), "pr_filelist.txt")
    logging.info("--------------------------------------------------------------------------------------------------")
    if not os.path.exists(diff_file_path):
        logging.warning(f"pr_filelist.txt not found: {diff_file_path}")
        return ""
    with open(diff_file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.read().splitlines()]
    file_list = "\n".join(line for line in lines if line)
    logging.info(f"diff files from pr_filelist.txt are: \n{file_list}")
    return file_list


def get_op_from_header_file(file_name):
    cmd = ["git", "log", "-p", "-1", "--", file_name]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)  # nosec B603
    lines = result.stdout.splitlines()
    op_params = set()
    for line in lines:
        logging.info(f"line: {line}")
        if re.search("struct .*Param {", str(line), re.I):
            words = line.split(" ")
            for word in words:
                if re.search("Param", word, re.I):
                    op_params.add(word.split("Param")[0].lower())
    logging.info("--------------------------------------------------------------------------------------------------")
    logging.info(f"ops from include header files are: {op_params}")
    return op_params


def build_ops_dict():
    op_dict = {}
    path0 = [f"{os.getcwd()}/src/ops_infer", f"{os.getcwd()}/src/ops_train"]
    for path in path0:
        if os.path.exists(path) and os.path.isdir(path):
            for file in os.listdir(path):
                key = file.split('.')[0].replace('_', '').lower()
                if key not in op_dict.keys():
                    op_dict[key] = file.split('.')[0]

    paths1 = [f"{os.getcwd()}/src/kernels/kernels", f"{os.getcwd()}/src/kernels/mixkernels"]
    for path in paths1:
        if os.path.exists(path) and os.path.isdir(path):
            for file in os.listdir(path):
                key = file.split('.')[0].replace('_', '').lower()
                if key not in op_dict.keys():
                    op_dict[key] = file.split('.')[0]

    paths2 = [f"{os.getcwd()}/tests/apitest/kernelstest/mix"]
    for path in paths2:
        if os.path.exists(path) and os.path.isdir(path):
            for file in os.listdir(path):
                key = file[len("test_") : -len(".py")]
                key = key.replace('_', '').lower()
                if key not in op_dict.keys():
                    op_dict[key] = file.split('.')[0]
    op_dict["mla"] = "1"

    paths3 = [f"{os.getcwd()}/src/ops/ops_infer", f"{os.getcwd()}/ops/src/ops_train"]
    for path in paths3:
        if os.path.exists(path) and os.path.isdir(path):
            for file in os.listdir(path):
                key = file.split('.')[0].replace('_', '').lower()
                if key not in op_dict.keys():
                    op_dict[key] = file.split('.')[0]

    return op_dict


def parse_cmd_entry(entry):
    """Parse (cmd, test_mode) or (cmd, test_mode, cwd) command tuples."""
    if len(entry) == 2:
        return entry[0], entry[1], None
    if len(entry) == 3:
        return entry[0], entry[1], entry[2]
    raise ValueError(f"Invalid command entry: {entry!r}")


def run_test_cmd(cmd, test_mode=None, cwd=None):
    try:
        log_dir = _get_log_dir()
    except OSError as e:
        logging.error(f"Failed to create log directory: {e}")
        exit(1)
    log_path = _get_log_path(test_mode, log_dir)
    if cwd:
        logging.info(f"[{test_mode}] cwd={cwd}")
    logging.info(f"[{test_mode}] Log file: {log_path}")
    try:
        with open(log_path, 'w', encoding='utf-8') as log_file:
            popen_kwargs = {"stdout": log_file, "stderr": subprocess.STDOUT, "shell": False}
            if cwd:
                popen_kwargs["cwd"] = cwd
            process = subprocess.Popen(cmd, **popen_kwargs)  # nosec B603
            _register_process(process)
            process.communicate()
            _unregister_process(process)
        return_code = process.poll()
    except FileNotFoundError:
        logging.error(f"{test_mode} Not Found!")
        _cat_log(log_path)
        _error_event.set()
        _terminate_active_processes()
        exit(1)

    _cat_log(log_path)
    if return_code != 0:
        logging.error(f"Run {test_mode} failed with return code {return_code}")
        _error_event.set()
        _terminate_active_processes()
        exit(1)
    else:
        logging.info(f"{test_mode} finished with exit code: {return_code}")
        return True


def run_test_cmd_soft(cmd, test_mode=None, cwd=None):
    try:
        log_dir = _get_log_dir()
    except OSError as e:
        logging.error(f"Failed to create log directory: {e}")
        return False
    log_path = _get_log_path(test_mode, log_dir)
    if cwd:
        logging.info(f"[{test_mode}] cwd={cwd}")
    logging.info(f"[{test_mode}] Log file: {log_path}")
    try:
        with open(log_path, 'w', encoding='utf-8') as log_file:
            popen_kwargs = {"stdout": log_file, "stderr": subprocess.STDOUT, "shell": False}
            if cwd:
                popen_kwargs["cwd"] = cwd
            process = subprocess.Popen(cmd, **popen_kwargs)  # nosec B603
            _register_process(process)
            process.communicate()
            _unregister_process(process)
        return_code = process.poll()
    except FileNotFoundError:
        logging.error(f"{test_mode} Not Found!")
        _cat_log(log_path)
        return False

    _cat_log(log_path)
    if return_code != 0:
        logging.error(f"Run {test_mode} failed with return code {return_code}")
        return False
    else:
        logging.info(f"{test_mode} finished with exit code: {return_code}")
        return True


def shell_source(script):
    try:
        process = subprocess.Popen('/bin/bash -c "source %s; env"' % script, stdout=subprocess.PIPE, shell=True)  # nosec B602
    except FileNotFoundError:
        logging.error(f"{script} Not Found!")
        exit(1)
    output = process.communicate()[0]
    for line in output.splitlines():
        line_decode = line.decode()
        logging.info(line_decode)
        if len(line_decode.split("=")) > 1:
            key, value = line.decode().split("=", 1)
            os.environ.update({key: value})
    process.wait()
    return_code = process.poll()
    if return_code != 0:
        logging.error(f"Run {script} failed with return code {return_code}")
        exit(1)
    else:
        logging.info(f"{script} finished with exit code: {return_code}")
        return True


def install_torch_atb():
    python_version = f"{sys.version_info.major}{sys.version_info.minor}"
    target_dict = f"{os.getcwd()}/output/whl"
    if not os.path.exists(target_dict):
        logging.error(f"whl directory not found: {target_dict}, skip install_torch_atb")
        return False
    whl_file = next((f for f in os.listdir(target_dict) if re.match(rf'.*-cp{python_version}-.*\.whl$', f)), None)
    logging.info(f"whl_file:  {whl_file}")
    if whl_file:
        subprocess.run(["pip", "install", "--force-reinstall", f"{target_dict}/{whl_file}"], check=True)  # nosec B603 B607
        logging.info(f"Successfully installed {whl_file}!")
        return True
    else:
        logging.error(f"whl file not found! current path: {os.getcwd()}")
        return False


def run_deepseek_smoke_testcase(deepssek_op):
    smoke_testcase_path = f"{os.getcwd()}/tests/high_level_test/{deepssek_op}/Smoke/"
    if os.path.exists(smoke_testcase_path):
        logging.info(
            f"---------------------------------------- Starting Run Python Operations Smoke Test For {deepssek_op} ----------------------------------------"
        )
        ATB_HOME_PATH = os.environ.get('ATB_HOME_PATH')
        logging.info(ATB_HOME_PATH)
        cmd = ["python3", "-m", "unittest", "discover", "-s", smoke_testcase_path, "-p", "test*.py"]
        if run_test_cmd(cmd, test_mode=f"pythontest for {deepssek_op}"):
            logging.info(
                f"---------------------------------------- Run Python Operations Smoke Test For {deepssek_op} Done!----------------------------------------"
            )
        for _, _, files in os.walk(smoke_testcase_path):
            for filename in files:
                if filename.endswith('.csv'):
                    logging.info(
                        f"---------------------------------------- Starting Run CSV Smoke Test For {deepssek_op} ----------------------------------------"
                    )
                    cmd = [
                        "bash",
                        f"{os.getcwd()}/scripts/build.sh",
                        "csvopstest",
                        f"--csvopstest_options= -i {os.path.join(smoke_testcase_path, filename)}",
                        "--skip_build",
                    ]
                    if run_test_cmd(cmd, test_mode=f"csv test for {deepssek_op}"):
                        logging.info(
                            f"----------------------------------------Run CSV Smoke Test For {deepssek_op} Done! ----------------------------------------"
                        )


def run_cmd_with_devices(cmd, visible_devices, test_mode=None):
    env = os.environ.copy()
    env["ASCEND_RT_VISIBLE_DEVICES"] = visible_devices
    logging.info(f"[{test_mode}] ASCEND_RT_VISIBLE_DEVICES={visible_devices}")
    try:
        log_dir = _get_log_dir()
    except OSError as e:
        logging.error(f"Failed to create log directory: {e}")
        return False
    log_path = _get_log_path(test_mode, log_dir)
    try:
        with open(log_path, 'w', encoding='utf-8') as log_file:
            process = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT, shell=False)  # nosec B603
            _register_process(process)
            process.communicate()
            _unregister_process(process)
        return_code = process.poll()
    except FileNotFoundError:
        logging.error(f"{test_mode} Not Found!")
        _cat_log(log_path)
        return False

    _cat_log(log_path)
    if return_code != 0:
        logging.error(f"Run {test_mode} failed with return code {return_code}")
        return False
    else:
        logging.info(f"{test_mode} finished with exit code: {return_code}")
        return True


def run_group_serial(cmds, visible_devices, error_event=None, cat_on_failure=True):
    env = os.environ.copy()
    env["ASCEND_RT_VISIBLE_DEVICES"] = visible_devices
    try:
        log_dir = _get_log_dir()
    except OSError as e:
        logging.error(f"Failed to create log directory: {e}")
        if error_event:
            error_event.set()
        return False, None

    for entry in cmds:
        cmd, test_mode, cwd = parse_cmd_entry(entry)
        if error_event and error_event.is_set():
            logging.warning(f"[Group(devices={visible_devices})] Skipping {test_mode} due to error in parallel group")
            return False, None

        log_path = _get_log_path(f"g{visible_devices}_{test_mode}", log_dir)
        cwd_info = f" | cwd={cwd}" if cwd else ""
        logging.info(f"[Group(devices={visible_devices})] Starting: {test_mode}{cwd_info} | Log: {log_path}")
        try:
            with open(log_path, 'w', encoding='utf-8') as log_file:
                popen_kwargs = {
                    "env": env,
                    "stdout": log_file,
                    "stderr": subprocess.STDOUT,
                    "shell": False,
                }
                if cwd:
                    popen_kwargs["cwd"] = cwd
                process = subprocess.Popen(cmd, **popen_kwargs)  # nosec B603
                _register_process(process)
                process.communicate()
                _unregister_process(process)
            return_code = process.poll()
        except FileNotFoundError:
            logging.error(f"{test_mode} Not Found!")
            if cat_on_failure:
                _cat_log(log_path)
            if error_event:
                error_event.set()
            return False, log_path

        if return_code != 0:
            logging.error(f"[Group(devices={visible_devices})] {test_mode} failed with return code {return_code}")
            if cat_on_failure:
                _cat_log(log_path)
            if error_event:
                error_event.set()
            return False, log_path
        else:
            _cat_log(log_path)
            logging.info(f"[Group(devices={visible_devices})] {test_mode} finished with exit code: {return_code}")
    return True, None


def run_legacy_two_groups(group1_cmds, group2_cmds):
    try:
        log_dir = _get_log_dir()
    except OSError as e:
        logging.error(f"Failed to create log directory: {e}")
        exit(1)
    logging.info(f"Log directory: {log_dir}")
    results = {}
    error_event = threading.Event()
    error_log_path = [None]

    def _run_group(cmds, devices, group_name):
        rc, failed_log = run_group_serial(cmds, devices, error_event=error_event, cat_on_failure=False)
        results[group_name] = rc
        if not rc and failed_log:
            error_log_path[0] = failed_log

    t1 = threading.Thread(target=_run_group, args=(group1_cmds, GROUP1_DEVICES, "group1"))
    t2 = threading.Thread(target=_run_group, args=(group2_cmds, GROUP2_DEVICES, "group2"))

    logging.info(
        "======================================== Starting Parallel Groups ========================================"
    )
    logging.info(f"Group 1 (devices={GROUP1_DEVICES}): {[parse_cmd_entry(e)[1] for e in group1_cmds]}")
    logging.info(f"Group 2 (devices={GROUP2_DEVICES}): {[parse_cmd_entry(e)[1] for e in group2_cmds]}")
    logging.info(
        "========================================================================================================"
    )

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    if not results.get("group1", False) or not results.get("group2", False):
        logging.error(f"Parallel groups failed: group1={results.get('group1')}, group2={results.get('group2')}")
        if error_log_path[0]:
            logging.error("========== Error log from failed task ==========")
            _cat_log(error_log_path[0])
        _terminate_active_processes()
        exit(1)
    logging.info(
        "======================================== Parallel Groups Completed ========================================"
    )


def run_n_groups(groups):
    """Run multiple device groups in parallel.

    groups: [(cmds, devices, group_name), ...]. Each group runs cmds serially on its devices.
    """
    _get_log_dir()
    results = {}
    error_event = threading.Event()
    error_log_path = [None]

    def _run_group(cmds, devices, group_name):
        rc, failed_log = run_group_serial(cmds, devices, error_event=error_event, cat_on_failure=False)
        results[group_name] = rc
        if not rc and failed_log:
            error_log_path[0] = failed_log

    threads = [
        threading.Thread(target=_run_group, args=(cmds, devices, group_name)) for cmds, devices, group_name in groups
    ]

    logging.info(
        "======================================== Starting Parallel Groups ========================================"
    )
    for cmds, devices, group_name in groups:
        logging.info(f"Group {group_name} (devices={devices}): {[parse_cmd_entry(e)[1] for e in cmds]}")
    logging.info(
        "========================================================================================================"
    )

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    failed = [name for name, rc in results.items() if not rc]
    if failed:
        logging.error(f"Parallel groups failed: {failed}")
        if error_log_path[0]:
            logging.error("========== Error log from failed task ==========")
            _cat_log(error_log_path[0])
        _terminate_active_processes()
        exit(1)
    logging.info(
        "======================================== Parallel Groups Completed ========================================"
    )
