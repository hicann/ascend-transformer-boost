# pre-commit配置指导书

## [TOC]

## 1 概述

本指导书主要用于指导CANN社区在代码仓中部署pre-commit能力（主要包括代码格式化及OAT扫描能力），并指导社区贡献者如何在本地使用。

### 1.1 GitCode 镜像仓库说明

**重要**: 由于国内网络环境，访问 GitHub 可能受限，CANN 社区已将常用的 pre-commit hooks 仓库镜像到 GitCode。

**常用镜像仓库对照表**:

| GitHub 原仓库 | GitCode 镜像仓库 | 说明 |
|---|---|---|
| `https://github.com/pre-commit/pre-commit-hooks` | `https://gitcode.com/pre-commit/pre-commit-hooks` | 基础检查 hooks |
| `https://github.com/pre-commit/mirrors-clang-format` | `https://gitcode.com/pre-commit-clang/mirrors-clang-format` | C++ 格式化 |
| `https://github.com/astral-sh/ruff-pre-commit` | `https://gitcode.com/gh_mirrors/ru/ruff-pre-commit` | Python lint/format |
| `https://github.com/codespell-project/codespell` | `https://gitcode.com/gh_mirrors/co/codespell` | 拼写检查 |
| `https://github.com/crate-ci/typos` | `https://gitcode.com/gh_mirrors/ty/typos` | typos 检查 |
| `https://github.com/pylint-dev/pylint` | `https://gitcode.com/gh_mirrors/pyl/pylint` | Python 代码检查 |
| `https://github.com/PyCQA/bandit` | `https://gitcode.com/gh_mirrors/ba/bandit` | Python 安全检查 |

**使用建议**: 在国内开发时，推荐使用 GitCode 镜像仓库，避免网络访问问题。

## 2 代码仓中部署pre-commit能力指导

1、在代码仓根目录下创建".pre-commit-config.yaml"文件，如果已经存在，表示该代码仓中已经启用pre-commit能力，可参考后续步骤配置代码格式化及OAT检查能力；

2、在".pre-commit-config.yaml"文件中配置C++代码格式化能力：

- **在".pre-commit-config.yaml"文件添加如下配置：**

```text
repos:
  - repo: https://gitcode.com/pre-commit-clang/mirrors-clang-format
    rev: v18.1.8
    hooks:
      - id: clang-format
        files: \.(c|h|cpp|hpp|cc|hh|cxx|hxx)$
```

**说明**：

- 对于空的".pre-commit-config.yaml"直接拷贝上述内容
- 如果已经有配置，请忽略首行的"repos:"
- **重要**: 使用 GitCode 镜像仓库替代 GitHub 仓库，避免国内网络访问问题
- **将".clang-format"配置文件拷贝至代码仓根目录下** 配套提供的".clang-format"为与CANN社区编程规范配套的格式化模板文件，如果代码仓有与规范不匹配的代码风格，可自主调整配置参数。

3、在".pre-commit-config.yaml"文件中配置OAT扫描能力：

- **在".pre-commit-config.yaml"文件添加如下配置：**

```yaml
  - repo: local
    hooks:
      - id: oat-check
        name: OAT Compliance Check (Python)
        entry: bash scripts/oat_check.sh
        language: system
        pass_filenames: true
        types: [file]
        stages: [pre-commit]
        verbose: true
```

- **将"oat_check.sh"脚本拷贝至代码仓中的`scripts/`目录下；**

**说明**："oat_check.sh"脚本如果放在其他路径下，需要在".pre-commit-config.yaml"中的oat-check配置中同步调整执行"oat_check.sh"脚本路径信息。

4、至此将完成在代码仓中配置pre-commit机制下的代码格式化、OAT检查配置，如果还需要其他功能，可继续在".pre-commit-config.yaml"中配置。

## 3 社区贡献者使用pre-commit能力

### 3.1 pre-commit安装步骤

步骤 1: 安装 pre-commit 框架

```bash
# 使用 pip（推荐）
pip install pre-commit

# 验证安装
pre-commit --version
# 输出: pre-commit 3.x.x
```

**Windows 用户**: 确保已安装 Python 和 pip。

步骤 2: 进入项目目录

```bash
cd /path/to/your/project

# 例如
cd d:\complianceRepo\CANN
```

步骤 3: 安装 Git Hooks

```bash
# 在项目根目录运行
pre-commit install
```

步骤 4: 验证安装（可选）

```bash
# 测试 hook（不会真正提交）
git commit --allow-empty -m "test pre-commit"
```

后续在提交代码前会自动进行代码格式化处理及触发OAT检查。

### 3.2 OAT使用指导

**OAT（Open Source Audit Tool）Python 版** 是一个开源合规性检查工具，自动集成到 Git 提交流程中。仅依赖 **Python 3.7+**，`oat-py` 包在首次运行时自动安装。

#### 3.2.1 检查内容

**文件类型检查** - 禁止提交二进制文件（.so, .dll, .exe 等）
**许可证头检查** - 验证源代码文件包含合规的许可证声明

#### 3.2.2 核心特点

- **增量检查** - 仅检查待提交文件，速度快（通常 < 2 秒）
- **自动触发** - 每次 `git commit` 自动运行
- **详细报告** - 自动生成 `oat_reports/result.txt` 摘要报告
- **零配置** - 首次运行自动安装 `oat-py`（需可访问 PyPI 或内部镜像）
- **跨平台** - Windows / Linux / macOS 全支持（只需 Python 3.7+）

#### 3.2.3 必需软件

| 软件 | 版本要求 | 用途 | 安装方式 |
|---|---|---|---|
| **Python** | 3.7+ | 运行 OAT | 通常已安装；参见下方安装指导 |
| **oat-py** | >=1.0.0 | OAT 核心包 | **首次运行自动安装** |
| **Git** | 2.0+ | 版本控制 | 通常已安装 |
| **pre-commit** | 2.0+ | Hook 框架 | `pip install pre-commit` |

#### 3.2.4 自动安装支持

`oat_check.sh` 脚本在运行时会自动检测并安装 `oat-py`：

| 平台 | Python 检测 | oat-py 安装 | 首次安装时间 |
|---|---|---|---|
| **Linux** | 自动检测 `python3/python/py` | 自动（`pip install`） | < 30 秒 |
| **macOS** | 自动检测 `python3/python/py` | 自动（`pip install`） | < 30 秒 |
| **Windows（Git Bash）** | 自动检测 `python3/python/py` | 自动（`pip install`） | < 30 秒 |

#### 3.2.5 重要提示：环境问题自动跳过

**友好的设计**：如果 Python 未找到或 `oat-py` 安装失败，OAT 检查会**自动跳过**，提交仍会继续。

**会自动跳过的场景**

| 场景 | 行为 | 提示 |
|---|---|---|
| Python 3.7+ 未安装 | 跳过检查，允许提交 | 提供安装指引 |
| oat-py 自动安装失败 | 跳过检查，允许提交 | 提示手动安装方法 |
| OAT 扫描执行异常（退出码非 0/1） | 跳过检查，允许提交 | 提示手动运行命令排查 |

**仍会阻止提交的场景**

| 场景 | 行为 | 原因 |
|---|---|---|
| **发现二进制文件** | 阻止提交 | 真正的合规性问题 |
| **许可证头缺失/错误** | 阻止提交 | 真正的合规性问题 |

**跳过检查的提示示例**

```text
[OAT] [WARNING] Python 3.7+ is required but not found. Please install Python 3.7 or later.
[OAT] Skipping OAT check, continuing commit...
```

**后续手动运行检查**

配置好环境后，可以手动运行检查：

```bash
# 推荐方式
pre-commit run oat-check

# 或直接运行脚本
bash scripts/oat_check.sh
```

#### 3.2.6 合规性问题（阻止提交）

**重要**: 以下问题会**阻止提交**，必须修复。

**1) 发现无效文件类型**

**场景**: 尝试提交二进制文件（.so, .dll, .exe 等）。

**输出**:

```text
====================================================================
  OAT: Compliance issues found. Commit blocked.
====================================================================

[OAT] Found 1 compliance issue(s):
  - Invalid File Type:       1
  - License Header Invalid:  0

[OAT] Details:
  cat oat_reports/result.txt

Fix the issues and recommit, or skip with:
  git commit --no-verify
```

**行为**: **阻止提交，必须修复**

**查看详情**:

```bash
cat oat_reports/result.txt
```

**报告内容示例**:

```text
===================================
OAT Scan Result Summary
===================================
Scan Time: 2026-03-25 14:30:15
Project: CANN
Files Checked: 1

-----------------------------------
Invalid File Type Total Count: 1
lib/libtest.so: BINARY_FILE_TYPE

-----------------------------------
License Header Invalid Total Count: 0

===================================
```

**解决方案**:

```bash
# 方法 1: 移除二进制文件
git reset HEAD lib/libtest.so

# 方法 2: 将二进制文件添加到 .gitignore
echo "*.so" >> .gitignore
echo "*.dll" >> .gitignore
echo "*.exe" >> .gitignore

# 重新提交
git add .gitignore
git commit -m "update: add binary files to gitignore"
```

**2) 许可证头无效**

**场景**: 源代码文件缺少或许可证头格式不正确。

**输出**:

```text
====================================================================
  OAT: Compliance issues found. Commit blocked.
====================================================================

[OAT] Found 2 compliance issue(s):
  - Invalid File Type:       0
  - License Header Invalid:  2

[OAT] Details:
  cat oat_reports/result.txt
```

**行为**: **阻止提交，必须修复**

**查看详情**:

```bash
cat oat_reports/result.txt
```

**报告内容示例**:

```text
===================================
OAT Scan Result Summary
===================================

-----------------------------------
Invalid File Type Total Count: 0

-----------------------------------
License Header Invalid Total Count: 2
src/main.cpp: MISSING_LICENSE_HEADER
src/utils.cpp: MISSING_LICENSE_HEADER

===================================
```

**解决方案**:

在文件顶部添加许可证头，例如 CANN-2.0：

```cpp
/**
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

```

**重新提交**:

```bash
git add src/main.cpp src/utils.cpp
git commit -m "fix: add license headers"
```

---

#### 3.2.7 报告查看

**报告文件位置**

| 报告类型 | 文件路径 | 内容 |
|---|---|---|
| **摘要报告** | `oat_reports/result.txt` | 关键问题汇总 |

> `oat_reports/` 目录会被脚本自动加入 `.gitignore`，无需手动维护。

**查看命令**

```bash
# 查看报告
cat oat_reports/result.txt

# 使用编辑器查看
code oat_reports/result.txt
vim oat_reports/result.txt
```

**摘要报告内容**

```text
===================================
OAT Scan Result Summary
===================================
Scan Time: 2026-03-25 14:30:15
Project: CANN
Files Checked: 3

-----------------------------------
Invalid File Type Total Count: 0

-----------------------------------
License Header Invalid Total Count: 0

===================================
```

#### 3.2.8 环境问题

**1) Python 未安装或版本过低**

**场景**: 系统未安装 Python，或版本低于 3.7。

**输出**:

```text
[OAT] [WARNING] Python 3.7+ is required but not found. Please install Python 3.7 or later.
[OAT] Skipping OAT check, continuing commit...
```

**行为**: **跳过检查，允许提交**

**解决方案**：安装 Python 3.7+，参见下方《Python 安装指导》。安装完成后运行 `pre-commit run oat-check` 验证。

---

**2) oat-py 自动安装失败**

**场景**: `pip install oat-py>=1.0.0` 执行失败（网络问题或权限问题）。

**输出**:

```text
[OAT] oat-py not found. Installing oat-py>=1.0.0 ...
[OAT] [WARNING] Failed to install oat-py. Please run: pip install oat-py>=1.0.0
[OAT] Skipping OAT check, continuing commit...
```

**行为**: **跳过检查，允许提交**

**解决方案**:

```bash
# 手动安装
pip install "oat-py>=1.0.0"

# 国内网络使用镜像加速
pip install "oat-py>=1.0.0" -i https://mirrors.huaweicloud.com/repository/pypi/simple/

# 验证安装
python -m oat -v

# 手动运行检查
pre-commit run oat-check
```

---

**3) OAT 扫描执行异常**

**场景**: `oat` 命令运行时出现意外错误（退出码非 0 或 1）。

**输出**:

```text
[OAT] [WARNING] oat exited with unexpected code 2.
[OAT] Try re-running manually:
  python -m oat -mode s -s /path/to/repo -r /path/to/repo/oat_reports -n REPO_NAME -w 1 -f file1,file2
[OAT] Skipping OAT check, continuing commit...
```

**行为**: **跳过检查，允许提交**

**解决方案**:

```bash
# 步骤 1: 手动执行输出的命令，查看详细报错
python -m oat -mode s -s . -r oat_reports -n CANN -w 1 -f src/foo.cpp

# 步骤 2: 检查 oat-py 版本
python -m oat -v

# 步骤 3: 重装 oat-py
pip install --upgrade oat-py

# 步骤 4: 验证
pre-commit run oat-check
```

---

##### Python 安装指导

OAT Python 版需要 **Python 3.7+**，以下提供各平台安装方式。

**Linux**

```bash
# Ubuntu / Debian
sudo apt update && sudo apt install python3 python3-pip

# CentOS / RHEL
sudo yum install python3 python3-pip

# 验证
python3 --version
```

**macOS**

```bash
# 推荐使用 Homebrew
brew install python@3.11

# 验证
python3 --version
```

**Windows**

**方式一：官网安装包（推荐）**

1. 访问 [https://www.python.org/downloads/windows/](https://www.python.org/downloads/windows/)
2. 下载最新 Python 3.x（3.7 及以上均可）的 **Windows installer (64-bit)**
3. 双击安装，**务必勾选"Add Python to PATH"**
4. 安装完成后重启终端，验证：

    ```powershell
    python --version
    pip --version
    ```

**方式二：使用 Scoop 包管理器**

```powershell
# 安装 Scoop（首次使用）
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
irm get.scoop.sh | iex

# 安装 Python
scoop install python

# 验证
python --version
```

**方式三：使用 Winget（Windows 11 / Windows 10 内置）**

```powershell
winget install Python.Python.3.11
```

> **国内网络备用下载**：
> 华为镜像：[https://mirrors.huaweicloud.com/python/](https://mirrors.huaweicloud.com/python/)
