/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <benchmark/benchmark.h>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
// Run the benchmark

bool isBaseLine = false;

void loadArgs(const std::vector<std::string> &args)
{
    for (const auto &arg : args) {
        if (arg == "atb_proftest_baseline") {
            isBaseLine = true;
        }
    }
}

class ProftestReporter : public benchmark::ConsoleReporter {
public:
    ProftestReporter(std::string execPath) : execPath(execPath) {}
    void ReportRuns(const std::vector<Run> &reports) override
    {
        for (auto &report : reports) {
            std::string caseName = report.benchmark_name();
            size_t index = caseName.find('/');
            if (index != std::string::npos) {
                caseName = caseName.substr(0, index);
            }
            std::filesystem::path baselinePath = std::filesystem::weakly_canonical(
                std::filesystem::path(execPath + "/../../../../proftest/BaseLine/" + caseName + "_BaseLine.csv"));
            std::filesystem::path resultPath = std::filesystem::weakly_canonical(
                std::filesystem::path(execPath + "/../../../../proftest/Result/" + caseName + "_Result.csv"));
            if (isBaseLine) {
                std::filesystem::create_directories(baselinePath.parent_path());
                std::ofstream baselineFile(baselinePath, std::ios::out | std::ios::trunc);
                if (!baselineFile.is_open()) {
                    std::cerr << "Failed to open baseline file: " << baselinePath << std::endl;
                    exit(1);
                }
                baselineFile << "Benchmark_Name,Real Time (s),CPU Time (s)\n";
                baselineFile << caseName << "," << report.real_accumulated_time << "," << report.cpu_accumulated_time << "\n";
                baselineFile.close();
                return;
            }

            std::filesystem::create_directories(resultPath.parent_path());
            std::ofstream resultFile(resultPath, std::ios::out | std::ios::trunc);
            if (!resultFile.is_open()) {
                std::cerr << "Failed to open result file: " << resultPath << std::endl;
                exit(1);
            }
            resultFile << "Benchmark_Name,Real Time (s),CPU Time (s)\n";
            resultFile << caseName << "," << report.real_accumulated_time << "," << report.cpu_accumulated_time << "\n";
            resultFile.close();

            if (std::filesystem::exists(baselinePath)) {
                std::ifstream baselineFile(baselinePath, std::ios::in);
                if (!baselineFile.is_open()) {
                    std::cerr << "Failed to open baseline file: " << baselinePath << " , skip compare.\n";
                    return;
                }
                double baselineCpuTime = 0;
                std::string line;
                while (std::getline(baselineFile, line)) {
                    std::istringstream iss(line);
                    std::string benchmarkName, realTime, cpuTime;
                    if (std::getline(iss, benchmarkName, ',') && std::getline(iss, realTime, ',') &&
                        std::getline(iss, cpuTime, ',')) {
                        if (benchmarkName == caseName) {
                            baselineCpuTime = std::stod(cpuTime);
                        }
                    }
                }
                baselineFile.close();
                double resultCpuTime = report.cpu_accumulated_time;
                if (resultCpuTime > baselineCpuTime * 1.05) {
                    std::cerr << "Benchmark " << caseName << " has degraded. Please check:\n";
                    std::cerr << "Baseline file: " << baselinePath << "\n";
                    std::cerr << "Result file: " << resultPath << "\n";
                    exit(1);
                }
            }
        }

        return;
    }

private:
    std::string execPath;
};


int main(int argc, char **argv)
{
    // 保存执行文件的路径，用于确定基线和输出文件的目录。
    std::string execPath = argv[0];
    benchmark::MaybeReenterWithoutASLR(argc, argv);
    char arg0_default[] = "benchmark";
    char *args_default = reinterpret_cast<char *>(arg0_default);
    if (!argv) {
        argc = 1;
        argv = &args_default;
    }
    {
        // 读取自定义参数，并从benchmark的参数中剔除。
        std::vector<std::string> proftestArgs;
        std::vector<char *> new_argv;
        new_argv.push_back(argv[0]);
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg.substr(0, 13) == "atb_proftest_") {
                proftestArgs.push_back(arg);
            } else {
                new_argv.push_back(argv[i]);
            }
        }
        argc = new_argv.size();
        argv = new_argv.data();

        loadArgs(proftestArgs);
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks(new ProftestReporter(execPath));
    benchmark::Shutdown();
    std::cout << "Result: OK. All case's proftest are done." << std::endl;
    return 0;
}
int main(int, char **);