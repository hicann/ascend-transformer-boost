/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef MODEL_MODEL_H
#define MODEL_MODEL_H
#include <string>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <thread>
#include <queue>
#include <torch/torch.h>
#include <asdops/utils/time/timer.h>
#include "acltransformer/operation.h"
#include "acltransformer/plan.h"

namespace AclTransformer {
class Model {
public:
    enum TensorType {
        INTERMEDIATE_TENSOR = 0,
        NOT_INTERMEDIATE_TENSOR,
    };

    struct Node {
        std::shared_ptr<Operation> operation;
        std::shared_ptr<Plan> plan;
        std::vector<AsdOps::Tensor *> inTensors;
        std::vector<AsdOps::Tensor *> outTensors;
        VariantPack variantPack;
        std::vector<torch::Tensor> torchTensors;
        AsdOps::SVector<TensorType> inTensorTypes;
        AsdOps::SVector<TensorType> outTensorTypes;
    };

    struct Graph {
        std::vector<AsdOps::Tensor> weightTensors;
        std::vector<AsdOps::Tensor> inTensors;
        std::vector<AsdOps::Tensor> outTensors;
        std::vector<AsdOps::Tensor> internalTensors;
        std::vector<Node> nodes;
        void Init();
        std::string ToString() const;

    private:
        void InitTensorType();
        bool IsInternalTensor(const AsdOps::Tensor *tensor);
    };

    Model(const std::string &modelName, const std::string &param);
    ~Model();
    void Init();

    virtual uint64_t GetInTensorCount() const = 0;
    virtual uint64_t GetOutTensorCount() const = 0;
    virtual AsdOps::Status InferShape(const std::vector<AsdOps::Tensor> &inTensors,
                                      std::vector<AsdOps::TensorDesc> &outTensorDescs) = 0;

    void SetWeight(const std::vector<AsdOps::Tensor> &weightTensors);
    AsdOps::Status Execute(Handle handle, std::vector<AsdOps::Tensor> &inTensors,
                           std::vector<AsdOps::Tensor> &outTensors, const std::string &param);

protected:
    virtual void BuildGraph() = 0;
    virtual AsdOps::Status ParseVarintPackParam(const std::string &param, int nodeId, AsdOps::Any &variantPackParam);

private:
    void BuildNodeVariantPack(int nodeId);
    void ExecuteNode(int nodeId);
    void ThreadProcessTask();
    void ExecutePlanSync(int nodeId);
    void ExecutePlanAsync(int nodeId);
    void PushTask(int nodeId);
    int PopTask();
    void WaitAsyncPlanExecuteFinish();
    std::string GetSaveTensorDir();

protected:
    std::string modelName_;
    std::string param_;
    Graph graph_;

    uint64_t executeCount_ = 0;
    AclTransformer::Handle handle_;
    AsdOps::Timer timer_;

    bool isUsePlanExecuteAsync_ = false;
    bool isTaskQueueEnable_ = false;
    std::queue<int> taskQueue_;
    std::mutex mutex_;
    std::condition_variable cond_;
    std::thread taskProcessThread_;
    std::atomic_bool allTaskFinish_;
    int32_t currentDevId_ = 0;
};
} // namespace AclTransformer

#endif