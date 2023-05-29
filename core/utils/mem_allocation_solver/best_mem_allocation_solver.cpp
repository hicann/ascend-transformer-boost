/*
 * Copyright(C) 2023. Huawei Technologies Co.,Ltd. All rights reserved.
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
#include "acltransformer/utils/mem_allocation_solver/best_mem_allocation_solver.h"
#include <asdops/utils/log/log.h>

namespace AclTransformer {
BestMemAllocationSolver::BestMemAllocationSolver() {}

BestMemAllocationSolver::~BestMemAllocationSolver() {}

char *BestMemAllocationSolver::Malloc(uint64_t blockSize)
{
    ASD_LOG(INFO) << "BestMemAllocationSolver::Malloc, blockSize:" << blockSize;
    mallocTotalSize_ += blockSize;
    for (auto &block : blocks_) {
        if (!block.used && block.size >= blockSize) {
            block.used = true;
            return (char *)block.offset;
        }
    }

    Block newBlock;
    newBlock.offset = totalSize_;
    newBlock.size = blockSize;
    newBlock.used = true;
    totalSize_ += blockSize;
    blocks_.push_back(newBlock);

    return (char *)newBlock.offset;
}

void BestMemAllocationSolver::Free(char *blockAddress)
{
    ASD_LOG(INFO) << "BestMemAllocationSolver::Free, blockAddress:" << (uint64_t)blockAddress;
    for (auto &block : blocks_) {
        uint64_t offset = (uint64_t)blockAddress;
        if (offset == block.offset) {
            block.used = false;
        }
    }
}

void BestMemAllocationSolver::Reset()
{
    blocks_.clear();
    MemAllocationSolver::Reset();
}

} // namespace AclTransformer