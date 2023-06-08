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
#ifndef ASDTRANSFORM_BEST_MEM_ALLOCATION_SOLVER_H
#define ASDTRANSFORM_BEST_MEM_ALLOCATION_SOLVER_H
#include "acltransformer/utils/mem_allocation_solver/mem_allocation_solver.h"

namespace AclTransformer {
struct Block {
    uint64_t offset = 0;
    uint64_t size = 0;
    bool used = false;
};

class BestMemAllocationSolver : public MemAllocationSolver {
public:
    BestMemAllocationSolver();
    virtual ~BestMemAllocationSolver();
    char *Malloc(uint64_t blockSize) override;
    void Free(char *blockAddress) override;
    void Reset() override;

private:
    std::vector<Block> blocks_;
};
} // namespace AclTransformer
#endif