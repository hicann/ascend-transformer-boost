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
#include "acltransformer/utils/mem_allocation_solver/mem_allocation_solver.h"
#include <asdops/utils/log/log.h>

namespace AclTransformer {
MemAllocationSolver::MemAllocationSolver() {}

MemAllocationSolver::~MemAllocationSolver() {}

uint64_t MemAllocationSolver::GetSize() const
{
    ASD_LOG(INFO) << "MemAllocationSolver::GetSize " << totalSize_;
    return totalSize_;
}

uint64_t MemAllocationSolver::GetMallocSize() const
{
    ASD_LOG(INFO) << "MemAllocationSolver::GetMallocSize " << mallocTotalSize_;
    return mallocTotalSize_;
}

void MemAllocationSolver::Reset()
{
    totalSize_ = 0;
    mallocTotalSize_ = 0;
}

} // namespace AclTransformer