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
#ifndef ASDTRANSFORM_MEM_ALLOCATION_SOLVER_H
#define ASDTRANSFORM_MEM_ALLOCATION_SOLVER_H
#include <cstdint>
#include <vector>

namespace AclTransformer {
class MemAllocationSolver {
public:
    MemAllocationSolver();
    virtual ~MemAllocationSolver();
    virtual char *Malloc(uint64_t blockSize) = 0;
    virtual void Free(char *blockAddress) = 0;
    virtual uint64_t GetSize() const;
    virtual uint64_t GetMallocSize() const;
    virtual void Reset();

protected:
    uint64_t totalSize_ = 0;
    uint64_t mallocTotalSize_ = 0;
};
} // namespace AclTransformer
#endif