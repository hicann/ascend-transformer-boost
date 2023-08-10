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
#ifndef ACLTRANSFORMER_SHAREMEMORY_H
#define ACLTRANSFORMER_SHAREMEMORY_H

#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ipc.h>
#include <sys/shm.h>

#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <semaphore.h>
#include <iostream>


class CShareMemory {
public:
    CShareMemory(const char *name, uint32_t size);
    ~CShareMemory();

    void *GetShm() { return share_memory; };

    void SemLock() { sem_wait(sem); };

    void SemUnLock() { sem_post(sem); };

private:
    void *CreateShareMemory(const char *name, uint32_t size);

    sem_t *sem;
    uint8_t *share_memory;
    uint32_t mem_ize;
};

#endif
