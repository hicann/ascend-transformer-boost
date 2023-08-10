#include "acltransformer/share_memory.h"
#include <asdops/utils/log/log.h>
#include <cstring>

CShareMemory::CShareMemory(const char *name, uint32_t size) : mem_ize(size)
{
    sem = sem_open(name, O_CREAT, S_IWOTH, 1);
    if (SEM_FAILED == sem) {
        ASD_LOG(ERROR) << "share memory open fail";
    }
    ASD_LOG(INFO) << "create share memory begin";
    SemLock();
    share_memory = (uint8_t *)CreateShareMemory(name, mem_ize);
    ASD_LOG(INFO) << "create share memory success";
    SemUnLock();
}

void *CShareMemory::CreateShareMemory(const char *name, uint32_t size)
{
    int fd, shmid;
    void *memory;
    struct shmid_ds buf;
    char filename[32];

    snprintf(filename, 32, "/tmp/.%s", name);
    if ((fd = open(filename, O_RDWR | O_CREAT | O_EXCL, O_RDWR)) > 0) {
        close(fd);
        ASD_LOG(ERROR) << "open: " << filename << "fail";
    }

    shmid = shmget(ftok(filename, 'g'), size, IPC_CREAT | 0660);
    if (-1 == shmid) {
        ASD_LOG(ERROR) << "shmget err";
        close(fd);
        return NULL;
    }

    memory = shmat(shmid, NULL, 0);
    if ((void *)-1 == memory) {
        ASD_LOG(ERROR) << "shmmat err";
        close(fd);
        return NULL;
    }

    shmctl(shmid, IPC_STAT, &buf);

    if (buf.shm_nattch == 1) {
        memset(memory, 0, size);
    }

    return memory;
}

CShareMemory::~CShareMemory()
{
    shmdt(share_memory);
    sem_close(sem);
}
