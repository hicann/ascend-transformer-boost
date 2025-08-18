#include <unordered_map>
#include <lcal_comm.h>
#include "lcal_internal.h"

#include <chrono>
#include <vector>
#include <mutex>
#include <map>
#include <set>
#include <thread>
#include <sstream>
#include <iomanip>

#include <hccl/hccl.h>
#include "mki/utils/log/log.h"
#include "mki/utils/env/env.h"
#include "tools/socket/lcal_sock_excahnge.h"

#include "runtime/kernel.h"
#include "runtime/mem.h"
#include "runtime/dev.h"
#include "runtime/rt_ffts.h"
#include "profiling/report_timing.h"

constexpr int AI_CORE_NUM_24 = 24;
constexpr int AI_CORE_NUM_20 = 20;
constexpr int AI_CORE_NUM_2 = 2;

enum TopologyType : int {
    TOPOLOGY_HCCS = 0,
    TOPOLOGY_PIX,
    TOPOLOGY_PIB,
    TOPOLOGY_PHB,
    TOPOLOGY_SYS,
    TOPOLOGY_SIO,
    TOPOLOGY_HCCS_SW
};

using namespace std;
using namespace chrono;
using namespace Mki;

namespace Lcal {
constexpr int HCCL_IPC_PID_ARRAY_SIZE = 1;
constexpr int LCAL_INIT_TIMEOUT = 600;

static map<string, GM_ADDR [LCAL_MAX_RANK_SIZE]> g_localPeerMemMap;
static map<string, int[LCAL_MAX_RANK_SIZE]> g_devList;
static std::mutex g_mtx;

static const std::unordered_map<std::string, ChipName> CHIP_MAP = {
    {"Ascend310P", ChipName::CHIP_3010P3},
    {"Ascend910B1", ChipName::CHIP_910B1},
    {"Ascend910B2", ChipName::CHIP_910B2},
    {"Ascend910B2C", ChipName::CHIP_910B2C},
    {"Ascend910B3", ChipName::CHIP_910B3},
    {"Ascend910B4", ChipName::CHIP_910B4},
    {"Ascend910B4-1", ChipName::CHIP_910B41},
    {"Ascend910_9391", ChipName::CHIP_910_9391},
    {"Ascend910_9381", ChipName::CHIP_910_9381},
    {"Ascend910_9392", ChipName::CHIP_910_9392},
    {"Ascend910_9382", ChipName::CHIP_910_9382},
    {"Ascend910_9372", ChipName::CHIP_910_9372},
    {"Ascend910_9361", ChipName::CHIP_910_9361},
    {"Ascend910_9362", ChipName::CHIP_910_9362}
}

ChipName GetChipName()
{
    static curChipName = ChipName::RESERVED;
    if (curChipName != ChipName::RESERVED) {
        return curChipName;
    }
    constexpr int socVerLength = 100;
    char ver[socVerLength];
    auto ret - rtGetSocVersion(ver, socVerLength);
    if (ret != RT_ERROR_NONE) {
        MKI_LOG(ERROR) << "rtGetSocVersion failed, not sure whether the function is normal, please use it with caution";
        return ChipName::RESERVED;
    }
    string chipName(ver);
    MKI_LOG(DEBUG) << "rtGetSocVersion: -- The result after converting ver to string is :" << chipName;

    auto it = CHIP_MAP.find(chipName);
    if (it != CHIP_MAP.end()) {
        curChipName = it->second;
    } else {
        MKI_LOG(WARN) << "There is no commitment to the supported chip types yet," <<
            " and it is not certain whether the functions will work properly."
    }
    return curChipName;    
}

uint32_t GetCoreNum(ChipName chipName)
{
    switch (chipName) {
        case ChipName::CHIP_910B1:
        case ChipName::CHIP_910B2:
        case ChipName::CHIP_910_9391:
        case ChipName::CHIP_910_9381:
        case ChipName::CHIP_910_9392:
        case ChipName::CHIP_910_9382:
        case ChipName::CHIP_910B2C:
            return AI_CORE_NUM_24;
        case ChipName::CHIP_910B3:
        case ChipName::CHIP_910B4:
        case ChipName::CHIP_910B41:
        case ChipName::CHIP_910_9372:
        case ChipName::CHIP_910_9361:
        case ChipName::CHIP_910_9362:
        case ChipName::CHIP_910A5:
            return AI_CORE_NUM_20;
        case ChipName::CHIP_3010P3:
            return AI_CORE_NUM_2;
        default:
            MKI_LOG(ERROR) << "Unknown chip name";
            return 0;
    }
}

bool SkipUnusedChannel910B2C(int curRank, int peerRank, ChipName chipName)
{
    if (chipName == ChipName::CHIP_910B2C) {
        constexpr int rankSizePerNode = 8;
        if ((curRank / rankSizePerNode) != (peerRank / rankSizePerNode))
            && (std::abs(curRank - peerRank) != rankSizePerNode)) {
            return true;
        }
    }
    return false;
}

int LcalComm::InitDumpAddr()
{
    constexpr uint32_t dumpCoreCnt = 75;
    constexpr uint32_t dumpSizePerCore = 1 * 1024 * 1024;
    constexpr uint32_t dumpWorkspaceSize = dumpCoreCnt * dumpSizePerCore;
    GM_ADDR dumpAddr = nullptr;
    int ret = 0;
    ret = aclrtMalloc(reinterpret_cast<void **>(&dumpAddr), dumpWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        MKI_LOG(ERROR) << "aclrtMalloc err " << __LINE__;
        return LCAL_ERROR_INTERNAL;
    }
    aclrtMemset(dumpAddr, dumpWorkspaceSize, 0, dumpWorkspaceSize);

    GM_ADDR memory = static_cast<GM_ADDR>(std::malloc(dumpWorkspaceSize));
    if (!memory) {
        MKI_LOG(ERROR) << "std::malloc err " << __LINE__;
        return LCAL_ERROR_INTERNAL;
    }
    errno_t result = memcpy_s(memory, dumpWorkspaceSize, 0, dumpWorkspaceSize);
    if (result != 0) {
        MKI_LOG(ERROR) << "memcpy_s err " << __LINE__;
    }
    for (uint32_t i = 0; i < dumpCoreCnt; i++) {
        GM_ADDR block_start = memory + i * dumpSizePerCore;
        GM_ADDR deviceBlockStart = dumpAddr + i * dumpSizePerCore;

        LcclDumpBlockInfo* block_info = reinterpret_cast<LcclDumpBlockInfo*>(block_start);
        block_info->len = dumpSizePerCore;
        block_info->core = i;
        block_info->blockNum = 0;
        block_info->dumpOffset = dumpSizePerCore - sizeof(LcclDumpBlockInfo);
        block_info->magic = 0;
        block_info->dumpAddr = reinterpret_cast<uint64_t>(deviceBlockStart + sizeof(LcclDumpBlockInfo));
    }

    ret = aclrtMemcpy(dumpAddr, dumpWorkspaceSize, memory, dumpWorkspaceSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        MKI_LOG(ERROR) << "aclrtMemcpy err " << __LINE__ << " " << ret;
        return LCAL_ERROR_INTERNAL;
    }
    std::free(memory);

    commArgs_.dumpAddr = dumpAddr;
    return LCAL_SUCCESS;
}

int LcalComm::SyncCommArgs()
{
    commArgs_.rank = rank_;
    commArgs_.localRank = localRank_;
    commArgs_.rankSize = rankSize_;
    commArgs_.localRankSize = localRankSize_;
    for (int i = 0; i < rankSize_; i++) {
        commArgs_.peerMems[i] = peerMems_[i];
    }

    if (isEnableMsprofOp_ && InitDumpAddr() != LCAL_SUCCESS) {
        return LCAL_ERROR_INTERNAL;
    }

    if (isEnableMix_) {
        uint64_t fftsVal = 0;
        uint32_t fftsLen = 0;
        int error = rtGetC2cCtrlAddr(&fftsVal, &fftsLen);
        if (error != RT_ERROR_NONE) {
            MKI_LOG(ERROR) << "rtGetC2cCtrlAddr err:" << error;
            return LCAL_ERROR_MKIRT;
        }
        commArgs_.fftsVal = fftsVal;
    }

    int ret = 0;
    ret = aclrtMalloc(reinterpret_cast<void **>(&commArgsPtr_), sizeof(commArgs_), ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        MKI_LOG(ERROR) << "aclrtMalloc err " << __LINE__ << " " << ret;
        return LCAL_ERROR_INTERNAL;
    }
    
}
}

