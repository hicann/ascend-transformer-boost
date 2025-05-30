#include "c_interface_utils.h"
#include "atb/utils/config.h"
#include "atb/utils/singleton.h"
using namespace atb;
using namespace atb::cinterfaceTest;

TEST(TestATBACL, TestMLAPreFillM0C2C1)
{
    if (!GetSingleton<Config>().Is910B()) {
        exit(0);
    }
    atb::Context *context = nullptr;
    aclrtStream stream = nullptr;
    int64_t deviceId = 0;
    int64_t batch = 4;
    Init(&context, &stream, &deviceId);
    uint8_t *inoutHost[MLAPPREFILLINOUT];
    uint8_t *inoutDevice[MLAPPREFILLINOUT];
    aclTensor *tensorList[MLAPPREFILLINOUT];
    size_t inoutSize[MLAPPREFILLINOUT] = {
        numTokens * kvHeads * embeddimV * sizeofFP16,
        numTokens * kvHeads * 64 * sizeofFP16,
        batch * maxSeqLen * kvHeads * embeddimV * sizeofFP16,
        batch * maxSeqLen * kvHeads * 64 * sizeofFP16,
        batch * maxSeqLen * kvHeads * embeddimV * sizeofFP16,
        batch * sizeof(int),
        batch * sizeof(int),
        512 * 512 * sizeofFP16,
        numTokens * kvHeads * embeddimV * sizeofFP16,
    };
    CreateInOutData(9, inoutHost, inoutDevice, inoutSize);
    size_t i = 0;
    aclDataType inputFormat = ACL_FLOAT16;
    // 0
    std::vector<std::vector<int64_t>> viewDim = {
        {numTokens, kvHeads, embeddimV},         // q
        {numTokens, kvHeads, 64},                // qrope
        {batch, maxSeqLen, kvHeads * embeddimV}, // k
        {batch, maxSeqLen, kvHeads * 64},        // kRope
        {batch, maxSeqLen, kvHeads * embeddimV}, // v
        {batch},                                 // qSeqLen
        {batch},                                 // kvSeqLen
        {512, 512},                              // mask
        {numTokens, kvHeads, embeddimV},         // attenOut
    };
    while (i < viewDim.size()) {
        if (i == 5 || i == 6) {
            std::vector<int32_t> seqlen = {4, 4, 4, 4};
            CreateACLTensorInOut(viewDim[i], ACL_INT32, ACL_FORMAT_ND, tensorList, i, seqlen.data());
        } else {
            CreateACLTensorInOut(viewDim[i], inputFormat, ACL_FORMAT_ND, tensorList, i, inoutDevice[i]);
        }
    }
    uint64_t workspaceSize = 0;
    atb::Operation *op = nullptr;
    Status ret = AtbMLAPreFillGetWorkspaceSize(tensorList[0], tensorList[1], tensorList[2], tensorList[3],
                                               tensorList[4], tensorList[5], tensorList[6], tensorList[7], kvHeads, 1.0,
                                               kvHeads, 2, 1, tensorList[8], &workspaceSize, &op, context);
    EXPECT_EQ(ret, atb::NO_ERROR);
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        EXPECT_EQ(ret, ACL_ERROR_NONE);
    }
    ret = AtbMLAPreFill(workspaceAddr, workspaceSize, op, context);
    EXPECT_EQ(ret, atb::NO_ERROR);

    ret = aclrtSynchronizeStream(stream);
    EXPECT_EQ(ret, ACL_ERROR_NONE);

    if (workspaceSize > 0) {
        EXPECT_EQ(aclrtFree(workspaceAddr), ACL_ERROR_NONE);
    }
    EXPECT_EQ(atb::DestroyOperation(op), NO_ERROR);
    Destroy(&context, &stream);
    for (i = 0; i < MLAPPREFILLINOUT; i++) {
        aclrtFreeHost(inoutHost[i]);
        aclrtFree(inoutDevice[i]);
    }
}