#ifndef CUSTOMIZE_PARAMS_BLOCKCOPY_H
#define CUSTOMIZE_PARAMS_BLOCKCOPY_H

namespace AtbOps {
namespace OpParam {
struct CustomizeBlockCopy {
    enum Type {
        BLOCK_COPY_CACHE_ND = 0,
        BLOCK_COPY_CACHE_NZ = 1
    };
    Type type = BLOCK_COPY_CACHE_ND;
    bool operator==(const BlockCopy &other) const
    {
        return this->type == other.type;
    }
};
} // namespace OpParam
} // namespace AtbOps

#endif // ATBOPS_PARAMS_BLOCKCOPY_H