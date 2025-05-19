#ifndef CUSTOMIZE_OPPARAM_H
#define CUSTOMIZE_OPPARAM_H
#include <cstdint>
#include <string>
#include <limits>
#include <hccl/hccl_types.h>
#include <acl/acl.h>
#include "atb/svector.h"

//!
//! \file customize_op_params.h
//!
//! \brief 定义加速库所有用户自定义算子参数
//!

//!
//! \namespace atb
//!
//! \brief 加速库命名空间.
//!
namespace atb {

namespace customize {

//!
//! \struct AddParam
//!
//! \brief 两个张量相加
//!
struct AddParam {
    //!
    //! \brief 占位参数
    //!
    //! \note 默认值为0。
    //!
    uint32_t paramForNothing = 0;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[64] = {0};
};
} // namespace customize
} // namespace atb
#endif