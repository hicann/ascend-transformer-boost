/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_INFEROPPARAM_H
#define ATB_INFEROPPARAM_H
#include <cstdint>
#include <string>
#include <limits>
#include <hccl/hccl_types.h>
#include <acl/acl.h>
#include "atb/svector.h"

//!
//! \file infer_op_params.h
//!
//! \brief 定义加速库所有推理算子参数
//!

//!
//! \namespace atb
//!
//! \brief 加速库命名空间.
//!
namespace atb {

namespace infer {

//!
//! \enum InputLayout
//!
//! \brief 数据排布类型
//!
enum InputLayout : int {
    TYPE_BSND = 0, //!< 默认值，表示数据排布为BSND
    TYPE_BNSD      //!< 表示数据排布为BNSD
};

//!
//! \enum QuantType
//!
//! \brief 量化支持的类型
//!
enum QuantType : int {
    QUANT_UNDEFINED = 0, //!< 不量化
    QUANT_INT4,          //!< 当前不支持
    QUANT_INT8,          //!< int8量化
    QUANT_INT16,         //!< 当前不支持
    QUANT_FLOAT8,        //!< 当前不支持
    QUANT_FLOAT16,       //!< 当前不支持
};

//!
//! \enum DynamicQuantType
//!
//! \brief 动态量化支持的类型
//!
enum DynamicQuantType : int {
    DYNAMIC_QUANT_UNDEFINED = 0, //!< 非动态量化
    DYNAMIC_QUANT_SYMMETRIC,     //!< 对称动态量化
    DYNAMIC_QUANT_ASYMMETRIC,    //!< 非对称动态量化，暂不支持
};

//!
//! \enum ActivationType
//!
//! \brief 激活支持的类型
//! ACTIVATION_FAST_GELU：快速运算的Gelu激活函数，对Tensor内每个element做Gelu激活函数近似计算，计算速度更快，同时保持较高的准确性。
//! ACTIVATION_SWIGLU_FORWARD: Swiglu正向激活函数。Atlas 推理系列产品中只支持32位对齐的数据。
//! ACTIVATION_FASTER_GELU_FORWARD: 简化后的FastGelu激活函数，计算速度更快。
//! ACTIVATION_SWIGLU_BACKWARD: Swiglu正向激活函数的反向，求梯度时使用。只支持Atlas 800I A2推理产品。
//!
enum ActivationType : int {
    ACTIVATION_UNDEFINED = 0,       //!< 未定义
    ACTIVATION_RELU,                //!< RELU激活类型
    ACTIVATION_GELU,                //!< GELU激活类型
    ACTIVATION_FAST_GELU,           //!< FAST_GELU激活类型
    ACTIVATION_SWISH,               //!< SWISH激活类型
    ACTIVATION_LOG,                 //!< LOG激活类型
    ACTIVATION_SWIGLU_FORWARD,      //!< SWIGLU_FORWARD激活类型
    ACTIVATION_SWIGLU_BACKWARD,     //!< SWIGLU_BACKWARD激活类型
    ACTIVATION_SIGMOID,             //!< SIGMOID激活类型
    ACTIVATION_FASTER_GELU_FORWARD, //!< FASTER_GELU_FORWARD激活类型
    ACTIVATION_MAX,                 //!< 枚举最大值, 非激活类型
};

//!
//! \enum CommMode
//!
//! \brief 通信算子支持的通信模式.
//!
enum CommMode : int {
    COMM_UNDEFINED = -1, //!< 未定义
    COMM_MULTI_PROCESS,  //!< 指定多进程通信
    COMM_MULTI_THREAD,   //!< 指定多线程通信
};

//!
//! \brief 激活函数。
//!
struct ActivationParam {
    //! \enum GeLUMode
    //! \brief GeLU激活函数可选的计算模式
    enum GeLUMode : int {
        TANH_MODE = 0, //!< 默认值，使用tanh估算
        NONE_MODE,     //!< 原GeLU计算公式
    };
    //! 激活函数类型，ActivationType类型枚举值.
    ActivationType activationType = ACTIVATION_UNDEFINED;
    //! SWISH激活函数的参数.
    float scale = 1.0f;
    //! SWIGLU激活函数的参数.
    int32_t dim = -1;
    //! GeLU模式选择参数
    GeLUMode geluMode = TANH_MODE;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[8] = {0};
};

//!
//! \brief InTensor根据指定参数，生成一个数据重新排布过的OutTensor.
//!
//! \warning 输出y基于输入x的总偏移量要求小于输入x的大小.
//!
struct AsStridedParam {
    //!
    //! \brief OutTensor的shape.
    //!
    //! \warning size的长度要求小于或等于8且各元素要求大于0.
    //!
    SVector<int64_t> size;
    //!
    //! \brief 用于从InTensor推导OutTensor的各维度的步长.
    //!
    //! \warning stride的长度要求与size一致，各元素要求大于或等于0.
    //!
    SVector<int64_t> stride;
    //!
    //! \brief OutTensor内存相对于InTensor内存的偏移，作为常数使用.
    //!
    //! \warning offset的长度要求为1且元素要求大于或等于0.
    //!
    SVector<int64_t> offset;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[8] = {0};
};

//!
//! \brief 后处理累积和计算.
//!
struct CumsumParam {
    //!
    //! \brief 指定axis轴(维度)上计算累加和，只能包含一个轴索引.
    //!
    //! \warning axes的值必须小于输入x的维度数。
    //!
    SVector<int64_t> axes;
    //!
    //! \brief 在某一个轴上的累加结果从第几个元素开始，默认为false.
    //!
    //! \note true：从第一个元素开始（暂不支持） false：从第0个元素开始.
    //!
    bool exclusive = false;
    //!
    //! \brief 正向累加或逆向累加，默认为false.
    //!
    //! \note true：输出逆向累加（暂不支持） false：输出正向累加.
    //!
    bool reverse = false;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[14] = {0};
};

//!
//! \brief 推理的长度大于训练长度时，embedding需要进行特殊处理。
//! 推理长度小于等于训练长度时，不进行插值；推理长度大于训练长度时，放大base动态插值。
//! 将输入的token序列的位置信息positionIds和inv_freq进行外积，再cos/sin运算得到最终的Rotary embedding的结果。
//!
struct DynamicNTKParam {
    //! 选择输出数据类型的参数
    aclDataType outDataType = ACL_DT_UNDEFINED;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[12] = {0};
};

//!
//! \brief 从输入张量中根据索引收集切片，并将这些切片组合成一个新的张量.
//!
struct GatherParam {
    //!
    //! \brief 指定要收集切片的轴。默认值为0.
    //!
    //! \warning 该参数必须大于或等于0
    //!
    int64_t axis = 0;
    //!
    //! \brief  允许从一个batch的每个元素中收集不同的项目，默认值为0.
    //!
    //! \warning 该参数必须大于或等于0,且小于或等于axis.
    //!
    int64_t batchDims = 0;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[16] = {0};
};

//!
//! \brief 采样功能。对最后一个轴进行采样，随机抽取numSamples个值，输出下标。
//!
//! \warning 用户需确保对最后一个轴进行归一化操作。
//!
struct MultinomialParam {
    //!
    //! \brief 随机采样数.
    //!
    //! \warning 小于等于输入张量对应的维度大小，最大为64。
    //!
    uint32_t numSamples = 1;
    //! \brief 随机数种子.
    uint32_t randSeed = 0;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[8] = {0};
};

//!
//! \brief 对输入张量指定维度切成多个张量。
//!
struct SplitParam {
    //!
    //! \brief 指定切分的维度索引
    //!
    //! splitDim须位于输入张量x的维度范围内，即如果x的维度为xDim，则等长切分下splitDim的取值范围为[-xDim, xDim - 1]。
    //! 当splitDim为负数时，其含义是从最高维度开始访问，如splitDim = -1，x维度数为dimNum，则拆分维度为dimNum - 1。
    //! \warning 当使用不等长切分时，splitDim的取值范围为[0, xDim - 1]。
    //!
    int32_t splitDim = 0;
    //!
    //! \brief 切分次数,当前支持2或3.
    //!
    //! \warning 等长切分下输入张量x的维度须能够被splitNum整除,且当splitNum = 3时输入x要求是float16或者bf16数据类型。
    //!
    int32_t splitNum = 2;
    //!
    //! \brief 指定每个输出tensor在切分维度上的大小
    //!
    //! 不传入此参数时使用等长切分，传入此参数时使用splitV不等长切分
    //! \warning splitSizes中的每一个元素要求大于等于1。splitSizes中的元素之和等于切分维度的大小。
    //!
    SVector<int32_t> splitSizes = {};
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[8] = {0};
};

//!
//! \brief 将两个输入张量在指定维度拼接成一个输出张量
//!
struct ConcatParam {
    //!
    //! \brief 指定拼接的维度索引
    //!
    //! 当concatDim为负数时，其含义是从最高维度开始访问，如concatDim = -1，输入张量维度数为dimNum，则拼接维度为dimNum - 1。
    //!
    //! \warning 输入x和y的维数要求一致。输入x或y的维度大小，除了concatDim维外，其他维度要求相同。Atlas 推理系列产品中不支持bf16类型数据。
    //!
    int concatDim = 0;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[12] = {0};
};

//!
//! \brief 从输入张量某个起始位置中提取指定大小的切片
//!
struct SliceParam {
    //!
    //! \brief 每个维度切片的起始位置
    //!
    //! 当offsets[i]为负数时，其含义是第i维最高维度开始访问，如offsets= -1，输入x的维度为dimNum，则对应维度切片的起始位置为dimNum - 1。
    //!
    //! \warning 当offsets元素x小于0时，该元素对应的维度大小为dimNum，要求dimNum与x之和大于等于0。
    //!
    SVector<int64_t> offsets;
    //!
    //! \brief 每个维度切片的大小
    //!
    //! 当size = -1时，表示切片的结束位置是对应维度最后一个位置。如果对应维度大小为dimNum，则结束位置为dimNum - 1。
    //!
    //! \warning size中元素要求大于等于-1。对应维度offsets，以及offsets + size须在x的对应维度的大小范围内。
    //!
    SVector<int64_t> size;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[8] = {0};
};

//!
//! \brief Softmax多分类激活函数，将多维（最大8维）Tensor数据在指定轴上映射到0到1之间，且softmax轴数值之和为1。
//!
struct SoftmaxParam {
    //!
    //! \brief 指定轴（维度），axes可以支持多个轴上进行处理
    //!
    //! \warning axes不能为空，当指定多个轴时，多个轴之间必须连续且从小到大排列。
    //! \warning axes的元素要求大于或等于-1且小于输入x的维度
    //!
    SVector<int64_t> axes;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[8] = {0};
};

//!
//! \brief 改变输入Tensor的排列顺序，在多个维度上进行转置
//!
struct TransposeParam {
    //! 指示输入维度的重排结果, 需要保证输入正确，维度和输入x一致
    SVector<int32_t> perm;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[8] = {0};
};

//!
//! \struct ElewiseParam
//!
//! \brief 常用的逐元素数值计算集合
//!
//! ELEWISE_ADD、ELEWISE_MUL、ELEWISE_REALDIV、ELEWISE_SUB计算类型将会对输入进行广播后再进行指定操作。
//! 输入x、y对应维度的对应值要求相同或至少其中一个为1
//!
struct ElewiseParam {
    //!
    //! \enum ElewiseType
    //!
    //! \brief 计算类型
    //!
    enum ElewiseType : int {
        ELEWISE_UNDEFINED = 0,       //!< 默认值，未定义
        ELEWISE_CAST,                //!< 数据类型转换
        ELEWISE_MULS,                //!< 向量逐元素乘值
        ELEWISE_COS,                 //!< 逐元素计算余弦值
        ELEWISE_SIN,                 //!< 逐元素计算正弦值
        ELEWISE_NEG,                 //!< 逐元素取相反数
        ELEWISE_QUANT,               //!< 量化, 仅在Atlas 800I A2推理产品上支持
        ELEWISE_LOGICAL_NOT,         //!< 逐元素逻辑非
        ELEWISE_ADD,                 //!< 逐元素相加
        ELEWISE_MUL,                 //!< 向量与向量逐元素相乘
        ELEWISE_REALDIV,             //!< 向量与向量逐元素相除
        ELEWISE_LOGICAL_AND,         //!< 逐元素逻辑与
        ELEWISE_LOGICAL_OR,          //!< 逐元素逻辑或
        ELEWISE_LESS,                //!< 逐元素判断是否小于
        ELEWISE_GREATER,             //!< 逐元素判断是否大于
        ELEWISE_SUB,                 //!< 逐元素相减
        ELEWISE_EQUAL,               //!< 逐元素判断是否相等
        ELEWISE_QUANT_PER_CHANNEL,   //!< 每个通道量化
        ELEWISE_DEQUANT_PER_CHANNEL, //!< 每个通道反量化
        ELEWISE_DYNAMIC_QUANT,       //!< 逐行动态量化
        ELEWISE_TANH,                //!< 逐元素计算双曲正切值
        ELEWISE_TYPE_MAX             //!< 边界值，仅用于判断是否出界，所有情况不能取该值。
    };

    //! 量化（非每通道）所需参数
    struct QuantParam {
        //! 量化的步长
        float inputScale = 1.0f;
        //! 动态量化的是否为非对称量化
        bool asymmetric = false; //!< false : symmetric，true : asymmetric
        //! 量化的偏移度
        int inputOffset = 0;
        //!
        //! \brief 预留参数
        //!
        uint8_t rsv[20] = {0};
    };

    //! 向量乘值所需参数
    struct MulsParam {
        //! 向量乘的值
        float varAttr = 0.0f;
        //!
        //! \brief 预留参数
        //!
        uint8_t rsv[12] = {0};
    };

    //! 计算方式
    ElewiseType elewiseType = ELEWISE_UNDEFINED;
    //! 量化参数
    QuantParam quantParam;
    //! 乘值参数
    MulsParam mulsParam;
    //! 指定数据类型转换输出的数据类型
    aclDataType outTensorType = ACL_DT_UNDEFINED;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[8] = {0};
};

//!
//! \struct KvCacheParam
//!
//! \brief KVCache处理。
//!
struct KvCacheParam {
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[8] = {0};
};

//!
//! \struct GatingParam
//!
//! \brief 主要功能为将token和专家的映射关系反转为专家与token的映射关系。算子输入为MoE模型每个token选中专家的索引，算子输出为MoE模型每个专家对应的token的索引。
//!
//! \note 该算子支持TP和EP场景，当参数deviceExpert为空时，为TP场景，否则为EP场景。
//!
//! \warning 非Atlas 800I A2推理产品仅支持TP场景。
//!
struct GatingParam {
    //!
    //! \brief 每个token选中的专家数。
    //!
    //! \note 默认值为0。
    //!
    //! \warning 当cumSumNum为0时，取值为1；否则，取值范围为(0, cumSumNum]。
    //!
    int32_t topkExpertNum = 0;
    //!
    //! \brief 专家总数。
    //!
    //! \note 默认值为0。
    //!
    //! \warning 取值范围为[0, 200]。
    //!
    int32_t cumSumNum = 0;
    //!
    //! \brief 输出的cumSum的类型是否为int64。
    //!
    //! \note 默认值为false。
    //!
    //! \warning 当为false时，输出的cumSum类型为int32.
    //!
    bool cumSumInt64 = false;
    //!
    //! \brief 当前device上的专家索引列表。
    //!
    //! \note 默认为空。
    //!
    //! \warning 列表中各个元素取值范围为[0, cumSumNum)，且其中元素值不可重复。
    //!
    //! \warning 当cumSumNum为0时，不可为空。
    //!
    std::vector<int32_t> deviceExpert;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[16] = {0};
};

//!
//! \brief 遍历每个key和value，将key和value(num_heads, head_size)按照slotmapping填入key_cache/value_cache指定位置
//!
struct ReshapeAndCacheParam {
    //!
    //! \enum CompressType
    //!
    //! \brief 压缩类型
    //!
    //! \note 默认值为COMPRESS_TYPE_UNDEFINED(0)，不开启压缩功能。
    //!
    //! \warning 仅在Atlas 800I A2推理产品上支持设置为非COMPRESS_TYPE_UNDEFINED(0)的值
    //!
    enum CompressType : int {
        COMPRESS_TYPE_UNDEFINED = 0, //!< 默认值，不压缩
        COMPRESS_TYPE_KVHEAD,        //!< alibi场景下压缩key_cache, value_cahe的kvHead维度
        COMPRESS_TYPE_KVHEAD_ROPE    //!< rope场景下压缩key_cache, value_cahe的kvHead维度
    };
    //!
    //! \enum KvCacheCfg
    //!
    //! \brief KvCache配置
    //!
    //! \note 默认值为K_CACHE_V_CACHE(0)，传入key_cache和value_cache
    //!
    //! \warning 仅在Atlas 800I A2推理产品上支持设置为K_CACHE_V_BYPASS(1)
    //!
    enum KvCacheCfg : int {
        K_CACHE_V_CACHE = 0, //!< 默认值,传入key_cache和value_cache
        K_CACHE_V_BYPASS,    //!< 只传入key_cache
    };

    //! 压缩方式
    CompressType compressType = COMPRESS_TYPE_UNDEFINED;
    //! kvcache配置
    KvCacheCfg kvCacheCfg = K_CACHE_V_CACHE;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[16] = {0};
};