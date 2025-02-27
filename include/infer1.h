//!
//! \struct LayerNormParam
//!
//! \brief LayerNorm归一化处理。当前支持三种：NORM、PRENORM、POSTNORM。
//!
//! \warning beginNormAxis维度小于等于输入x的维度。
//! 所有输入输出Tensor的最后一维大小相等。
//! Atlas 推理系列产品中不支持bf16类型数据。
//!
struct LayerNormParam {
    //!
    //! \enum LayerNormType
    //!
    //! \brief 归一化类型：NORM、PRENORM、POSTNORM。
    //!
    enum LayerNormType : int {
        LAYER_NORM_UNDEFINED = 0, //!< 默认值，未定义
        LAYER_NORM_NORM,          //!< norm
        LAYER_NORM_PRENORM,       //!< prenorm
        LAYER_NORM_POSTNORM,      //!< postnorm
        LAYER_NORM_MAX,
    };
    //!
    //! \brief NORM参数。
    //!
    struct NormParam {
        //! \brief 量化类型。
        //! 当前支持以下类型。
        //! QUANT_UNDEINFED；
        //! QUANT_INT8
        QuantType quantType = QUANT_UNDEFINED;
        //! \brief Epsilon，归一化时加在分母上防止除零。
        float epsilon = 1e-5;
        //! \brief 归一化的维度，默认值为0，从第几维开始norm，同时决定输入gamma和beta维度。
        int32_t beginNormAxis = 0;
        //! \brief 归一化的维度，默认值为0，决定从第几维开始把后面的维度按轴合并。
        int32_t beginParamsAxis = 0;
        //! \brief 动态量化类型。默认为DYNAMIC_QUANT_UNDEFINED非动态量化。当前版本暂不支持非对称动态量化。
        DynamicQuantType dynamicQuantType = DYNAMIC_QUANT_UNDEFINED;
        //!
        //! \brief 预留参数
        //!
        uint8_t rsv[20] = {0};
    };
    //!
    //! \brief PRENORM参数
    //!
    struct PreNormParam {
        //! \brief 量化类型。
        //! 当前仅支持QUANT_UNDEINFED。
        QuantType quantType = QUANT_UNDEFINED;
        //! \brief Epsilon，归一化时加在分母上防止除零。
        float epsilon = 1e-5;
        //! \brief 0：高精度 1：高性能（暂不支持）。
        uint64_t opMode = 0;
        //! \brief 缩放因子。
        float zoomScaleValue = 1.0f;
        //!
        //! \brief 预留参数
        //!
        uint8_t rsv[20] = {0};
    };
    //!
    //! \brief POSTNORM参数。
    //!
    struct PostNormParam {
        //! \brief 量化类型。
        //! 当前支持以下类型。
        //! QUANT_UNDEINFED；
        //! QUANT_INT8
        QuantType quantType = QUANT_UNDEFINED;
        //! \brief Epsilon，归一化时加在分母上防止除零。
        float epsilon = 1e-5;
        //! \brief 0：高精度 1：高性能（暂不支持）。
        uint64_t opMode = 0;
        //! \brief 缩放因子。
        float zoomScaleValue = 1.0f;
        //!
        //! \brief 预留参数
        //!
        uint8_t rsv[20] = {0};
    };
    //! \brief layerType
    LayerNormType layerType = LAYER_NORM_UNDEFINED;
    //! \brief normParam
    NormParam normParam;
    //! \brief preNormParam
    PreNormParam preNormParam;
    //! \brief postNormParam
    PostNormParam postNormParam;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[8] = {0};
};

//!
//! \struct RmsNormParam
//!
//! \brief RMS归一化处理。
//!
//! \warning 所有输入输出Tensor的最后一维大小相等。
//! Atlas 推理系列产品中不支持bf16类型数据。
//!
struct RmsNormParam {
    //!
    //! \brief RmsNormType
    //!
    enum RmsNormType : int {
        RMS_NORM_UNDEFINED = 0, //!< 默认值，未定义
        RMS_NORM_NORM,          //!< NORM参数。
        RMS_NORM_PRENORM,       //!< PRENORM参数。
        RMS_NORM_POSTNORM,      //!< POSTNORM参数
    };
    //!
    //! \brief PrecisionMode
    //!
    enum PrecisionMode : int {
        HIGH_PRECISION_MODE = 0, //!< 中间计算使用float类型
        HIGH_PERFORMANCE_MODE,   //!< 中间计算使用float16类型
    };
    //!
    //! \brief ModelType
    //!
    enum ModelType : int {
        LLAMA_MODEL = 0, //!< 默认值，使用Llama rmsnorm的公式
        GEMMA_MODEL,     //!< 使用Gemma rmsnorm的公式
    };
    //!
    //! \brief NormParam
    //!
    struct NormParam {
        //! \brief 量化类型。
        //! 当前支持以下类型。
        //! QUANT_UNDEINFED, QUANT_INT8
        QuantType quantType = QUANT_UNDEFINED;
        //! \brief Epsilon，归一化时加在分母上防止除零。
        float epsilon = 1e-5;
        //! \brief Epsilon，默认为1e-5，暂时不使用。
        double layerNormEps = 1e-5;
        //! \brief 默认为False，设置为true时会使用训练的rmsnormforward算子。仅在Atlas 800I A2推理产品上支持该设置。
        //!  不支持和“precisionMode”，“modelType”同时设置。量化场景下不支持使用“rstd”。
        bool rstd = false;
        //! \brief 默认为HIGH_PRECISION_MODE。
        //! 支持参数如下：
        //! HIGH_PRECISION_MODE：默认值，中间计算使用float类型
        //! HIGH_PERFORMANCE_MODE： 中间计算使用float16类型
        //! 不支持和“rstd”，“modelType”同时设置。输入类型只支持float16。
        //! 量化场景下不支持使用“precisionMode”，该场景下配置该参数将返回报错ERROR_INVALID_PARAM。
        PrecisionMode precisionMode = HIGH_PRECISION_MODE;
        //! \brief 默认为LLAMA_MODEL，设置为GEMMA_MODEL时使用gemma模型的rmsnorm计算公式。
        //! 支持参数如下：
        //! LLAMA_MODEL：默认值， Llama的rms norm计算公式。
        //! GEMMA_MODEL：Gemma的rms norm计算公式。
        //! 不支持和“rstd”，“precisionMode”同时启用。
        //! 量化场景下不支持使用“modelType”，该场景下配置该参数将返回报错ERROR_INVALID_PARAM。
        ModelType modelType = LLAMA_MODEL;
        //! \brief 动态量化类型。默认为DYNAMIC_QUANT_UNDEFINED非动态量化。当前版本暂不支持非对称动态量化。
        DynamicQuantType dynamicQuantType = DYNAMIC_QUANT_UNDEFINED;
        //!
        //! \brief 预留参数
        //!
        uint8_t rsv[32] = {0};
    };
    //!
    //! \brief PreNormParam
    //!
    struct PreNormParam {
        //! \brief 量化类型。
        //! 当前支持以下类型。
        //! QUANT_UNDEINFED
        //! QUANT_INT8
        QuantType quantType = QUANT_UNDEFINED;
        //! \brief Epsilon，归一化时加在分母上防止除零。
        float epsilon = 1e-5;
        //! \brief 是否叠加偏置。默认为False，当需要输入beta时设置为True。量化场景下不支持使用“hasBias”，该场景下配置该参数将返回报错ERROR_INVALID_PARAM。
        bool hasBias = false;
        //!
        //! \brief 预留参数
        //!
        uint8_t rsv[23] = {0};
    };
    //!
    //! \brief PostNormParam
    //!
    struct PostNormParam {
        //! \brief 量化类型。
        //! 当前仅支持QUANT_UNDEINFED。
        QuantType quantType = QUANT_UNDEFINED;
        //! \brief Epsilon，归一化时加在分母上防止除零。
        float epsilon = 1e-5;
        //! \brief 是否叠加偏置。默认为False，当需要输入beta时设置为True。
        bool hasBias = false;
        //!
        //! \brief 预留参数
        //!
        uint8_t rsv[23] = {0};
    };
    //! \brief 归一化类型，参数如下：
    //! RMS_NORM_UNDEFINED：默认值，未定义。
    //! RMS_NORM_NORM：NORM参数。
    //! RMS_NORM_PRENORM：PRENORM参数。
    //! RMS_NORM_POSTNORM：POSTNORM参数。
    RmsNormType layerType = RMS_NORM_UNDEFINED;
    //! \brief NORM参数。
    NormParam normParam;
    //! \brief PRENORM参数。
    PreNormParam preNormParam;
    //! \brief POSTNORM参数。
    PostNormParam postNormParam;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[8] = {0};
};

//!
//! \struct FillParam
//!
//! \brief 将指定位置设置为value值或者生成一个指定Shape的Tensor并填充为value。
//!
//! \warning 输入x不可以被broadcast。输入mask的元素只能是0或者1，且可以被broadcast。
//!
struct FillParam {
    //! \brief 是否Masked Fill。
    bool withMask = true;
    //! \brief 填充的元素，value是一个只含有一个元素的SVector。
    SVector<float> value;
    //! \brief withMask = false时，表示输出Tensor的Shape。
    SVector<int64_t> outDim;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[8] = {0};
};

//!
//! \struct AllGatherParam
//!
//! \brief 将多个通信卡上的数据按所属rank号的顺序在第一维进行聚合，然后发送到每张卡上.
//!
//! rank、rankSize、rankRoot需满足以下条件:
//! 0 ≤ rank < rankSize, 0 ≤ rankRoot < rankSize
//!
//! \note 1、多用户使用时需要使用ATB_SHARE_MEMORY_NAME_SUFFIX环境变量进行共享内存的区分，以进行初始化信息同步.
//! \note 2、当使用加速库的通信算子异常退出时，需要清空残留数据，避免影响之后的使用，命令参考如下：
//!
//! \code
//!         rm -rf /dev/shm/sem.lccl*
//!         rm -rf /dev/shm/sem.hccl*
//!         ipcrm -a
//! \endcode
//!
struct AllGatherParam {
    //! \brief 当前卡所属通信编号
    int rank = 0;
    //! \brief 通信的卡的数量
    int rankSize = 0;
    //! \brief 主通信编号
    int rankRoot = 0;
    //! \brief 通信后端指示，仅支持"hccl"和"lccl",Atlas 推理系列产品仅支持backend为"hccl"。
    //!
    //! 当backend为"lccl"时，且若机器拓扑为Atlas 800I A2推理产品单机16卡机器的拓扑时，只支持16卡全量拓扑通信或单节点内任意卡通信。
    //!
    std::string backend = "hccl";
    //! \brief HCCL通信域指针
    //! 默认为空，加速库为用户创建;若用户想要自己管理通信域,则需要传入该通信域指针,加速库使用传入的通信域指针来执行通信算子
    HcclComm hcclComm = nullptr;
    //! \brief 通信模式，CommMode类型枚举值。hccl多线程只支持外部传入通信域方式
    CommMode commMode = COMM_MULTI_PROCESS;
    //!
    //! \brief 集群信息的配置文件路径，适用单机以及多机通信场景，当前仅支持hccl后端场景,若单机配置了rankTable，则以ranktable来初始化通信域。
    //!
    std::string rankTableFile;
    //! \brief 通信device组用通信域名标识，多通信域时使用，当前仅支持hccl
    std::string commDomain;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[64] = {0};
};

//!
//! \struct AllReduceParam
//!
//! \brief 将多个通信卡上的数据进行计算，支持相加、取最大、最小、相乘四种计算，然后发送到每张卡上.
//!
//! rank、rankSize、rankRoot需满足以下条件:
//! 0 ≤ rank < rankSize, 0 ≤ rankRoot < rankSize
//!
//! \note 1、多用户使用时需要使用ATB_SHARE_MEMORY_NAME_SUFFIX环境变量进行共享内存的区分，以进行初始化信息同步.
//! \note 2、当使用加速库的通信算子异常退出时，需要清空残留数据，避免影响之后的使用，命令参考如下：
//!
//! \code
//!         rm -rf /dev/shm/sem.lccl*
//!         rm -rf /dev/shm/sem.hccl*
//!         ipcrm -a
//! \endcode
//!
struct AllReduceParam {
    //! \brief 量化类型
    enum QuantType : int {
        QUANT_TYPE_UNDEFINED = 0,   //!< 默认值
        QUANT_TYPE_PER_TENSOR = 1,  //!< 对整个张量进行量化
        QUANT_TYPE_PER_CHANNEL = 2, //!< 对张量中每个channel分别进行量化
        QUANT_TYPE_MAX = 3,         //!< 枚举类型最大值
    };

    //! \brief 当前卡所属通信编号.
    int rank = 0;
    //! \brief 通信的卡的数量.
    int rankSize = 0;
    //! \brief 主通信编号.
    int rankRoot = 0;
    //! \brief 通信计算类型，支持"sum","prod","max"和"min".
    std::string allReduceType = "sum";
    //!
    //! \brief 通信计算类型，仅支持"hccl"和"lccl".Atlas 推理系列产品仅支持backend为"hccl"。
    //!
    //! backend为"hccl"时，支持"sum","prod","max"和"min"; backend为"lccl"时，支持"sum","max"和"min".
    //! 当backend为"hccl"时，allReduceType为"prod"时，不支持数据类型为int16和bf16。
    //! 当backend为"hccl"时，Atlas 推理系列产品不支持int64,bf16,int16只有allReduceType为"sum"时支持
    //! 当backend为"lccl"时，不支持数据类型int64，且若机器拓扑为Atlas 800I A2推理产品单机16卡机器的拓扑时，只支持16卡全量拓扑通信或单节点内任意卡通信。
    //!
    std::string backend = "hccl";
    //! \brief HCCL通信域指针.
    //! 默认为空，加速库为用户创建;若用户想要自己管理通信域,则需要传入该通信域指针,加速库使用传入的通信域指针来执行通信算子
    HcclComm hcclComm = nullptr;
    //! \brief 通信模式，CommMode类型枚举值.hccl多线程只支持外部传入通信域方式
    CommMode commMode = COMM_MULTI_PROCESS;
    //!
    //! \brief 集群信息的配置文件路径，适用单机以及多机通信场景，当前仅支持hccl后端场景,若单机配置了rankTable，则以ranktable来初始化通信域。
    //!
    std::string rankTableFile;
    //! \brief 通信device组用通信域名标识，多通信域时使用。 当前LCCL支持单进程多线程多通信域并行，使用该功能的时候，commMode需要设置为COMM_MULTI_THREAD。
    //! LCCL的多通信域并行新增环境变量"LCCL_PARALLEL"需要设置为true, 该功能和确定性计算无法同时开启，"LCCL_DETERMINISTIC"需要为0或者false。
    std::string commDomain;
    //! \brief 量化类型
    QuantType quantType = QUANT_TYPE_UNDEFINED;
    //! 若为浮点AllReduce，参数outDataType配置为ACL_DT_UNDEFINED，表示输出tensor的数据类型与输入tensor一致；
    //! 若为量化AllReduce，输出tensor的数据类型与输入tensor不一致，则参数outDataType配置为用户预期输出tensor的数据类型，
    //! 量化只支持配置ACL_FLOAT16
    aclDataType outDataType = ACL_DT_UNDEFINED;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[64] = {0};
};

//!
//! \struct BlockCopyParam
//!
//! \brief 将KVCache里通过src indices指定的block数据copy到dst indices指定的block位置上。
//!
struct BlockCopyParam {
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[16] = {0};
};

//!
//! \struct BroadcastParam
//!
//! \brief 将通信主卡上的数据广播到其他每张卡上, 该算子不支持Atlas 推理系列产品。
//!
//! rank、rankSize、rankRoot需满足以下条件:
//! 0 ≤ rank < rankSize, 0 ≤ rankRoot < rankSize
//!
//! \note 1、多用户使用时需要使用ATB_SHARE_MEMORY_NAME_SUFFIX环境变量进行共享内存的区分，以进行初始化信息同步.
//! \note 2、当使用加速库的通信算子异常退出时，需要清空残留数据，避免影响之后的使用，命令参考如下：
//!
//! \code
//!         rm -rf /dev/shm/sem.lccl*
//!         rm -rf /dev/shm/sem.hccl*
//!         ipcrm -a
//! \endcode
//!

struct BroadcastParam {
    //! \brief 当前卡所属通信编号.
    int rank = 0;
    //! \brief 通信的卡的数量.
    int rankSize = 0;
    //! \brief 主通信编号.
    int rankRoot = 0;
    //! \brief HCCL通信域指针.
    //! 默认为空，加速库为用户创建;若用户想要自己管理通信域,则需要传入该通信域指针,加速库使用传入的通信域指针来执行通信算子
    HcclComm hcclComm = nullptr;
    //! \brief 通信模式，CommMode类型枚举值.hccl多线程只支持外部传入通信域方式
    CommMode commMode = COMM_MULTI_PROCESS;
    //! \brief 通信后端指示，仅支持"hccl"和"lccl"。
    std::string backend = "hccl";
    //!
    //! \brief 集群信息的配置文件路径，适用单机以及多机通信场景，当前仅支持hccl后端场景,若单机配置了rankTable，则以ranktable来初始化通信域。
    //!
    std::string rankTableFile;
    //! \brief 通信device组用通信域名标识，多通信域时使用，当前仅支持hccl
    std::string commDomain;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[64] = {0};
};

//!
//! \struct ReduceScatterParam
//!
//!
//! rank、rankSize、rankRoot需满足以下条件:
//! 0 ≤ rank < rankSize, 0 ≤ rankRoot < rankSize
//!
//! \note 1、多用户使用时需要使用ATB_SHARE_MEMORY_NAME_SUFFIX环境变量进行共享内存的区分，以进行初始化信息同步.
//! \note 2、当使用加速库的通信算子异常退出时，需要清空残留数据，避免影响之后的使用，命令参考如下：
//!
//! \code
//!         rm -rf /dev/shm/sem.lccl*
//!         rm -rf /dev/shm/sem.hccl*
//!         ipcrm -a
//! \endcode
//!
struct ReduceScatterParam {
    //! \brief 当前卡所属通信编号.
    int rank = 0;
    //! \brief 通信的卡的数量.
    int rankSize = 0;
    //! \brief 主通信编号.
    int rankRoot = 0;
    //! \brief 当前通信计算类型仅支持"sum","max"和"min",不支持"prod"。
    std::string reduceType = "sum";
    //! \brief HCCL通信域指针。 当前算子仅支持lccl,此参数为预留参数。
    //! 默认为空，加速库为用户创建;若用户想要自己管理通信域,则需要传入该通信域指针,加速库使用传入的通信域指针来执行通信算子。
    HcclComm hcclComm = nullptr;
    //! \brief 通信模式，CommMode类型枚举值。
    CommMode commMode = COMM_MULTI_PROCESS;
    //! \brief 通信后端指示，当前算子仅支持"lccl"，不支持数据类型int64, 若机器拓扑为Atlas 800I A2推理产品单机16卡机器的拓扑时，只支持16卡全量拓扑通信或单节点内任意卡通信。
    std::string backend = "lccl";
    //! \brief 集群信息的配置文件路径。
    std::string rankTableFile;
    //! \brief 通信device组用通信域名标识。
    std::string commDomain;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[64] = {0};
};

//!
//! \struct LinearParam
//!
//! \brief 将A、B两个矩阵进行矩阵乘运算，同时可以选择对矩阵乘的运算结果进行叠加偏置、InplaceAdd融合或反量化操作。
//!
//! \note 算子本质上是接收x和weight两个输入tensor作为A矩阵和B矩阵进行矩阵乘运算，可通过参数transposeA与transposeB控制做矩
//! 阵乘前是否需要对A矩阵和B矩阵进行行列转置，根据参数转置后的A矩阵和B矩阵需满足矩阵乘维度关系。例如，当transposeA为false，
//! transposeB为true时，x和weight的shape可以分别为[m, k]和[n, k]。
//!
//! \note 该算子支持浮点和量化场景，当参数outDataType值为ACL_DT_UNDEFINED时为浮点场景，否则为量化场景。
//!
struct LinearParam {
    //!
    //! \brief 是否转置A矩阵。
    //!
    //! \note 默认值为false，不转置。
    //!
    //! \warning 在量化场景下，非Atlas 800I A2推理产品仅支持配置为false。
    //!
    bool transposeA = false;
    //!
    //! \brief 是否转置B矩阵。
    //!
    //! \note 默认值为true，转置。
    //!
    //! \warning 在量化场景下，非Atlas 800I A2推理产品仅支持配置为true。
    //!
    bool transposeB = true;
    //!
    //! \brief 是否叠加偏置。
    //!
    //! \note 默认值为true，叠加偏置。
    //!
    //! \warning 在量化场景下，非Atlas 800I A2推理产品仅支持配置为true。
    //!
    //! \warning enAccum为true时，仅支持配置为false。
    //!
    bool hasBias = true;
    //!
    //! \brief 输出数据类型。
    //!
    //! \note 默认值为ACL_DT_UNDEFINED。
    //!
    //! \warning 浮点场景下：支持配置为ACL_DT_UNDEFINED。
    //!
    //! \warning 量化场景下：Atlas 800I A2推理产品支持配置为ACL_FLOAT16/ACL_BF16，否则，仅支持配置为ACL_FLOAT16。
    //!
    aclDataType outDataType = ACL_DT_UNDEFINED;
    //!
    //! \brief 是否使能累加。
    //!
    //! \note 默认值为false，不使能累加。
    //!
    //! \warning 仅在Atlas 800I A2推理产品支持配置为true。
    //!
    //! \warning hasBias为true时，仅支持配置为false。
    //!
    //! \warning 量化场景下，仅支持配置为false。
    //!
    bool enAccum = false;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[23] = {0};
};

//!
//! \struct LinearParallelParam
//!
//! \brief 通信计算并行算子,该算子功能为linear和通信算子组合
//!
//! 通信和计算是并行处理，和串行相比存在大幅度性能提升.
//!
//! \see LinearParam,AllReduceParam,AllGatherParam
//!
struct LinearParallelParam {
    //!
    //! \enum ParallelType
    //!
    //! \brief 通信类型
    //!
    enum ParallelType : int {
        UNDEFINED = -1,            //!< 默认值
        LINEAR_ALL_REDUCE = 0,     //!< linear+AllReduce
        LINEAR_REDUCE_SCATTER = 1, //!< linear+reduce_scatter
        ALL_GATHER_LINEAR = 2,     //!< AllGather+linear
        PURE_LINEAR = 3,           //!< linear
        MAX = 4,                   //!< 枚举类型最大值
    };
    //!
    //! \enum QuantType
    //!
    //! \brief QuantType类型
    //!
    enum QuantType : int {
        QUANT_TYPE_UNDEFINED = -1,  //!< 默认值
        QUANT_TYPE_PER_TENSOR = 0,  //!< 对整个张量进行量化
        QUANT_TYPE_PER_CHANNEL = 1, //!< 对张量中每个channel分别进行量化
        QUANT_TYPE_PER_GROUP = 2,   //!< 将张量按quantGroupSize划分后，分别进行量化
        QUANT_TYPE_MAX = 3,         //!< 枚举类型最大值
    };
    //! \brief 权重是否需要转置，默认为true。
    bool transWeight = true;
    //! \brief 当前卡所属通信编号.
    int rank = 0;
    //! \brief 通信的卡的数量
    int rankSize = 0;
    //! \brief 主通信编号
    int rankRoot = 0;
    //! \brief 是否叠加残差。配置为false时不叠加残差，为true时叠加残差。默认不叠加残差。
    bool hasResidual = false;
    //! \brief 通信后端指示。支持"hccl"，"lccl"，"lcoc"。
    std::string backend = "hccl";
    //! \brief HCCL通信域接口获取的地址指针，仅当"hcclComm"不为nullptr时可用。
    HcclComm hcclComm = nullptr;
    //! \brief 通信模式，CommMode类型枚举值
    CommMode commMode = COMM_MULTI_PROCESS;
    //! \brief 集群信息的配置文件路径，适用单机以及多机通信场景，当前仅支持hccl后端场景。
    std::string rankTableFile;
    //! \brief 权重并行类型。
    ParallelType type = LINEAR_ALL_REDUCE;
    //! \brief 是否返回中间结果，仅在使用ALL_GATHER_LINEAR时生效。
    bool keepIntermediate = false;
    //! \brief 量化类型。
    QuantType quantType = QUANT_TYPE_UNDEFINED;
    //! \brief 量化类型为QUANT_TYPE_PER_GROUP时生效。
    int32_t quantGroupSize = 0;
    //!
    //! 若为浮点linear，参数outDataType配置为ACL_DT_UNDEFINED，表示输出tensor的数据类型与输入tensor一致,
    //! 若为量化linear，输出tensor的数据类型与输入tensor不一致，则参数outDataType配置为用户预期输出tensor的数据类型,
    //! 如ACL_FLOAT16/ACL_BF16
    aclDataType outDataType = ACL_DT_UNDEFINED;
    //! \brief 通信device组用通信域名标识，多通信域时使用，当前仅支持hccl
    std::string commDomain;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[64] = {0};
};