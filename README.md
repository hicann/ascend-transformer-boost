# acltransformer

### 介绍
{**以下是 Gitee 平台说明，您可以替换此简介**
Gitee 是 OSCHINA 推出的基于 Git 的代码托管平台（同时支持 SVN）。专为开发者提供稳定、高效、安全的云端软件开发协作平台
无论是个人、团队、或是企业，都能够用 Gitee 实现代码托管、项目管理、协作开发。企业项目请看 [https://gitee.com/enterprises](https://gitee.com/enterprises)}

### 软件架构
软件架构说明


### 安装教程

1.  xxxx
2.  xxxx
3.  xxxx

### 使用说明

1.  xxxx
2.  xxxx
3.  xxxx

### 参与贡献

1.  Fork 本仓库
2.  新建 Feat_xxx 分支
3.  提交代码
4.  新建 Pull Request


### 特技

1.  使用 Readme\_XXX.md 来支持不同的语言，例如 Readme\_en.md, Readme\_zh.md
2.  Gitee 官方博客 [blog.gitee.com](https://blog.gitee.com)
3.  你可以 [https://gitee.com/explore](https://gitee.com/explore) 这个地址来了解 Gitee 上的优秀开源项目
4.  [GVP](https://gitee.com/gvp) 全称是 Gitee 最有价值开源项目，是综合评定出的优秀开源项目
5.  Gitee 官方提供的使用手册 [https://gitee.com/help](https://gitee.com/help)
6.  Gitee 封面人物是一档用来展示 Gitee 会员风采的栏目 [https://gitee.com/gitee-stars/](https://gitee.com/gitee-stars/)

### 模型环境构建
+ docker容器准备<br>
    使用 scripts/docker_util.sh 脚本下载docker镜像、启动docker容器；<br>
    docker_util.sh 脚本指令：
    >   pull_chatglm_image：下载并加载chatglm镜像；<br>
    >   pull_llama_image：下载并加载llama镜像；<br>
    >   run_chatglm_container：创建chatglm容器；<br>
    >   run_llama_container：创建llama容器；
    
    脚本参数：
    >   --container_name=：设置docker容器名；<br>
    >   --devid=：设置docker挂载的device id；
    
    创建容器后，通过
    ```sh
    docker exec -it --user root [容器名] bash
    ```
    指令进入容器。
+ chatglm环境准备
  - 代理设置<br>
    进入docker后，首先配置docker内代理以连接网络：
    ```sh
    export http_proxy="http://[代理IP]:[代理端口]"
    export https_proxy=$http_proxy
    ```
  - CANN 环境准备<br>
    如果是910B环境，需下载并安装适配910B的CANN toolkit；
    若为910A，则直接设置CANN环境变量即可：
    ```sh
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    ```
  - 加速库编译<br>
    拉取最新加速库代码并编译加速库，设置加速库环境变量：
    ```sh
    git clone https://gitee.com/ascend/ascend-transformer-acceleration.git
    cd ascend-transformer-acceleration
    bash scripts/build.sh examples
    cd output/acltransformer
    source set_env.sh
    ```
  - 模型执行<br>
    首先进入加速库chatglm模型文件夹：
    ```sh
    cd [加速库文件夹]/models/chatglm6b
    ```
      * 设置环境变量采用ppmatmul算子替换原matmul算子：
        ```sh
        export ASCEND_MATMUL_PP_FLAG=1
        ```
      * 修改config.json决定需要执行的模型脚本文件：<br>
        找到 config.json 文件中的这两行：
        >   "AutoModel": "modeling_chatglm.ChatGLMForConditionalGeneration",<br>
        >   "AutoModelForSeq2SeqLM": "modeling_chatglm.ChatGLMForConditionalGeneration"
        
        修改```modeling_chatglm```为想要执行的模型脚本名。
      * 模型文件包括名为```modeling_chatglm*```的所有文件，其中```modeling_chatglm.py```为原始模型文件，其余为替换部分operation、layer的模型文件。<br>
        不带```performance```后缀的为验证脚本，脚本会对比原生PTA算子与调用加速库两种途径的结果，验证加速库的正确性；<br>
        带```performance```后缀的为性能验证脚本，其中加速库替换部分不会再执行原生pta算子，并且在脚本中添加打点以输出模型执行时间。
      * 模型入口脚本:
          + main.py<br>
            原始入口脚本，能用来执行所有模型文件，不会打印性能打点结果。
          + main_performance.py<br>
            性能验证入口脚本，只能用来执行添加打点的模型脚本（目前只有```modeling_chatglm.py```和```modeling_chatglm_layer_performance.py```）,输出性能打点结果。<br>
            性能测试时，第一次需输入一个短句对模型进行warm up，第一个问题不参与计时，之后输入```clear```清空history，第二个问题开始计时并打印结果。脚本默认执行64个token后结束运行。
          + run.sh和run_performance.sh<br>
            创建模型权重文件的软连接并执行main.py或main_performance.py。**当提示找不到模型权重文件时，说明没有对权重文件创建软连接。**
          + rand_tensor_performance.py<br>
            随机tensor性能测试入口脚本，打点在入口脚本内，所以能执行所有模型文件。<br>
            rand_tensor_performance.py 脚本指令：
            >   full：仅进行全量测试；<br>
            >   full_and_incremental：进行全量+增量测试，默认执行该测试；
            
            脚本参数：
            >   seq_len=：设置输入seq_len大小，默认为512；<br>
            >   batch=：设置输入batch大小，默认为8；<br>
            >   test_cycle=：设置测试的token数，默认为100，全量+增量测试时，执行1次全量测试与99次增量测试；<br>
            >   device_id=：设置device id，默认为0；

          + main_web.py<br>
            WebUI入口脚本，执行后生成一个web demo，可在网页上与模型之间进行问答对话。<br>
            执行main_web.py需要额外安装两个python库：
            ```sh
            pip install gradio
            pip install mdtex2html
            ```
            安装成功后关闭网络代理，否则无法正常监听端口：
            ```sh
            unset http_proxy
            unset https_proxy
            ```
            之后执行脚本即可。
      * 设置device_id<br>
        在想要执行的入口脚本中，找到如下代码：
        ```python
        device_id = 0
        torch.npu.set_device(torch.device(f"npu:{device_id}"))
        ```
        修改```device_id```即可。一般情况下，docker中只会挂载一个device设备，所以不用进行修改。
  

+ llama 环境准备
  
