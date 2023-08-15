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
        设置npu id（不设默认为0）：
        ```sh
        export SET_NPU_DEVICE=0
        ```
        进入chatglm6b模型文件夹：
        ```sh
        cd [加速库根目录]/examples/chatglm6b
        ```
        使用run.sh文件启动模型。run.sh脚本指令如下：<br>
        ```sh
        bash run.sh [model script path] [--run|--performance|--webdemo|--zhipu|--profiling]
        # --run:            问答模式启动模型
        # --performance:    模型性能测试，测试方法参考run.sh文件备注
        # --webdemo:        启动模型webUI
        # --zhipu:          智谱标准模型性能测试
        # --profiling:      启动模型并开启profiling
        ```

+ llama 环境准备
  