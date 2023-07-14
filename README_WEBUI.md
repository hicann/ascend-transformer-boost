# 零、代理 / 权限

## 1. 连外网代理

```
export http_proxy="http://90.253.26.55:6688"
export https_proxy=$http_proxy
```

## 2. GIT CLONE 失败，报错 SSL
```
export GIT_SSL_NO_VERIFY=true
git config --global http.sslVerify "false"
### by zhengchenjun，如这个代理不行，可直接找 zhengchenjun 要一个新
```

---

# 一、所需文件

## 1. 下载镜像文件 [ chatglm6b_wmj_image.tar ]

### 1.1. 已保存在 **`/data/acltransformer_testdata/images/`** 里（建议不要改动！！）

#### 1.1.1. 没有找到？

  * 打开浏览器输入：
  **`https://ascend-transformer-acceleration.obs.cn-north-4.myhuaweicloud.com/acltransformer_testdata/images/chatglm6b_wmj_docker.tar`**
  然后上传到服务器中的 **`/data/acltransformer_testdata/images/`** 里
  如果没有这个文件夹，新建即可（**`mkdir -p /data/acltransformer_testdata/images/`** ）
  
或者

  * 直接在服务器里输入：
  **`wget --no-check-certificate -P /data/acltransformer_testdata/images/ "https://ascend-transformer-acceleration.obs.cn-north-4.myhuaweicloud.com/acltransformer_testdata/images/chatglm6b_wmj_docker.tar"`**

#### 1.1.2. 不给新建 **`/data`**？

* 把上传到服务器从 **`/data/`** 改为 **`/usr/local/`** 即可
* wget 脚本中，把 **`-P /data/acltransformer_testdata/images/`** 改为 **`-P /usr/local/acltransformer_testdata/images/`** 即可

---

## 2. 下载模型权重文件 [ chatglm6b 文件夹 (里头共有八个 bin 文件) ]

### 2.1. 已保存在 **`/data/acltransformer_testdata/weights/chatglm6b/`** 里（建议不要改动！！）

#### 2.1.1. 没有找到？

  * 打开浏览器输入：（下载八个文件需要八个链接）
  **`https://ascend-transformer-acceleration.obs.cn-north-4.myhuaweicloud.com/acltransformer_testdata/weights/chatglm6b/pytorch_model-00001-of-00008.bin`**
  **`https://ascend-transformer-acceleration.obs.cn-north-4.myhuaweicloud.com/acltransformer_testdata/weights/chatglm6b/pytorch_model-00002-of-00008.bin`**
  **`https://ascend-transformer-acceleration.obs.cn-north-4.myhuaweicloud.com/acltransformer_testdata/weights/chatglm6b/pytorch_model-00003-of-00008.bin`**
  **`https://ascend-transformer-acceleration.obs.cn-north-4.myhuaweicloud.com/acltransformer_testdata/weights/chatglm6b/pytorch_model-00004-of-00008.bin`**
  **`https://ascend-transformer-acceleration.obs.cn-north-4.myhuaweicloud.com/acltransformer_testdata/weights/chatglm6b/pytorch_model-00005-of-00008.bin`**
  **`https://ascend-transformer-acceleration.obs.cn-north-4.myhuaweicloud.com/acltransformer_testdata/weights/chatglm6b/pytorch_model-00006-of-00008.bin`**
  **`https://ascend-transformer-acceleration.obs.cn-north-4.myhuaweicloud.com/acltransformer_testdata/weights/chatglm6b/pytorch_model-00007-of-00008.bin`**
  **`https://ascend-transformer-acceleration.obs.cn-north-4.myhuaweicloud.com/acltransformer_testdata/weights/chatglm6b/pytorch_model-00008-of-00008.bin`**
  然后上传到服务器中的 **`/data/acltransformer_testdata/weights/chatglm6b/`** 里
  如果没有这个文件夹，新建即可（**`mkdir -p /data/acltransformer_testdata/weights/chatglm6b/`** ）
  
或者

  * 直接在服务器里输入：（下载八个文件需要八个链接）
  **`wget --no-check-certificate -P /data/acltransformer_testdata/weights/chatglm6b/ "https://ascend-transformer-acceleration.obs.cn-north-4.myhuaweicloud.com/acltransformer_testdata/weights/chatglm6b/pytorch_model-00001-of-00008.bin"`**
  **`wget --no-check-certificate -P /data/acltransformer_testdata/weights/chatglm6b/ "https://ascend-transformer-acceleration.obs.cn-north-4.myhuaweicloud.com/acltransformer_testdata/weights/chatglm6b/pytorch_model-00002-of-00008.bin"`**
  **`wget --no-check-certificate -P /data/acltransformer_testdata/weights/chatglm6b/ "https://ascend-transformer-acceleration.obs.cn-north-4.myhuaweicloud.com/acltransformer_testdata/weights/chatglm6b/pytorch_model-00003-of-00008.bin"`**
  **`wget --no-check-certificate -P /data/acltransformer_testdata/weights/chatglm6b/ "https://ascend-transformer-acceleration.obs.cn-north-4.myhuaweicloud.com/acltransformer_testdata/weights/chatglm6b/pytorch_model-00004-of-00008.bin"`**
  **`wget --no-check-certificate -P /data/acltransformer_testdata/weights/chatglm6b/ "https://ascend-transformer-acceleration.obs.cn-north-4.myhuaweicloud.com/acltransformer_testdata/weights/chatglm6b/pytorch_model-00005-of-00008.bin"`**
  **`wget --no-check-certificate -P /data/acltransformer_testdata/weights/chatglm6b/ "https://ascend-transformer-acceleration.obs.cn-north-4.myhuaweicloud.com/acltransformer_testdata/weights/chatglm6b/pytorch_model-00006-of-00008.bin"`**
  **`wget --no-check-certificate -P /data/acltransformer_testdata/weights/chatglm6b/ "https://ascend-transformer-acceleration.obs.cn-north-4.myhuaweicloud.com/acltransformer_testdata/weights/chatglm6b/pytorch_model-00007-of-00008.bin"`**
  **`wget --no-check-certificate -P /data/acltransformer_testdata/weights/chatglm6b/ "https://ascend-transformer-acceleration.obs.cn-north-4.myhuaweicloud.com/acltransformer_testdata/weights/chatglm6b/pytorch_model-00008-of-00008.bin"`**

#### 2.1.2. 不给新建 **`/data`**？

* 把上传到服务器从 **`/data/`** 改为 **`/usr/local/`** 即可
* wget 脚本中，把 **`-P /data/acltransformer_testdata/images/`** 改为 **`-P /usr/local/acltransformer_testdata/images/`** 即可

---

## 3. 下载两个代码仓库（算子库+加速库）[ ascend-transformer-acceleration 文件夹 ] 和 [ ascend-op-common-lib 文件夹 ]
### 3.1. 两个代码仓库（算子库+加速库）都已保存在 **`/home/BACKUP_REPOSITORIES/`** 里

#### 3.1.1. 没有找到？

  * 直接在服务器运行：【需要有代码仓库权限】
  **`git clone https://gitee.com/ascend/ascend-transformer-acceleration.git`** （在 master 分支）
  **`git clone https://gitee.com/ascend/ascend-op-common-lib.git`** （在 dev 分支）

#### 3.1.2. 服务器没有 GIT？

  * 等进了容器 DOCKER 再 GIT CLONE 就好
  * **所以这一小节【3. 两个代码仓库（算子库+加速库）】就可以不用再看下去了**

### 3.2. 更新代码仓库（建议把代码仓库复制到 **`/home/wumingjing/`** 然后再更新）

#### 3.2.1. 把代码仓库复制到你自己创建的目录底下

例如：
```
mkdir -p /home/wumingjing
cp -r /home/BACKUP_REPOSITORIES/ascend-op-common-lib /home/wumingjing/
cp -r /home/BACKUP_REPOSITORIES/ascend-transformer-acceleration /home/wumingjing/
```

#### 3.2.2. 更新代码仓库

  * 分别进入到代码仓库的根目录，然后直接在服务器运行 **`git pull`** 【需要有代码仓库权限】

例如：
```
cd /home/wumingjing/ascend-op-common-lib
pwd    ##### 这里就是模型的根目录
git pull
```

---

# 二、加载镜像

## 1. 直接在服务器运行：

**`docker load -i /data/acltransformer_testdata/images/chatglm6b_wmj_image.tar`**（需要 5 分钟加载）

### 1.1. 报错 `no such file or directory`？

那就要看你执行【一、所需文件】中【1. 下载镜像文件】下载到了哪个文件夹


### 1.2. 看是否加载了没有？

加载后就会出现 **镜像名字**，后面要用上！

**`Loaded image: swr.cn-south-292.ca-aicc.com/yulaoshi/chatglm_mindspore-2-0_pytorch1-11-0_cann6-3-rc1:v1`**

如果没看到，也可以直接在服务器运行 **`docker images`** 看已经安装的镜像，没有这个镜像名字的话，重新跑一遍 **`docker load -i XXX...`**

---

# 三、初始化容器

## 1. 直接在服务器运行：

```
docker run --name ONLY_FOR_WEB_CHATGLM -it -d --net=host --shm-size=500g --device=/dev/davinci1 \
--privileged=true \
-w /home \
--device=/dev/davinci_manager \
--device=/dev/hisi_hdc \
--device=/dev/devmm_svm \
--entrypoint=bash \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /home/wmj:/home/wmj \
-v /data/acltransformer_testdata:/data/acltransformer_testdata \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/sbin/:/usr/local/sbin \
-v /usr/local/bin/:/usr/local/bin \
-v /home/zcj:/home/zcj \
-v /home/hjj:/home/hjj \
-v /usr/local/asdops_opp_kernel:/usr/local/asdops_opp_kernel \
-e "ACLTRANSFORMER_TESTDATA=/data/acltransformer_testdata" \
swr.cn-south-292.ca-aicc.com/yulaoshi/chatglm_mindspore-2-0_pytorch1-11-0_cann6-3-rc1:v1

```
其中，
* **`--name ONLY_FOR_WEB_CHATGLM`**：定义容器名字，**可能会报错名字已使用，这时候就需要改一个名字！**
* **`--net=host`**：表示接受所有端口听写，WEBUI需要
* **`--device=/dev/davinci1`**：确认挂载在服务器哪个卡上，例如 /dev/davinci1 就是卡1
* **`-v /home/wmj:/home/wmj`**：在这个底下有 Cann 包环境（toolkit+kernel），**挂载进去，后面需要这个配置环境变量**
* **`-v /home/zcj:/home/zcj 和 -v /home/hjj:/home/hjj`**：在这个底下有 Cann 包的安装包
* **`-v /usr/local/asdops_opp_kernel:/usr/local/asdops_opp_kernel`**：不加这个可能有 kernel 编译问题
* **`-v /data/acltransformer_testdata:/data/acltransformer_testdata 和 -e "ACLTRANSFORMER_TESTDATA=/data/acltransformer_testdata"`**：**这个其实就是对应前面权重文件的下载地址，里面的函数需要用上该地址，建议不要改动！**

<u>**这里需要注意的是！！两个代码仓库所在的位置你也要挂载进去！！**</u>
<u>**例如，你把那两个代码仓库拷贝到 **`/home/wumingjing/`** 底下了**</u>
<u>**你就需要多加一行 **`-v /home/wumingjing/:/home/wumingjing/`** 进去**</u>

## 2. 查看初始化容器了没有？

加载后就会出现 **容器名字**，后面要用上！

如果没看到，也可以直接在服务器运行 **`docker ps`** 看正在运行的容器，没有这个的话，重新跑一遍 **`docker run XXX...`**

---

# 四、进入容器

## 1. 直接在服务器运行：

**`docker exec -it --uesr root ONLY_FOR_WEB_CHATGLM bash`**

其中，
* **`--uesr root`**：是约束**以root用户进去，不然模型跑的时候会报错**
* **`ONLY_FOR_WEB_CHATGLM`**：是容器名，以你起的名字为主，如果不知道容器名，可以 **`docker ps`** 看正在运行的容器

---

# 五、容器内的环境准备

## 1. 连外网代理

```
export http_proxy="http://90.253.26.55:6688"
export https_proxy=$http_proxy
```

## 2. GIT CLONE 失败，报错 SSL
```
export GIT_SSL_NO_VERIFY=true
git config --global http.sslVerify "false"
### by zhengchenjun，如这个代理不行，可直接找 zhengchenjun 要一个新
```

## 3. 下载两个代码仓库（算子库+加速库）
如果在服务器已经把代码仓库下载下来，并且在 **`docker run XXX...`** 时已挂载进来，那这一步就不用管了
**否则，按照【3. 下载两个代码仓库（算子库+加速库）】执行一遍！！**

---

# 六、编译

## 1. 来到加速库的根目录下，确认一下加速库的三方库（3rdparty）文件夹在不在

按照上面的建议，你的加速库根目录应该是
**`cd /home/wumingjing/ascend-transformer-acceleration/`**
如果有自定义，请自己 **`cd`** 到对应的加速库的根目录下

### 1.1. 如果在？
把三方库（3rdparty）文件夹底下的 **`asdops`** 删掉
**`rm -rf  /home/wumingjing/ascend-transformer-acceleration/3rdparty/asdops`**
然后返回到加速库的根目录下，编译三方库（3rdparty）
**`cd /home/wumingjing/ascend-transformer-acceleration/`**
**`bash scripts/build.sh 3rdparty`**

### 1.2. 如果不在？

直接到加速库的根目录下，编译三方库（3rdparty）
**`cd /home/wumingjing/ascend-transformer-acceleration/`**
**`bash scripts/build.sh 3rdparty`**

## 2. 编译 examples

继续来到加速库的根目录下，编译 examples
**`cd /home/wumingjing/ascend-transformer-acceleration/`**
**`bash scripts/build.sh examples`**

**注意！如果要用 torch_runner 的话，第二句要变成**
**`bash scripts/build.sh examples --use_torch_runner`**

---

# 七、最终运行

按照上面的建议，你的加速库根目录应该是
**`cd /home/wumingjing/ascend-transformer-acceleration/`**
如果有自定义，请自己 **`cd`** 到对应的加速库的根目录下

## 1. 设置加速库环境变量：


来到加速库的根目录下，进入 **`output/acltransformer`**，然后 **`source`** 里面的 **`set_env.sh`**
**`cd //home/wumingjing/ascend-transformer-acceleration/output/acltransformer/output/acltransformer`**
**`source set_env.sh`**


## 2. 设置 Cann 包环境变量：
来到Cann包的目录下，然后 **`source`** 里面的 **`set_env.sh`**
**`cd /home/wmj/Ascend/ascend-toolkit/`**
**`source set_env.sh`**

## 3. 取消打 LOG 的环境变量
```
unset ASDOPS_LOG_LEVEL
unset ASDOPS_LOG_TO_STDOUT
```
## 4. 取消代理的环境变量
```
unset http_proxy
unset http_proxy
```

## 5. 运行脚本

### 5.1 如果要运行原生 libtorch 的终端版本

来到加速库的模型代码文件中，运行 main.py 即可
**`cd /home/wumingjing/ascend-transformer-acceleration/model/chatglm6b`**

用 vi 打开 **`config.json`** 确认里面的第八行和第九行是这样子的
**`vi config.json`**
```
"AutoModel": "modeling_chatglm.ChatGLMForConditionalGeneration",
"AutoModelForSeq2SeqLM": "modeling_chatglm.ChatGLMForConditionalGeneration"

```

然后再运行 main.py 即可
**`python3 main.py`**

### 5.2 如果要运行原生 libtorch 的 WEBUI 版本

来到加速库的模型代码文件中，运行 main.py 即可
**`cd /home/wumingjing/ascend-transformer-acceleration/model/chatglm6b`**

用 vi 打开 **`config.json`** 确认里面的第八行和第九行是这样子的
**`vi config.json`**
```
"AutoModel": "modeling_chatglm.ChatGLMForConditionalGeneration",
"AutoModelForSeq2SeqLM": "modeling_chatglm.ChatGLMForConditionalGeneration"

```

然后再运行 main_web.py 即可
**`python3 main_web.py`**

### 5.3 如果要运行 torch runner 的终端版本

来到加速库的模型代码文件中
**`cd /home/wumingjing/ascend-transformer-acceleration/model/chatglm6b`**

修改 config.json 里面的内容， 把第八行和第九行
从
```
"AutoModel": "modeling_chatglm.ChatGLMForConditionalGeneration",
"AutoModelForSeq2SeqLM": "modeling_chatglm.ChatGLMForConditionalGeneration"

```
改成
```
"AutoModel": "modeling_chatglm_layer.ChatGLMForConditionalGeneration",
"AutoModelForSeq2SeqLM": "modeling_chatglm_layer.ChatGLMForConditionalGeneration"
```
即。把 **`modeling_chatglm`** 变成 **`modeling_chatglm_layer`**

然后再运行 main.py 即可
**`python3 main.py`**

<u>**注意！这里要确认前面编译的时候，已经修改过参数了，即，用 **`bash scripts/build.sh examples --use_torch_runner`** 编译**</u>

### 5.4 如果要运行原生  torch runner 的 WEBUI 版本

来到加速库的模型代码文件中
**`cd /home/wumingjing/ascend-transformer-acceleration/model/chatglm6b`**

修改 config.json 里面的内容， 把第八行和第九行
从
```
"AutoModel": "modeling_chatglm.ChatGLMForConditionalGeneration",
"AutoModelForSeq2SeqLM": "modeling_chatglm.ChatGLMForConditionalGeneration"

```
改成
```
"AutoModel": "modeling_chatglm_layer.ChatGLMForConditionalGeneration",
"AutoModelForSeq2SeqLM": "modeling_chatglm_layer.ChatGLMForConditionalGeneration"
```
即。把 **`modeling_chatglm`** 变成 **`modeling_chatglm_layer`**

然后再运行 main_web.py 即可
**`python3 main_web.py`**

<u>**注意！这里要确认前面编译的时候，已经修改过参数了，即，用 **`bash scripts/build.sh examples --use_torch_runner`** 编译**</u>

### 5.5 如果要运行 ops runner 的终端版本

来到加速库的模型代码文件中
**`cd /home/wumingjing/ascend-transformer-acceleration/model/chatglm6b`**

修改 config.json 里面的内容， 把第八行和第九行
从
```
"AutoModel": "modeling_chatglm.ChatGLMForConditionalGeneration",
"AutoModelForSeq2SeqLM": "modeling_chatglm.ChatGLMForConditionalGeneration"

```
改成
```
"AutoModel": "modeling_chatglm_layer_performance.ChatGLMForConditionalGeneration",
"AutoModelForSeq2SeqLM": "modeling_chatglm_layer_performance.ChatGLMForConditionalGeneration"
```
即。把 **`modeling_chatglm`** 变成 **`modeling_chatglm_layer_performance`**

然后再运行 main.py 即可
**`python3 main.py`**

<u>**注意！这里要确认前面编译的时候，已经修改过参数了，即，用 **`bash scripts/build.sh examples`** 编译**</u>

### 5.6 如果要运行 ops runner 的 WEBUI 版本


来到加速库的模型代码文件中
**`cd /home/wumingjing/ascend-transformer-acceleration/model/chatglm6b`**

修改 config.json 里面的内容， 把第八行和第九行
从
```
"AutoModel": "modeling_chatglm.ChatGLMForConditionalGeneration",
"AutoModelForSeq2SeqLM": "modeling_chatglm.ChatGLMForConditionalGeneration"

```
改成
```
"AutoModel": "modeling_chatglm_layer_performance.ChatGLMForConditionalGeneration",
"AutoModelForSeq2SeqLM": "modeling_chatglm_layer_performance.ChatGLMForConditionalGeneration"
```
即。把 **`modeling_chatglm`** 变成 **`modeling_chatglm_layer_performance`**

然后再运行 main_web.py 即可
**`python3 main_web.py`**

<u>**注意！这里要确认前面编译的时候，已经修改过参数了，即，用 **`bash scripts/build.sh examples`** 编译**</u>