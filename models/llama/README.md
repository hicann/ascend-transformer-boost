# FastChat

| [Demo](https://chat.lmsys.org/) | [Arena](https://arena.lmsys.org) | [Discord](https://discord.gg/h6kCZb72G7) | [Twitter](https://twitter.com/lmsysorg) |

An open platform for training, serving, and evaluating large language model based chatbots.

## Release

Using [code](https://github.com/lm-sys/FastChat/tree/76f0424d1add61aadc8e5bdeed5ebe540f266ba3) (commit id)

## Contents
- [Install](#install)
- [Model Weights](#model-weights)
- [Inference with Command Line Interface](#inference-with-command-line-interface)
- [Fine-tuning](#fine-tuning)

## Install

1. 

2. Create Env & Install Package
```bash
conda create -n py37 python=3.7
conda activate py37
```

Then Install Other Relative Package.

```bash
pip3 install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements.txt
pip3 install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
```

3. Install Ascend PyTorch version Relative Package.

Download torch_npu from \
http://120.46.158.234/PyTorch/open/v1.11.0/2023050613/ (select version for RC1/RC2)\
or https://gitee.com/ascend/pytorch/releases

```bash
pip unisntall torch
pip install apex-0.1_ascend_XXX.whlls
pip install torch-1.11.0+cpu-XXX.whl
pip install torch_npu-1.11.0.XXX.whl
```

4. Install deepspeed

```bash
pip3 install deepspeed==0.6.0 --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Install deepspeed_npu

```bash
git clone https://gitee.com/ascend/DeepSpeed
cd DeepSpeed
python setup.py develop
```

Add *import deepspeed_npu* in the head of following file. \
(The first file show by "whereis deepspeed")

```bash
vim path/to/environment/envs/py37/bin/deepspeed
```

```python
#!/root/miniconda3/envs/py37/bin/python

import deepspeed_npu
from deepspeed.launcher.runner import main

if __name__ == '__main__':
    main()
```

5. (Optional) Run on Multiple NPUs

Install PDSH for multinode task: https://github.com/chaos/pdsh/releases/download/pdsh-2.34/pdsh-2.34.tar.gz.

```bash
chmod 777 configure
./configure --with-ssh --build=arm-linux
make
make install
```

6. Bug Fix

uncommont the code of version check
```bash
vim /root/miniconda3/envs/py37/lib/python3.7/site-packages/transformers/training_args.py +1264
```
![img_4.png](img_4.png)
```bash
vim /root/miniconda3/envs/py37/lib/python3.7/site-packages/transformers/utils/versions.py +51
```
![img_5.png](img_5.png)

fix for npu
```bash
vim /root/miniconda3/envs/py37/lib/python3.7/site-packages/transformers/models/llama/modeling_llama.py +225
```
![img_6.png](img_6.png)
```bash
vim /root/miniconda3/envs/py37/lib/python3.7/site-packages/transformers/models/llama/modeling_llama.py +112
```
![img_7.png](img_7.png)

```bash
vim /root/miniconda3/envs/py37/lib/python3.7/site-packages/transformers/trainer.py +2424
```
![img_8.png](img_8.png)
```bash
vim /root/miniconda3/envs/py37/lib/python3.7/site-packages/transformers/trainer.py +2327
```
![img_9.png](img_9.png)
```bash
vim /root/miniconda3/envs/py37/lib/python3.7/site-packages/transformers/models/llama/modeling_llama.py +161
```
![img_10.png](img_10.png)
```bash
vim /root/miniconda3/envs/py37/lib/python3.7/site-packages/transformers/models/llama/modeling_llama.py +53
```
![img_11.png](img_11.png)

## Model Weights

### Vicuna Weights

We release [Vicuna](https://vicuna.lmsys.org/) weights as delta weights to comply with the LLaMA model license.
You can add our delta to the original LLaMA weights to obtain the Vicuna weights. Instructions:

1. Get the original LLaMA weights in the huggingface format by following the instructions [here](https://huggingface.co/docs/transformers/main/model_doc/llama).
2. Use the following scripts to get Vicuna weights by applying our delta. They will automatically download delta weights from our Hugging Face [account](https://huggingface.co/lmsys).

**NOTE**:
Weights v1.1 are only compatible with ```transformers>=4.28.0``` and ``fschat >= 0.2.0``.
Please update your local packages accordingly. If you follow the above commands to do a fresh install, then you should get all the correct versions.

#### Vicuna-7B

This conversion command needs around 30 GB of CPU RAM.
See the "Low CPU Memory Conversion" section below if you do not have enough memory.

```bash
python3 -m fastchat.model.apply_delta \
    --base-model-path /path/to/llama-7b \
    --target-model-path /output/path/to/vicuna-7b \
    --delta-path lmsys/vicuna-7b-delta-v1.1
```

#### Vicuna-13B

This conversion command needs around 60 GB of CPU RAM.
See the "Low CPU Memory Conversion" section below if you do not have enough memory.

```bash
python3 -m fastchat.model.apply_delta \
    --base-model-path /path/to/llama-13b \
    --target-model-path /output/path/to/vicuna-13b \
    --delta-path lmsys/vicuna-13b-delta-v1.1
```

## Inference with Command Line Interface

#### Supported Models

The following models are tested:
- Vicuna, LLaMA

#### Single NPU

The command below requires around 28GB of GPU memory for Vicuna-13B and 14GB of NPU memory for Vicuna-7B.
See the "No Enough Memory" section below if you do not have enough memory.

```
python3 -m fastchat.serve.cli --model-path path/to/FastChat/7B-vicuna --num-gpus 1
```

#### Multiple NPUs

You can use model parallelism to aggregate GPU memory from multiple NPUs on the same machine.
```
python3 -m fastchat.serve.cli --model-path path/to/FastChat/13B-vicuna --num-gpus 2
```

## Fine-tuning

### Data

Due to some concerns, we may not release the ShareGPT dataset at the moment. If you would like to try the fine-tuning code, you can run it with some dummy questions in [alpaca-data-conversation.json](https://github.com/lm-sys/FastChat/blob/v0.1.10/playground/data/alpaca-data-conversation.json). You can follow the same format and plug in your own data.

### Code and Hyperparameters

Our code is based on [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) with additional support for multi-round conversations.
We use similar hyperparameters as the Stanford Alpaca.

| Hyperparameter | Global Batch Size | Learning rate | Epochs | Max length | Weight decay |
| --- | ---: | ---: | ---: | ---: | ---: |
| Vicuna-13B | 128 | 2e-5 | 3 | 2048 | 0 |

### Fine-tuning Vicuna-7B with Local NPUs

You can use the following command to train Vicuna-7B with 8 x 910B (60GB).

```bash
source  /usr/local/Ascend/ascend-toolkit/set_env.sh

deepspeed --num_gpus=8  \
    fastchat/train/train_mem.py \
    --model_name_or_path ./7B-vicuna \
    --data_path ./playground/data/alpaca-data-conversation.json \
    --fp16 True \
    --output_dir ./ckpt \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --deepspeed ./deepspeed_conf_7B.json > train_7B.log 2>&1 &
```

### Fine-tuning Vicuna-13B with two nodes
rewrite config in hostfile, and then use the following command:

```bash
#! /bin/bash

NUM_WORKERS=2
NUM_GPUS_PER_WORKER=8
MASTER_PORT=12345

DATESTR=$(date +"%m-%d-%H-%M")

HOST_FILE_PATH="./hostfile"

source /usr/local/Ascend/ascend-toolkit/set_env.sh

run_cmd="HCCL_CONNECT_TIMEOUT=1200 deepspeed --master_port ${MASTER_PORT} --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER} --hostfile ${HOST_FILE_PATH} fastchat/train/train_mem.py \
    --model_name_or_path ./13B-vicuna \
    --data_path ./playground/data/alpaca-data-conversation.json \
    --fp16 True \
    --output_dir /ckpt \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --deepspeed ./deepspeed_conf_13B.json > train_13B.log 2>&1 &"

echo ${run_cmd}
eval ${run_cmd}

set +x
```
