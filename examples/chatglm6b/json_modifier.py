import transformers
import readline
from transformers import AutoTokenizer, AutoModel
import signal
import platform
import os
import torch
import sys

def main(model_type, model_name):
    file_path = "./patches/"+model_type+"/config.json"
    with open(file_path, "w") as file:
        file.write("{\n")
        file.write("\"_name_or_path\": \"THUDM/chatglm-6b\",\n")
        file.write("\"architectures\": [\n")
        file.write("\"ChatGLMModel\"\n")
        file.write("],\n")
        file.write("\"auto_map\": {\n")
        file.write("\"AutoConfig\": \"configuration_chatglm.ChatGLMConfig\",\n")
        file.write("\"AutoModel\": \""+model_name+".ChatGLMForConditionalGeneration\",\n")
        file.write("\"AutoModelForSeq2SeqLM\": \""+model_name+".ChatGLMForConditionalGeneration\"\n")
        file.write("},\n")
        file.write("\"bos_token_id\": 150004,\n")
        file.write("\"eos_token_id\": 150005,\n")
        file.write("\"hidden_size\": 4096,\n")
        file.write("\"inner_hidden_size\": 16384,\n")
        file.write("\"layernorm_epsilon\": 1e-05,\n")
        file.write("\"max_sequence_length\": 2048,\n")
        file.write("\"model_type\": \"chatglm\",\n")
        file.write("\"num_attention_heads\": 32,\n")
        file.write("\"num_layers\": 28,\n")
        file.write("\"position_encoding_2d\": true,\n")
        file.write("\"torch_dtype\": \"float16\",\n")
        file.write("\"transformers_version\": \"4.23.1\",\n")
        file.write("\"use_cache\": true,\n")
        file.write("\"vocab_size\": 150528\n")
        file.write("}\n")


if __name__ == "__main__":
    model_type = sys.argv[1]
    model_name = sys.argv[2]
    main(model_type, model_name)