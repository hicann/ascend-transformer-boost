import argparse
import torch
import os
import transformers
from transformers import AutoTokenizer, AutoModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cut Model weights.")
    parser.add_argument(
        "--input_path",
        default = "./",
        help="Location of Model weights, which contains model folders",
    )
    parser.add_argument(
        "--output_path",
        default ='./tensor_parallel',
        help="Location to write the part weights",
    )
    parser.add_argument(
        "--world_size",
        default = 2,
        help="world_size",
    )
    args = parser.parse_args()
    args.world_size = int(args.world_size)
    return args


def cut_weights(model, world_size):
    # init state_dict_list
    state_dict_list=[{} for i in range(world_size)]
    for key, tensor in model.state_dict().items():
        # cut tensors
        cut_tensor_list = []
        key_short = ".".join([key.split(".")[-2], key.split(".")[-1]])
        if key_short in ["dense_h_to_4h.weight"]:
            chunk_tensors = torch.chunk(tensor, world_size*2, dim=0)
            for i in range(world_size):
                cut_tensor_list.append(
                    torch.cat([chunk_tensors[i], chunk_tensors[i+2]], dim=0)
                    )
        elif key_short in ["query_key_value.weight", "query_key_value.bias"]:
            hidden_size_per_attention_head = 128
            num_attention_heads_per_partition = 32
            num_multi_query_groups_per_partition = 2
            query_layer, key_layer, value_layer = tensor.split(
                [
                    hidden_size_per_attention_head * num_attention_heads_per_partition,
                    hidden_size_per_attention_head * num_multi_query_groups_per_partition,
                    hidden_size_per_attention_head * num_multi_query_groups_per_partition
                ],
                dim = 0
            )
            query_list = torch.chunk(query_layer, world_size, dim=0)
            key_list = torch.chunk(key_layer, world_size, dim=0)
            value_list = torch.chunk(value_layer, world_size, dim=0)
            for i in range(world_size):
                cut_tensor_list.append(
                    torch.cat([query_list[i], key_list[i], value_list[i]], dim=0)
                )
        elif key_short in ["dense_4h_to_h.weight", "dense.weight"]:
            cut_tensor_list = torch.chunk(tensor, world_size, dim=1)            
        else:
            cut_tensor_list = [tensor] * world_size
        
        # assign state_dict_list
        for i in range(world_size):
            state_dict_list[i][key]=cut_tensor_list[i]
    
    return state_dict_list

if __name__ == "__main__":
    # parse args
    opts = parse_args()

    # load original model and cut weights
    tokenizer = AutoTokenizer.from_pretrained("./", trust_remote_code=True)
    tokenizer.save_pretrained(opts.output_path + "/tokenizer")
    model = AutoModel.from_pretrained("./", trust_remote_code=True).half()
    state_dict_list = cut_weights(model, opts.world_size)

    # save new model config and new weights
    model_config = model.config
    model_config.world_size = opts.world_size
    creat_model = AutoModel.from_config(model_config, trust_remote_code=True)
    for i in range(opts.world_size):
        creat_model.load_state_dict(state_dict_list[i]) #load the weights to the model
        creat_model.save_pretrained(opts.output_path + "/part_model/" + str(i) + "/") #save model
    print('save succcessfully')

