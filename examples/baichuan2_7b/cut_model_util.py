import torch
import os
import argparse
from transformers import AutoTokenizer
from transformers.models.llama.configuration_llama import LlamaConfig
from modeling_baichuan2_parallel import LlamaForCausalLM

# cut_row_keys: dim 0  cut_col_keys: dim 1  nn.linear: x*A.T
def cut_weights(model, world_size, cut_W_pack_keys=['W_pack'], cut_row_keys=['gate_proj','up_proj'], cut_col_keys=['o_proj','down_proj']):
    state_dict_list = [{} for i in range(world_size)]
    for key, tensor in model.state_dict().items():
        key_short = key.split('.')[-2]
        if key_short in cut_W_pack_keys:
            split_linear_size = 3 #  q k v linear
            full_q_weights, full_k_weights, full_v_weights = torch.chunk(tensor, split_linear_size, dim=0)
            cut_q_weights = torch.chunk(full_q_weights, world_size, dim=0)
            cut_k_weights = torch.chunk(full_k_weights, world_size, dim=0)
            cut_v_weights = torch.chunk(full_v_weights, world_size, dim=0)
            cut_tensor_list = []
            for i in range(world_size):
                cut_tensor_list.append(torch.concat((cut_q_weights[i], cut_k_weights[i], cut_v_weights[i]), dim=0))
        elif key_short in cut_row_keys:
            cut_tensor_list = torch.chunk(tensor, world_size, dim=0)
        elif key_short in cut_col_keys:
            cut_tensor_list = torch.chunk(tensor, world_size, dim=1)
        else:
            cut_tensor_list = [tensor] * world_size
        for i in range(world_size):
            state_dict_list[i][key] = cut_tensor_list[i]
    return state_dict_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cut Model weights.")
    parser.add_argument(
        "--input_path",
        default = "/data/models/baichuan2",
        help="Location of Model weights, which contains model folders",
    )
    parser.add_argument(
        "--output_path",
        default ='/data/models/baichuan2/baichuan2-7b-part',
        help="Location to write the part weights",
    )
    parser.add_argument(
        "--world_size",
        default = 2,
        help="world_size",
    )
    parser.add_argument(
        "--cut_W_pack_keys",
        default = ['W_pack'],
        help="cut_W_pack_keys",
    )
    parser.add_argument(
        "--cut_row_keys",
        default = ['gate_proj','up_proj'],
        help="cut_row_keys",
    )
    parser.add_argument(
        "--cut_col_keys",
        default = ['o_proj','down_proj'],
        help="cut_col_keys",
    )

    args = parser.parse_args()
    args.world_size = int(args.world_size) 
    
    # step 1: load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.input_path, use_fast=False)
    
    # step 2: save the tokenizer
    tokenizer.save_pretrained(args.output_path + '/tokenizer')
    
    # step 3: load the raw model
    model = LlamaForCausalLM.from_pretrained(args.input_path, torch_dtype=torch.float16)

    # step 4: cut weight
    state_dict_list = cut_weights(model, args.world_size, args.cut_W_pack_keys, args.cut_row_keys, args.cut_col_keys)

    # step 5: create new model config, add the world size parameter, the model size will be cut according to the world size in the model file
    model_config = model.config
    create_config = LlamaConfig(
            vocab_size=model_config.vocab_size,
            hidden_size=model_config.hidden_size,
            intermediate_size=model_config.intermediate_size,
            num_hidden_layers=model_config.num_hidden_layers,
            num_attention_heads=model_config.num_attention_heads,
            hidden_act=model_config.hidden_act,
            initializer_range=model_config.initializer_range,
            rms_norm_eps=model_config.rms_norm_eps,
            use_cache=model_config.use_cache,
            pad_token_id=model_config.pad_token_id,
            bos_token_id=model_config.bos_token_id,
            eos_token_id=model_config.eos_token_id,
            world_size=args.world_size,
            architectures=model_config.architectures,
            model_type= model_config.model_type,
            torch_dtype= model_config.torch_dtype,
            transformers_version= model_config.transformers_version,
            baichuan_model_type= model_config.baichuan_model_type,
            tie_word_embeddings= model_config.tie_word_embeddings
    )

    # step 6: create new model according to the new model config
    creat_model = LlamaForCausalLM(create_config)
    for i in range(args.world_size):
        # step 7: load weights to each rank model
        creat_model.load_state_dict(state_dict_list[i])
        # step 8: save each rank model
        creat_model.save_pretrained(args.output_path + '/part_model/' + str(i) + '/')
    print('save succcessfully')