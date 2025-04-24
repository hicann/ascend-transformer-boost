import torch_atb
import unittest
 
# 模糊校验小数点后6位相同，即c++ float默认单精度
decimal_palace = 6
message = "almost equal assertion failed"
 
class Test(unittest.TestCase):
    def test_layer_norm(self):
        layer_norm_param = torch_atb.LayerNormParam()
 
        expected_layer_type = 0
 
        expected_norm_param_begin_norm_axis = 0
        expected_norm_param_begin_params_axis = 0
        expected_norm_param_epsilon = 1e-5
        expected_norm_param_quant_type = 0
 
        expected_post_norm_param_epsilon = 1e-5
        expected_post_norm_param_op_mode = 0
        expected_post_norm_param_quant_type = 0
        expected_post_norm_param_zoom_scale_value = 1.0
 
        expected_pre_norm_param_epsilon = 1e-5
        expected_pre_norm_param_op_mode = 0
        expected_pre_norm_param_quant_type = 0
        expected_pre_norm_param_zoom_scale_value = 1.0
 
        self.assertEqual(layer_norm_param.layer_type, expected_layer_type)
        self.assertEqual(layer_norm_param.norm_param.begin_norm_axis, expected_norm_param_begin_norm_axis)
        self.assertEqual(layer_norm_param.norm_param.begin_params_axis, expected_norm_param_begin_params_axis)
        self.assertAlmostEqual(layer_norm_param.norm_param.epsilon, expected_norm_param_epsilon, decimal_palace, message)
        self.assertEqual(layer_norm_param.norm_param.quant_type, expected_norm_param_quant_type)
 
        self.assertAlmostEqual(layer_norm_param.post_norm_param.epsilon, expected_post_norm_param_epsilon, decimal_palace, message)
        self.assertEqual(layer_norm_param.post_norm_param.op_mode, expected_post_norm_param_op_mode)
        self.assertEqual(layer_norm_param.post_norm_param.quant_type, expected_post_norm_param_quant_type)
        self.assertEqual(layer_norm_param.post_norm_param.zoom_scale_value, expected_post_norm_param_zoom_scale_value)
 
        self.assertAlmostEqual(layer_norm_param.pre_norm_param.epsilon, expected_pre_norm_param_epsilon, decimal_palace, message)
        self.assertEqual(layer_norm_param.pre_norm_param.op_mode, expected_pre_norm_param_op_mode)
        self.assertEqual(layer_norm_param.pre_norm_param.quant_type, expected_pre_norm_param_quant_type)
        self.assertEqual(layer_norm_param.pre_norm_param.zoom_scale_value, expected_pre_norm_param_zoom_scale_value)
 
    def test_elewise(self):
        elewise_param = torch_atb.ElewiseParam()
 
        expected_elewise_type = 0
        expected_var_attr = 0.0
        expected_tensor_type = -1
        expected_asymmetric = False
        expected_input_offset = 0
        expected_input_scale = 1.0
 
        self.assertEqual(elewise_param.elewise_type, expected_elewise_type)
        self.assertEqual(elewise_param.muls_param.var_attr, expected_var_attr)
        self.assertEqual(elewise_param.out_tensor_type, expected_tensor_type)
        self.assertEqual(elewise_param.quant_param.asymmetric, expected_asymmetric)
        self.assertEqual(elewise_param.quant_param.input_offset, expected_input_offset)
        self.assertEqual(elewise_param.quant_param.input_scale, expected_input_scale)
 
    def test_linear(self):
        linear_param = torch_atb.LinearParam()
 
        expected_en_accum = False
        expected_has_bias = True
        expected_out_data_type = -1
        expected_transpose_a = False
        expected_transpose_b = True
        expected_matmul_type = 0
 
        self.assertEqual(linear_param.en_accum, expected_en_accum)
        self.assertEqual(linear_param.has_bias, expected_has_bias)
        self.assertEqual(linear_param.out_data_type, expected_out_data_type)
        self.assertEqual(linear_param.transpose_a, expected_transpose_a)
        self.assertEqual(linear_param.transpose_b, expected_transpose_b)
        self.assertEqual(linear_param.matmul_type, expected_matmul_type)
 
    def test_softmax(self):
        softmax_param = torch_atb.SoftmaxParam()
        expected_axes = []
        self.assertEqual(softmax_param.axes, expected_axes)
 
    def test_self_attention(self):
        self_attention_param = torch_atb.SelfAttentionParam()
        
        expected_batch_run_status_enable = False
        expected_cache_type = 0
        expected_calc_type = 0
        expected_clamp_max = 0.0
        expected_clamp_min = 0.0
        expected_clamp_type = 0
        expected_head_num = 0
        expected_input_layout = 0
        expected_is_triu_mask = 0
        expected_kernel_type = 0
        expected_kv_head_num = 0
        expected_kvcache_cfg = 0
        expected_mask_type = 0
        expected_mla_v_head_size = 0
        expected_out_data_type = -1
        expected_q_scale = 1.0
        expected_qk_scale = 1.0
        expected_quant_type = 0
        expected_scale_type = 0
        expected_window_size = 0
 
        self.assertEqual(self_attention_param.batch_run_status_enable, expected_batch_run_status_enable)
        self.assertEqual(self_attention_param.cache_type, expected_cache_type)
        self.assertEqual(self_attention_param.calc_type, expected_calc_type)
        self.assertEqual(self_attention_param.clamp_max, expected_clamp_max)
        self.assertEqual(self_attention_param.clamp_min, expected_clamp_min)
        self.assertEqual(self_attention_param.clamp_type, expected_clamp_type)
        self.assertEqual(self_attention_param.head_num, expected_head_num)
        self.assertEqual(self_attention_param.input_layout, expected_input_layout)
        self.assertEqual(self_attention_param.is_triu_mask, expected_is_triu_mask)
        self.assertEqual(self_attention_param.kernel_type, expected_kernel_type)
        self.assertEqual(self_attention_param.kv_head_num, expected_kv_head_num)
        self.assertEqual(self_attention_param.kvcache_cfg, expected_kvcache_cfg)
        self.assertEqual(self_attention_param.mask_type, expected_mask_type)
        self.assertEqual(self_attention_param.mla_v_head_size, expected_mla_v_head_size)
        self.assertEqual(self_attention_param.out_data_type, expected_out_data_type)
        self.assertEqual(self_attention_param.q_scale, expected_q_scale)
        self.assertEqual(self_attention_param.qk_scale, expected_qk_scale)
        self.assertEqual(self_attention_param.quant_type, expected_quant_type)
        self.assertEqual(self_attention_param.scale_type, expected_scale_type)
        self.assertEqual(self_attention_param.window_size, expected_window_size)
 
    def test_rope(self):
        rope_param = torch_atb.RopeParam()
 
        expected_cos_format = 0
        expected_rotary_coeff = 4
 
        self.assertEqual(rope_param.cos_format, expected_cos_format)
        self.assertEqual(rope_param.rotary_coeff, expected_rotary_coeff)
 
    def test_split(self):
        split_param = torch_atb.SplitParam()
 
        expected_split_dim = 0
        expected_split_num = 2
        expected_split_sizes = []
 
        self.assertEqual(split_param.split_dim, expected_split_dim)
        self.assertEqual(split_param.split_num, expected_split_num)
        self.assertEqual(split_param.split_sizes, expected_split_sizes)
 
    def test_gather(self):
        gather_param = torch_atb.GatherParam()
 
        expected_axis = 0
        expected_batch_dims = 0
 
        self.assertEqual(gather_param.axis, expected_axis)
        self.assertEqual(gather_param.batch_dims, expected_batch_dims)
 
    def test_activation(self):
        activation_param = torch_atb.ActivationParam()
 
        expected_activation_type = 0
        expected_dim = -1
        expected_scale = 1.0
 
        self.assertEqual(activation_param.activation_type, expected_activation_type)
        self.assertEqual(activation_param.dim, expected_dim)
        self.assertEqual(activation_param.scale, expected_scale)
 
    def test_rms_norm(self):
        rms_norm_param = torch_atb.RmsNormParam()
 
        expected_layer_type = 0
        expected_norm_param_epsilon = 1e-5
        expected_norm_param_layer_norm_eps = 1e-5
        expected_norm_param_model_type = 0
        expected_norm_param_precision_mode = 0
        expected_norm_param_quant_type = 0
        expected_norm_param_rstd = False
        expected_norm_param_dynamic_quant_type = 0
        expected_post_norm_param_epsilon = 1e-5
        expected_post_norm_param_has_bias = False
        expected_post_norm_param_quant_type = 0
        expected_pre_norm_param_epsilon = 1e-5
        expected_pre_norm_param_has_bias = False
        expected_pre_norm_param_quant_type = 0
        
        self.assertEqual(rms_norm_param.layer_type, expected_layer_type)
        self.assertAlmostEqual(rms_norm_param.norm_param.epsilon, expected_norm_param_epsilon, decimal_palace, message)
        self.assertAlmostEqual(rms_norm_param.norm_param.layer_norm_eps, expected_norm_param_layer_norm_eps, decimal_palace, message)
        self.assertEqual(rms_norm_param.norm_param.model_type, expected_norm_param_model_type)
        self.assertEqual(rms_norm_param.norm_param.precision_mode, expected_norm_param_precision_mode)
        self.assertEqual(rms_norm_param.norm_param.quant_type, expected_norm_param_quant_type)
        self.assertEqual(rms_norm_param.norm_param.rstd, expected_norm_param_rstd)
        self.assertEqual(rms_norm_param.norm_param.dynamic_quant_type, expected_norm_param_dynamic_quant_type)
        self.assertAlmostEqual(rms_norm_param.post_norm_param.epsilon, expected_post_norm_param_epsilon, decimal_palace, message)
        self.assertEqual(rms_norm_param.post_norm_param.has_bias, expected_post_norm_param_has_bias)
        self.assertEqual(rms_norm_param.post_norm_param.quant_type, expected_post_norm_param_quant_type)
        self.assertAlmostEqual(rms_norm_param.pre_norm_param.epsilon, expected_pre_norm_param_epsilon, decimal_palace, message)
        self.assertEqual(rms_norm_param.pre_norm_param.has_bias, expected_pre_norm_param_has_bias)
        self.assertEqual(rms_norm_param.pre_norm_param.quant_type, expected_pre_norm_param_quant_type)
 
    def test_all_gather(self):
        all_gather_param = torch_atb.AllGatherParam()
 
        expected_rank = 0
        expected_rank_size = 0
        expected_rank_root = 0
        expected_backend = "hccl"
        expected_hccl_comm = None
        expected_comm_mode = 0
        expected_rank_table_file = ""
        expected_comm_domain = ""
 
        self.assertEqual(all_gather_param.rank, expected_rank)
        self.assertEqual(all_gather_param.rank_size, expected_rank_size)
        self.assertEqual(all_gather_param.rank_root, expected_rank_root)
        self.assertEqual(all_gather_param.backend, expected_backend)
        self.assertEqual(all_gather_param.hccl_comm, expected_hccl_comm)
        self.assertEqual(all_gather_param.comm_mode, expected_comm_mode)
        self.assertEqual(all_gather_param.rank_table_file, expected_rank_table_file)
        self.assertEqual(all_gather_param.comm_domain, expected_comm_domain)
    
    def test_as_strided(self):
        as_strided_param = torch_atb.AsStridedParam()
 
        expected_size = []
        expected_stride = []
        expected_offset = []
 
        self.assertEqual(as_strided_param.size, expected_size)
        self.assertEqual(as_strided_param.stride, expected_stride)
        self.assertEqual(as_strided_param.offset, expected_offset)
 
    def test_cumsum(self):
        cumsum_param = torch_atb.CumsumParam()
 
        expected_axes = []
        expected_exclusive = False
        expected_reverse = False
 
        self.assertEqual(cumsum_param.axes, expected_axes)
        self.assertEqual(cumsum_param.exclusive, expected_exclusive)
        self.assertEqual(cumsum_param.reverse, expected_reverse)
 
    def test_dynamic_NTK(self):
        dynamic_NTK_param = torch_atb.DynamicNTKParam()
 
        expected_out_data_type = -1
 
        self.assertEqual(dynamic_NTK_param.out_data_type, expected_out_data_type)
 
    def test_multinomial(self):
        multinomial_param = torch_atb.MultinomialParam()
 
        expected_num_samples = 1
        expected_rand_seed = 0
 
        self.assertEqual(multinomial_param.num_samples, expected_num_samples)
        self.assertEqual(multinomial_param.rand_seed, expected_rand_seed)
 
if __name__ == "__main__":
    unittest.main()