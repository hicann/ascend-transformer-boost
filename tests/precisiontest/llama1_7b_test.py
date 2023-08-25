import sys
sys.path.append('../..')
import tests.dailytest.model_test as model_test
import os

LLAMA1_7B_PATH = os.path.join(model_test.ACLTRANSFORMER_HOME_PATH, "examples/llama")
RUN_SHELL_PATH = os.path.join(LLAMA1_7B_PATH, "run.sh")
MODEL_SCRIPT_PATH = os.path.join(LLAMA1_7B_PATH, "transformers_patch/layer/modeling_layer_performance.py")
# GOLDEN_SCRIPT_PATH = os.path.join(LLAMA1_7B_PATH, "patches/models/modeling_chatglm_model_precision.py")

class Llama1_7b_ModelTest(model_test.ModelTest):
    def __init__(self, model_name) -> None:
        super().__init__(model_name)
    
    def generate_precision_performance(self):
        os.system(f"bash {RUN_SHELL_PATH} --precision --llama1-7b")
        # os.system(f"bash {RUN_SHELL_PATH} {GOLDEN_SCRIPT_PATH} --precision")
        self.append_precision_golden(LLAMA1_7B_PATH + "/hidden_states_golden.pth")
        self.append_precision_result(LLAMA1_7B_PATH + "/hidden_states.pth")
        self.precision_compare()

def main():
    test_body = Llama1_7b_ModelTest("llama1_7b")
    test_body.generate_precision_performance()

main()