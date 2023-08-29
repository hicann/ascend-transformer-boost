import sys
sys.path.append('../..')
import tests.dailytest.model_test as model_test
import os

BLOOM_7B_PATH = os.path.join(model_test.ACLTRANSFORMER_HOME_PATH, "examples/bloom7b")
RUN_SHELL_PATH = os.path.join(BLOOM_7B_PATH, "run.sh")
MODEL_SCRIPT_PATH = os.path.join(BLOOM_7B_PATH, "patches/layers/modeling_bloom_layer_performance.py")
# GOLDEN_SCRIPT_PATH = os.path.join(BLOOM_7B_PATH, "patches/models/modeling_chatglm_model_precision.py")

class Bloom_7BModelTest(model_test.ModelTest):
    def __init__(self, model_name) -> None:
        super().__init__(model_name)
    
    def generate_precision_performance(self):
        os.system(f"bash {RUN_SHELL_PATH} {MODEL_SCRIPT_PATH} --precision")
        # os.system(f"bash {RUN_SHELL_PATH} {GOLDEN_SCRIPT_PATH} --precision")
        self.append_precision_golden(BLOOM_7B_PATH + "/hidden_states_golden.pth")
        self.append_precision_result(BLOOM_7B_PATH + "/hidden_states.pth")
        self.precision_compare()

def main():
    test_body = Bloom_7BModelTest("bloom_7b")
    test_body.generate_precision_performance()

main()