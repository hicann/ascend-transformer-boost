import sys
sys.path.append('../..')
import tests.dailytest.model_test as model_test
import os

LLAMA2_13B_PATH = os.path.join(model_test.ACLTRANSFORMER_HOME_PATH, "examples/llama")
RUN_SHELL_PATH = os.path.join(LLAMA2_13B_PATH, "run.sh")
MODEL_SCRIPT_NAME = "modeling_llama_layer.py"

class Llama2_13b_ModelTest(model_test.ModelTest):
    def __init__(self, model_name) -> None:
        super().__init__(model_name)
    
    def generate_precision_performance(self):
        os.system(f"bash {RUN_SHELL_PATH} --precision --llama2-13b {MODEL_SCRIPT_NAME}")
        self.append_precision_golden(LLAMA2_13B_PATH + "/hidden_states_golden.pth")
        self.append_precision_result(LLAMA2_13B_PATH + "/hidden_states.pth")
        self.precision_compare()

def main():
    test_body = Llama2_13b_ModelTest("llama2_13b")
    test_body.generate_precision_performance()

main()