import sys
sys.path.append('../..')
import tests.dailytest.model_test as model_test
import os

CHATGLM6B_PATH = os.path.join(model_test.ACLTRANSFORMER_HOME_PATH, "examples/chatglm6b")
RUN_SHELL_PATH = os.path.join(CHATGLM6B_PATH, "run.sh")
MODEL_SCRIPT_PATH = os.path.join(CHATGLM6B_PATH, "patches/models/modeling_chatglm_model_flashattention_temp_performance.py")
GOLDEN_SCRIPT_PATH = os.path.join(CHATGLM6B_PATH, "patches/models/modeling_chatglm_model_precision.py")

class Chatglm6BModelTest(model_test.ModelTest):
    def __init__(self, model_name) -> None:
        super().__init__(model_name)
    
    def generate_precision_performance(self):
        os.system(f"bash {RUN_SHELL_PATH} {MODEL_SCRIPT_PATH} --precision")
        os.system(f"bash {RUN_SHELL_PATH} {GOLDEN_SCRIPT_PATH} --precision")
        self.append_precision_golden(CHATGLM6B_PATH + "/hidden_states_golden.pth")
        self.append_precision_result(CHATGLM6B_PATH + "/hidden_states.pth")
        self.precision_compare()

def main():
    test_body = Chatglm6BModelTest("chatglm6b")
    test_body.generate_precision_performance()

main()