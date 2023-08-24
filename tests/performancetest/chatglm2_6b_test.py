import sys
sys.path.append('../..')
import tests.dailytest.model_test as model_test
import os

CHATGLM2_6B_PATH = os.path.join(model_test.ACLTRANSFORMER_HOME_PATH, "examples/chatglm2_6b")
RUN_SHELL_PATH = os.path.join(CHATGLM2_6B_PATH, "run.sh")
MODEL_SCRIPT_PATH = os.path.join(CHATGLM2_6B_PATH, "modeling_chatglm_model_flashattention.py")

class Chatglm2ModelTest(model_test.ModelTest):
    def __init__(self, model_name) -> None:
        super().__init__(model_name)
    
    def generate_time_performance(self):
        statistic = model_test.Statistics()
        
        performance_file_path = os.path.join(CHATGLM2_6B_PATH, f"zhiputest_{self.device_type}.csv")
        os.system(f"bash {RUN_SHELL_PATH} {MODEL_SCRIPT_PATH} --zhipu")
        if not os.path.exists(performance_file_path):
            print(f"file {performance_file_path} not exist!")
            return

        with open(performance_file_path, 'r') as performance_file:
            lines = performance_file.readlines()
            for i in range(1, len(lines)):
                contents = lines[i].split(',')
                statistic.batch = contents[0]
                statistic.max_seq_len = contents[1]
                statistic.input_seq_len = contents[2]
                statistic.output_seq_len = contents[3]
                statistic.tokens_per_second = contents[4]
                statistic.response_time = contents[5]
                statistic.first_token_time = contents[6]
                statistic.time_per_tokens = contents[7]
                self.append_time(statistic)
                
        
def main():
    test_body = Chatglm2ModelTest("chatglm2_6b")
    test_body.generate_time_performance()

if __name__ == "__main__":
    main()