import model_test
import os
import argparse

LLAMA7B_PATH = os.path.join(model_test.ACLTRANSFORMER_HOME_PATH, "examples/llama")
RUN_SHELL_PATH = os.path.join(LLAMA7B_PATH, "run.sh")
MODEL_SCRIPT_PATH = os.path.join(LLAMA7B_PATH, "transformers_patch/layer/modeling_layer_performance.py")

class Llama7bModelTest(model_test.ModelTest):
    def __init__(self) -> None:
        super().__init__()
    
    def generate_time_performance(self, args):
        print(f"[args.model]: {args.model}")
        statistic = model_test.Statistics()
        statistic.model_name = args.model
        self.create_time(statistic)
        
        performance_file_path = os.path.join(LLAMA7B_PATH, f"zhiputest_{self.device_type}_{args.model}.csv")
        print("-----llama_test-----")
        print(f"[args.model]: {args.model}")
        print(f"[RUN_SHELL_PATH]: {RUN_SHELL_PATH}")
        print(f"[MODEL_SCRIPT_PATH]: {MODEL_SCRIPT_PATH}")
        os.system(f"bash {RUN_SHELL_PATH} --zhipu --{args.model} {MODEL_SCRIPT_PATH} ")
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
                
        
def main(args):
    test_body = Llama7bModelTest()
    test_body.generate_time_performance(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        help="Select One LLaMA Model [llama1-7b | llama1-13b | llama2-7b | llama2-13b]",
    )
    args = parser.parse_args()
    main(args)