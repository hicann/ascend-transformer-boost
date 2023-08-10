import model_test

class Chatglm6BModelTest(model_test.ModelTest):
    def __init__(self) -> None:
        super().__init__()
    
    def generate_time_performance(self):
        statistic = model_test.Statistics()
        self.create_time(statistic)
        self.append_time(statistic)
        
def main():
    test_body = Chatglm6BModelTest()
    test_body.generate_time_performance()

main()