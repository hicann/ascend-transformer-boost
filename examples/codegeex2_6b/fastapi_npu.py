from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModel
import uvicorn, json, datetime
import torch
import argparse

try:
    import chatglm_cpp
    enable_chatglm_cpp = True
except:
    print("[WARN] chatglm-cpp not found. Install it by `pip install chatglm-cpp` for better performance. "
          "Check out https://github.com/li-plus/chatglm.cpp for more details.")
    enable_chatglm_cpp = False
import torch_npu
import os
from torch_npu.contrib import transfer_to_npu
DEVICE_ID = os.environ.get("SET_NPU_DEVICE")
device_id = 0
if DEVICE_ID is not None:
    device_id = int(DEVICE_ID)
print(f"using npu:{device_id}")
torch.npu.set_device(torch.device(f"npu:{device_id}"))

torch.npu.set_compile_mode(jit_compile=False)
option = {}
option["NPU_FUZZY_COMPILE_BLACKLIST"] = "Tril,ReduceProd"
torch.npu.set_option(option)


#获取选项        
def add_code_generation_args(parser):
    group = parser.add_argument_group(title="CodeGeeX2 DEMO")
    group.add_argument(
        "--model-path",
        type=str,
        default="THUDM/codegeex2-6b",
    )
    group.add_argument(
        "--listen",
        type=str,
        default="127.0.0.1",
    )
    group.add_argument(
        "--port",
        type=int,
        default=7860,
    )
    group.add_argument(
        "--workers",
        type=int,
        default=1,
    )
    group.add_argument(                      
        "--cpu",
        action="store_true",
    )
    group.add_argument(                      
        "--half",
        action="store_true",
    )
    group.add_argument(
        "--quantize",
        type=int,
        default=None,
    )
    group.add_argument(
        "--chatglm-cpp",
        action="store_true",
    )
    return parser

LANGUAGE_TAG = {
    "Abap"         : "* language: Abap",
    "ActionScript" : "// language: ActionScript",
    "Ada"          : "-- language: Ada",
    "Agda"         : "-- language: Agda",
    "ANTLR"        : "// language: ANTLR",
    "AppleScript"  : "-- language: AppleScript",
    "Assembly"     : "; language: Assembly",
    "Augeas"       : "// language: Augeas",
    "AWK"          : "// language: AWK",
    "Basic"        : "' language: Basic",
    "C"            : "// language: C",
    "C#"           : "// language: C#",
    "C++"          : "// language: C++",
    "CMake"        : "# language: CMake",
    "Cobol"        : "// language: Cobol",
    "CSS"          : "/* language: CSS */",
    "CUDA"         : "// language: Cuda",
    "Dart"         : "// language: Dart",
    "Delphi"       : "{language: Delphi}",
    "Dockerfile"   : "# language: Dockerfile",
    "Elixir"       : "# language: Elixir",
    "Erlang"       : f"% language: Erlang",
    "Excel"        : "' language: Excel",
    "F#"           : "// language: F#",
    "Fortran"      : "!language: Fortran",
    "GDScript"     : "# language: GDScript",
    "GLSL"         : "// language: GLSL",
    "Go"           : "// language: Go",
    "Groovy"       : "// language: Groovy",
    "Haskell"      : "-- language: Haskell",
    "HTML"         : "<!--language: HTML-->",
    "Isabelle"     : "(*language: Isabelle*)",
    "Java"         : "// language: Java",
    "JavaScript"   : "// language: JavaScript",
    "Julia"        : "# language: Julia",
    "Kotlin"       : "// language: Kotlin",
    "Lean"         : "-- language: Lean",
    "Lisp"         : "; language: Lisp",
    "Lua"          : "// language: Lua",
    "Markdown"     : "<!--language: Markdown-->",
    "Matlab"       : f"% language: Matlab",
    "Objective-C"  : "// language: Objective-C",
    "Objective-C++": "// language: Objective-C++",
    "Pascal"       : "// language: Pascal",
    "Perl"         : "# language: Perl",
    "PHP"          : "// language: PHP",
    "PowerShell"   : "# language: PowerShell",
    "Prolog"       : f"% language: Prolog",
    "Python"       : "# language: Python",
    "R"            : "# language: R",
    "Racket"       : "; language: Racket",
    "RMarkdown"    : "# language: RMarkdown",
    "Ruby"         : "# language: Ruby",
    "Rust"         : "// language: Rust",
    "Scala"        : "// language: Scala",
    "Scheme"       : "; language: Scheme",
    "Shell"        : "# language: Shell",
    "Solidity"     : "// language: Solidity",
    "SPARQL"       : "# language: SPARQL",
    "SQL"          : "-- language: SQL",
    "Swift"        : "// language: swift",
    "TeX"          : f"% language: TeX",
    "Thrift"       : "/* language: Thrift */",
    "TypeScript"   : "// language: TypeScript",
    "Vue"          : "<!--language: Vue-->",
    "Verilog"      : "// language: Verilog",
    "Visual Basic" : "' language: Visual Basic",
}

app = FastAPI()
def device():
    model = AutoModel.from_pretrained("./", trust_remote_code=True).half().npu()
    return model.eval()

@app.post("/")
async def create_item(request: Request):
    global model, tokenizer
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    lang = json_post_list.get('lang')
    prompt = json_post_list.get('prompt')
    max_length = json_post_list.get('max_length', 512)
    top_p = json_post_list.get('top_p', 0.95)
    temperature = json_post_list.get('temperature', 0.8)
    top_k = json_post_list.get('top_k', 0)
    if lang != "None":
        prompt = LANGUAGE_TAG[lang] + "\n" + prompt
    if enable_chatglm_cpp and args.chatglm_cpp:
        response = model.generate(prompt,
                                  max_length=max_length,
                                  do_sample=temperature > 0,
                                  top_p=top_p,
                                  top_k=top_k,
                                  temperature=temperature)
    else:
        inputs = tokenizer([prompt], return_tensors="pt").to("npu")
        print("inputs", inputs)
        outputs = model.generate(**inputs,
                                max_length=max_length,
                                do_sample=True,
                                top_p=top_p,
                                top_k=top_k,
                                temperature=temperature,
                                eos_token_id=2,
                                pad_token_id=2)
        outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):]
        response = tokenizer.decode(outputs)
        print(response)

    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "message": "success",
        "result": {
            "code":[response],
            "completion_token_num":0,
            "errcode":0,
            "prompt_token_num":0
        },
        "status":0
    }
    log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(response) + '"'
    print(answer)

    return answer


if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser = add_code_generation_args(parser)
    args, _ = parser.parse_known_args()
    tokenizer = AutoTokenizer.from_pretrained("./", trust_remote_code=True)
    model = device()
    uvicorn.run(app, host=args.listen, port=args.port, workers=args.workers)