# test_builder_builder_imports.py

import faulthandler
faulthandler.enable()  # 如果出现 Segfault，会打印 C/C++ 回溯

# 按 builder.py 中的顺序，逐条测试下面的导入语句
import_statements = [
    "import os",
    "import warnings",
    "import shutil",
    "import pdb",
    "from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig",
    "import torch",
    "from omni_speech.model import *",
    "from omni_speech.model.speech_encoder.builder import build_speech_encoder",
]

for stmt in import_statements:
    print(f"🔍 Testing: {stmt}")
    try:
        exec(stmt, {})  # 在隔离的全局命名空间中执行
        print("  ✔ Success")
    except Exception as e:
        print(f"  ✖ Python error: {e}")
        break
    # 如果出现 Segmentation fault，faulthandler 会在这里中断并打印回溯