# test_omni_speech_arch_imports.py

import faulthandler
faulthandler.enable()  # 如果出现 segfault，会打印本地 C/C++ 回溯

def test_imports(statements):
    """
    Execute each import in its own namespace.
    Stop on Python exception or real segmentation fault.
    """
    for stmt in statements:
        print(f"🔍 Testing: {stmt}")
        try:
            exec(stmt, {})  # isolated namespace, no side‑effects
            print("   ✔ Success")
        except Exception as e:
            print(f"   ✖ Python exception: {e}")
            return

if __name__ == "__main__":
    # 从 omni_speech/model/language_model/omni_speech_arch.py 顶层拷贝的导入语句
    imports_to_test = [
        "from typing import List, Optional, Tuple, Union",
        "import torch",
        "import torch.nn as nn",
        "from transformers import AutoConfig",
        "from transformers import AutoModelForCausalLM",
        "from transformers.modeling_outputs import CausalLMOutputWithPast",
        "from transformers.generation.utils import GenerateOutput",
        # 注意：相对导入需要改为绝对路径
        "from omni_speech.model.language_model.omni_speech_arch import OmniSpeechMetaModel, OmniSpeechMetaForCausalLM",
    ]
    test_imports(imports_to_test)