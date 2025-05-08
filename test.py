# test_llama_imports.py

import faulthandler
faulthandler.enable()  # on segfault, print C/C++ backtrace

def test_imports(statements):
    """
    Execute each import statement in isolation.
    Stop on Python exception or segmentation fault.
    """
    for stmt in statements:
        print(f"🔍 Testing: {stmt}")
        try:
            # use a fresh namespace to avoid side‑effects
            exec(stmt, {})
            print("   ✔ Success")
        except Exception as e:
            print(f"   ✖ Failed with Python exception: {e}")
            return

if __name__ == "__main__":
    imports_to_test = [
        # Standard library
        "from typing import List, Optional, Tuple, Union",
        # PyTorch core
        "import torch",
        "import torch.nn as nn",
        # Transformers pieces (split out to isolate failures)
        "from transformers import AutoConfig",
        "from transformers import AutoModelForCausalLM",
        "from transformers.modeling_outputs import CausalLMOutputWithPast",
        "from transformers.generation.utils import GenerateOutput",
        # Your local model classes (use absolute path)
        (
            "from omni_speech.model.language_model.omni_speech_arch "
            "import OmniSpeechMetaModel, OmniSpeechMetaForCausalLM"
        ),
    ]
    test_imports(imports_to_test)