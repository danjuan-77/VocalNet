# test_omni_language_models.py

import faulthandler
faulthandler.enable()  # 如果出现 Segmentation fault，会打印本地 C/C++ 回溯

# 逐个测试 __init__.py 中导入的四个 language_model 子模块
language_modules = [
    "omni_speech.model.language_model.omni_speech_llama",
    "omni_speech.model.language_model.omni_speech2s_llama",
    "omni_speech.model.language_model.omni_speech_qwen2",
    "omni_speech.model.language_model.omni_speech2s_qwen2",
]

for module_name in language_modules:
    print(f"🔍 Testing import: {module_name}")
    try:
        __import__(module_name)
        print(f"  ✔ Success: {module_name}")
    except Exception as e:
        # 捕获纯 Python 异常并打印，然后退出
        print(f"  ✖ Python exception importing {module_name}: {e}")
        break
    # 如果是真正的 Segfault，faulthandler 会在这里终止并打印回溯