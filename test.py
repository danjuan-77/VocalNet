# test_builder_imports.py
import faulthandler
faulthandler.enable()

import_statements = [
    "import torch",
    "from fairseq.checkpoint_utils import load_model_ensemble_and_task",  # 举例
    "from transformers import AutoModel",                               # 举例
    # …把第一步中 grep 出来的每条语句依次加进来
]

for stmt in import_statements:
    try:
        exec(stmt)
        print(f"✔ 成功：{stmt}")
    except Exception as e:
        print(f"✖ 失败：{stmt} -> {e}")
        break