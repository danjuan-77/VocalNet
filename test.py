# deep_import_test.py

import faulthandler
faulthandler.enable()  # enable C‑level backtrace on segfault

# List of (name, statement) pairs to test
import_tests = [
    ("torch",                   "import torch"),
    ("typing.Tuple",            "from typing import Tuple"),
    ("omni_speech.builder",     "from omni_speech.model.builder import load_pretrained_model"),
    ("os",                      "import os"),
    ("omni_speech.preprocess",  "from omni_speech.datasets.preprocess import preprocess_llama_3_v1"),
    ("whisper",                 "import whisper"),
    ("numpy",                   "import numpy as np"),
    ("sys",                     "import sys"),
    ("hyperpyyaml",             "from hyperpyyaml import load_hyperpyyaml"),
    ("functools.partial",       "from functools import partial"),
    ("file_utils.load_wav",     "from cosyvoice.utils.file_utils import load_wav"),
    ("CosyVoice2Model",         "from cosyvoice.cli.model import CosyVoice2Model"),
    ("frontend_utils",          "from cosyvoice.utils.frontend_utils import contains_chinese, replace_blank, replace_corner_mark, remove_bracket, spell_out_number, split_paragraph"),
    ("logging",                 "import logging"),
    ("librosa",                 "import librosa"),
    ("torchaudio",              "import torchaudio"),
    ("json",                    "import json"),
    ("onnxruntime",             "import onnxruntime"),
    ("kaldi",                   "import torchaudio.compliance.kaldi as kaldi"),
    ("re",                      "import re"),
    ("argparse",                "import argparse"),
]

for name, stmt in import_tests:
    try:
        exec(stmt)
        print(f"✔ Success: {stmt}")
    except Exception as e:
        # If a Python exception is raised, we report and stop.
        print(f"✖ Python error on {stmt}: {e}")
        break
    # If a segfault occurs, faulthandler will print a C‑level backtrace and abort here.