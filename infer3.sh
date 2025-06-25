export CUDA_VISIBLE_DEVICES=3
python3 omni_speech/infer/vocalnet_batch_infer.py --model_path /share/nlp/tuwenming/projects/VocalNet/checkpoints/qwen25-7b-instruct-s2s-mtp-ultravoice-all-sft-llm-and-decoder/checkpoint-2000 \
    --test_jsonl /share/nlp/tuwenming/projects/UltraVoice_dev/data/metadata_tiny/test/ultravoice_testset.jsonl \
    --s2s \
    --save_dir ./infer_results/VocalNet-qwen25-7b-SFT-llm-and-decoder-2000/wav


python3 omni_speech/infer/vocalnet_batch_infer.py --model_path /share/nlp/tuwenming/projects/VocalNet/checkpoints/llama32-1B-instruct-s2s-mtp-ultravoice-all-sft-llm-and-decoder/checkpoint-2000 \
    --test_jsonl /share/nlp/tuwenming/projects/UltraVoice_dev/data/metadata_tiny/test/ultravoice_testset.jsonl \
    --s2s \
    --save_dir ./infer_results/VocalNet-llama32-1B-SFT-llm-and-decoder-2000/wav