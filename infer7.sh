export CUDA_VISIBLE_DEVICES=3
python3 omni_speech/infer/vocalnet_batch_infer.py --model_path /share/nlp/tuwenming/projects/VocalNet/checkpoints/llama32-8B-instruct-s2s-mtp-ultravoice-all-sft/checkpoint-4905 \
    --test_jsonl /share/nlp/tuwenming/projects/UltraVoice_dev/data/metadata_tiny/test/ultravoice_testset.jsonl \
    --s2s \
    --save_dir ./infer_results/VocalNet-llama32-8B-SFT-llm-and-decoder-1000/wav