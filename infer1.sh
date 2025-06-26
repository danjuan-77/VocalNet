export CUDA_VISIBLE_DEVICES=1
python3 omni_speech/infer/vocalnet_batch_infer.py --model_path /share/nlp/tuwenming/projects/VocalNet/checkpoints/qwen25-7b-instruct-s2s-mtp-ultravoice-all-sft/checkpoint-1000 \
    --test_jsonl /share/nlp/tuwenming/projects/UltraVoice_dev/data/metadata_tiny/test/ultravoice_testset.jsonl \
    --s2s \
    --save_dir ./infer_results/VocalNet-qwen25-7b-SFT-1000/wav