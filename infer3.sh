export CUDA_VISIBLE_DEVICES=3
python3 omni_speech/infer/vocalnet_batch_infer.py --model_path /mnt/buffer/tuwenming/checkpoints/VocalNet/qwen25-7b-instruct-s2s-mtp-ultravoice100k-clean-all-sft-llm-and-decoder/checkpoint-4000 \
    --test_jsonl /share/nlp/tuwenming/projects/UltraVoice_dev/data/metadata_tiny/test/ultravoice_testset.jsonl \
    --s2s \
    --save_dir ./infer_results/VocalNet-qwen25-7b-SFT-llm-and-decoder-ultravoice100k-clean-4000/wav