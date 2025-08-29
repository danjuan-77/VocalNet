export CUDA_VISIBLE_DEVICES=0
python3 omni_speech/infer/vocalnet_batch_infer.py --model_path /share/nlp/tuwenming/projects/VocalNet/checkpoints/qwen25-7B-instruct-s2s-mtp-ultravoice100k-clean-all-sft-llm-and-decoder-save-steps200/checkpoint-3200 \
    --test_jsonl /share/nlp/tuwenming/projects/UltraVoice_dev/data/metadata_tiny/test/ultravoice_testset.jsonl \
    --s2s \
    --save_dir /share/nlp/kangyipeng/infer_results/VocalNet-qwen25-7B-SFT-llm-and-decoder-ultravoice100k-clean-3200/wav