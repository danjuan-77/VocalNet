export CUDA_VISIBLE_DEVICES=1
python3 omni_speech/infer/vocalnet_batch_infer.py --model_path /share/nlp/tuwenming/projects/VocalNet/checkpoints/llama32-8B-instruct-s2s-mtp-ultravoice100k-clean-all-sft-llm-and-decoder-save-steps200/checkpoint-2400 \
    --test_jsonl /share/nlp/tuwenming/projects/UltraVoice_dev/data/metadata_tiny/test/ultravoice_testset.jsonl \
    --s2s \
    --save_dir /share/nlp/kangyipeng/infer_results/VocalNet-llama32-8B-SFT-llm-and-decoder-2400/wav