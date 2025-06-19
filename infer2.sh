export CUDA_VISIBLE_DEVICES=2
python3 omni_speech/infer/vocalnet_batch_infer.py --model_path /share/nlp/tuwenming/projects/VocalNet/checkpoints/llama32-1B-instruct-s2s-mtp-ultravoice-all-sft/checkpoint-3000 \
    --test_jsonl /share/nlp/tuwenming/projects/UltraVoice_dev/data/metadata_tiny/test/ultravoice_testset.jsonl \
    --s2s \
    --save_dir ./infer_results/VocalNet-1B-SFT-3000/wav