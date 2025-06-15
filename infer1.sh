export CUDA_VISIBLE_DEVICES=1
python3 omni_speech/infer/vocalnet_batch_infer.py --model_path /share/nlp/tuwenming/models/VocalNet/VocalNet-8B/ \
    --test_jsonl /share/nlp/tuwenming/projects/UltraVoice_dev/data/metadata_tiny/test/ultravoice_testset.jsonl \
    --s2s \
    --save_dir ./infer_results/VocalNet-8B/wav