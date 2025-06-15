export CUDA_VISIBLE_DEVICES=0
python3 omni_speech/infer/vocalnet_batch_infer.py --model_path /share/nlp/tuwenming/models/VocalNet/VocalNet-1B/ \
    --test_jsonl /share/nlp/tuwenming/projects/UltraVoice_dev/data/metadata_tiny/test/ultravoice_testset.jsonl \
    --s2s \
    --save_dir ./infer_results/VocalNet-1B/wav