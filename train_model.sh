python run_medclip.py \
    --output_dir model \
    --text_model_name_or_path="allenai/scibert_scivocab_uncased" \
    --vision_model_name_or_path="openai/clip-vit-base-patch32" \
    --tokenizer_name="allenai/scibert_scivocab_uncased" \
    --mimic_data_dir="/home/shared/data/mimic-cxr/" \
    --mimic_train_file="train_dataset.json" \
    --mimic_validation_file="validate_dataset.json" \
    --mimic_mode="docs" \
    --roco_data_dir="/home/shpotes/data/roco-dataset/" \
    --do_train --do_eval \
    --num_train_epochs="10" --max_seq_length 256 \
    --per_device_train_batch_size="128" \
    --per_device_eval_batch_size="128" \
    --dtype "bfloat16" \
    --learning_rate="3e-4" --warmup_steps="10000" --weight_decay=0.1 \
    --overwrite_output_dir \
    --preprocessing_num_workers=32
