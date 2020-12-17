#!/usr/bin/env bash
export PYTHONPATH=PYTHONPATH:$PWD
{
    python train_paired_cococn_en_fc.py --caption_model newfc --batch_size 50 --accumulate_number 2 --use_ssg 0 --use_isg 0 --freeze_i2t 0 --use_batch_norm 0 \
                    --input_fc_dir data/cocobu_fc --input_att_dir data/cocobu_att \
                    --input_json data/COCOCN_en.json --input_label_h5 data/COCOCN_en_label.h5 \
                    --learning_rate 5e-5 --learning_rate_decay_start -1 --scheduled_sampling_start 0 --learning_rate_decay_every 5 \
                    --save_checkpoint_every 1000 --language_eval 1 --beam_size 5 --val_images_use 1000 --max_epochs 100 \
                    --checkpoint_path save  --train_split train --gpu 0 --self_critical_after 10000000 --seq_per_img 5 --p_flag 1
}