
###-----------------------------------
# 1. zh/fixed learing rate=5e-5
# with data_aug
# concate word level and subgraph level and global level(HGM_base)
func_train_sepself_att_sep_v2_b50()
{
    python train_paired_mt_RCSLS_submap.py --caption_model gtssg_sep_self_att_sep_vv1_RCSLS_submap_v2_add_wordmap_add_global --batch_size 50 --accumulate_number 2 --use_ssg 1 --use_isg 0 --freeze_i2t 0 --use_batch_norm 0 \
                    --input_fc_dir data/cocobu_fc --input_att_dir data/cocobu_att --input_ssg_dir data/ALL_11683_v3_COCOCN_spice_sg_t5_en_dict2_aug \
                    --input_json data/cocobu_ALL_11683_v3_COCOCN_t5.json --input_label_h5 data/cocobu_ALL_11683_v3_COCOCN_t5_label.h5 --ssg_dict_path data/ALL_11683_v3_COCOCN_spice_sg_dict_t5.npz_revise.npz \
                    --learning_rate 5e-5 --learning_rate_decay_start -1 --scheduled_sampling_start 0 --learning_rate_decay_every 5 \
                    --save_checkpoint_every 500 --language_eval 1 --beam_size 5 --val_images_use 5000 --max_epochs 80 \
                    --checkpoint_path save  --train_split train --gpu 1 --self_critical_after 1000000  --seq_per_img 1
}


# 1. zh/fixed learing rate=5e-5
# with data_aug  (HGM)
func_train_sepself_att_sep_v2_b50()
{
    python train_paired_mt_RCSLS_submap_naacl.py --caption_model gtssg_sep_self_att_sep_vv1_RCSLS_submap_v2_add_wordmap_add_global_naacl_self_gate --batch_size 50 --accumulate_number 2 --use_ssg 1 --use_isg 0 --freeze_i2t 0 --use_batch_norm 0 \
                    --input_fc_dir data/cocobu_fc --input_att_dir data/cocobu_att --input_ssg_dir data/aic_process/ALL_11683_v3_COCOCN_spice_sg_t5_en_dict2_aug \
                    --input_json data/aic_process/cocobu_ALL_11683_v3_COCOCN_t5.json --input_label_h5 data/aic_process/cocobu_ALL_11683_v3_COCOCN_t5_label.h5 --ssg_dict_path data/aic_process/ALL_11683_v3_COCOCN_spice_sg_dict_t5.npz_revise.npz \
                    --learning_rate 5e-5 --learning_rate_decay_start -1 --scheduled_sampling_start 0 --learning_rate_decay_every 5 \
                    --save_checkpoint_every 1000 --language_eval 1 --beam_size 5 --val_images_use 5000 --max_epochs 80 \
                    --checkpoint_path save_naacl  --train_split train --gpu 0 --self_critical_after 1000000  --seq_per_img 1
}