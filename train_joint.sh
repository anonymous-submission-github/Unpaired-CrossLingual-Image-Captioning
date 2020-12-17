#!/usr/bin/env bash
export PYTHONPATH=PYTHONPATH:$PWD

##-————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
##__________________________________________________________________________________joint training for word level mapping__________

## joint training 2
# fixed lr=5e-5
{
  python train_paired_cross_en_zh_joint_rcsls.py --input_ssg_dir='data/aic_process/ALL_11683_v3_COCOCN_spice_sg_t5_en_dict2_aug'\
                  --input_json='data/aic_process/cocobu_ALL_11683_v3_COCOCN_t5.json'\
                  --input_label_h5='data/aic_process/cocobu_ALL_11683_v3_COCOCN_t5_label.h5'\
                  --ssg_dict_path='data/aic_process/ALL_11683_v3_COCOCN_spice_sg_dict_t5.npz_revise.npz'\
                  --input_ssg_dir_en='data/aic_process/ALL_11683_v3_COCOCN_spice_sg_t5_en_dict2_aug'\
                  --input_json_en='data/aic_process/cocobu_ALL_11683_v3_COCOCN_t5_en.json'\
                  --input_label_h5_en='data/aic_process/cocobu_ALL_11683_v3_COCOCN_t5_en_label.h5'\
                  --ssg_dict_path_en='data/aic_process/ALL_11683_COCOCN_en_v3_spice_sg_dict_t5.npz_revise.npz'\
                  --use_paired_ssg=1 --self_critical_after=100000 \
                  --caption_model_en='gtssg_sep_self_att_sep_vv1_RCSLS_en_p2' \
                  --caption_model_zh='gtssg_sep_self_att_sep_vv1_RCSLS_p2' \
                  --gpu=1 --batch_size=50\
                  --use_ssg=1 --use_isg=0 --freeze_i2t=1 --use_batch_norm=0 \
                  --learning_rate=5e-5 --learning_rate_decay_start=-1 --scheduled_sampling_start=0 --learning_rate_decay_every=5 \
                  --save_checkpoint_every=1000 --language_eval=1 --beam_size=5 --val_images_use=5000 --max_epochs=20 --self_critical_after=100000 --seq_per_img=1 --checkpoint_path save_for_joint
}



##-————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
##__________________________________________________________________________________joint training for sub-graph mapping__________


# fixed lr=5e-5
{
  python train_paired_cross_en_zh_joint_rcsls.py --input_ssg_dir='data/ALL_11683_v3_COCOCN_spice_sg_t5_en_dict2_aug'\
                  --input_json='data/cocobu_ALL_11683_v3_COCOCN_t5.json'\
                  --input_label_h5='data/cocobu_ALL_11683_v3_COCOCN_t5_label.h5'\
                  --ssg_dict_path='data/ALL_11683_v3_COCOCN_spice_sg_dict_t5.npz_revise.npz'\
                  --input_ssg_dir_en='data/ALL_11683_v3_COCOCN_spice_sg_t5_en_dict2_aug'\
                  --input_json_en='data/aic_process/cocobu_ALL_11683_v3_COCOCN_t5_en.json'\
                  --input_label_h5_en='data/aic_process/cocobu_ALL_11683_v3_COCOCN_t5_en_label.h5'\
                  --ssg_dict_path_en='data/aic_process/ALL_11683_COCOCN_en_v3_spice_sg_dict_t5.npz_revise.npz'\
                  --use_paired_ssg=1 --self_critical_after=100000 \
                  --caption_model_en='gtssg_sep_self_att_sep_vv1_RCSLS_en_p2' \
                  --caption_model_zh='gtssg_sep_self_att_sep_vv1_RCSLS_submap_v2_add_wordmap_p2' \
                  --gpu=1 --batch_size=50\
                  --use_ssg=1 --use_isg=0 --freeze_i2t=1 --use_batch_norm=0 \
                  --learning_rate=5e-5 --learning_rate_decay_start=-1 --scheduled_sampling_start=0 --learning_rate_decay_every=5 \
                  --save_checkpoint_every=1000 --language_eval=1 --beam_size=5 --val_images_use=5000 --max_epochs=20 --self_critical_after=100000 --seq_per_img=1 --checkpoint_path save_for_joint
}

##-————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
##__________________________________________________________________________________joint training for hirachy level mapping(base)__________


# fixed lr=5e-5
{
  python train_paired_cross_en_zh_joint_rcsls.py --input_ssg_dir='data/aic_process/ALL_11683_v3_COCOCN_spice_sg_t5_en_dict2_aug'\
                  --input_json='data/aic_process/cocobu_ALL_11683_v3_COCOCN_t5.json'\
                  --input_label_h5='data/aic_process/cocobu_ALL_11683_v3_COCOCN_t5_label.h5'\
                  --ssg_dict_path='data/aic_process/ALL_11683_v3_COCOCN_spice_sg_dict_t5.npz_revise.npz'\
                  --input_ssg_dir_en='data/aic_process/ALL_11683_v3_COCOCN_spice_sg_t5_en_dict2_aug'\
                  --input_json_en='data/aic_process/cocobu_ALL_11683_v3_COCOCN_t5_en.json'\
                  --input_label_h5_en='data/aic_process/cocobu_ALL_11683_v3_COCOCN_t5_en_label.h5'\
                  --ssg_dict_path_en='data/aic_process/ALL_11683_COCOCN_en_v3_spice_sg_dict_t5.npz_revise.npz'\
                  --use_paired_ssg=1 --self_critical_after=100000 \
                  --caption_model_en='gtssg_sep_self_att_sep_vv1_RCSLS_en_p2' \
                  --caption_model_zh='gtssg_sep_self_att_sep_vv1_RCSLS_submap_v2_add_wordmap_add_global_p2' \
                  --gpu=0 --batch_size=50\
                  --use_ssg=1 --use_isg=0 --freeze_i2t=1 --use_batch_norm=0 \
                  --learning_rate=5e-5 --learning_rate_decay_start=-1 --scheduled_sampling_start=0 --learning_rate_decay_every=5 \
                  --save_checkpoint_every=1000 --language_eval=1 --beam_size=5 --val_images_use=5000 --max_epochs=20 --self_critical_after=100000 --seq_per_img=1 --checkpoint_path save_for_joint
}

##-————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
##__________________________________________________________________________________joint training for hirachy level mapping__________


# fixed lr=5e-5

{
  python train_paired_cross_en_zh_joint_rcsls.py --input_ssg_dir='/GPUFS/hku_sac_yu_vs/data/coco_cn/coco_COCOCN_pred_fuse_sg_v3_newid_small_en_and_trans_ssg_en'\
                  --input_json='/GPUFS/hku_sac_yu_vs/data/aic_process/cocobu_ALL_11683_v3_COCOCN_t5.json'\
                  --input_label_h5='/GPUFS/hku_sac_yu_vs/data/aic_process/cocobu_ALL_11683_v3_COCOCN_t5_label.h5'\
                  --ssg_dict_path='/GPUFS/hku_sac_yu_vs/data/aic_process/ALL_11683_v3_COCOCN_spice_sg_dict_t5.npz_revise.npz'\
                  --input_ssg_dir_en='/GPUFS/hku_sac_yu_vs/data/coco_cn/coco_COCOCN_pred_fuse_sg_v3_newid_small_en_and_trans_ssg_en'\
                  --input_json_en='/GPUFS/hku_sac_yu_vs/data/aic_process/cocobu_ALL_11683_v3_COCOCN_t5_en.json'\
                  --input_label_h5_en='/GPUFS/hku_sac_yu_vs/data/aic_process/cocobu_ALL_11683_v3_COCOCN_t5_en_label.h5'\
                  --ssg_dict_path_en='/GPUFS/hku_sac_yu_vs/data/aic_process/ALL_11683_COCOCN_en_v3_spice_sg_dict_t5.npz_revise.npz'\
                  --use_paired_ssg=1 --self_critical_after=100000 \
                  --caption_model_en='gtssg_sep_self_att_sep_vv1_RCSLS_en_p2' \
                  --caption_model_zh='gtssg_sep_self_att_sep_vv1_RCSLS_submap_v2_add_wordmap_add_global_naacl_self_gate_p2' \
                  --gpu=0 --batch_size=50\
                  --use_ssg=1 --use_isg=0 --freeze_i2t=1 --use_batch_norm=0 \
                  --learning_rate=5e-4 --learning_rate_decay_start=0 --scheduled_sampling_start=0 --learning_rate_decay_every=5 \
                  --save_checkpoint_every=1000 --language_eval=1 --beam_size=5 --val_images_use=5000 --max_epochs=20 --self_critical_after=100000 --seq_per_img=1 --checkpoint_path save_for_joint\
                  --init_path_en='save_for_joint/20200531_163514_gtssg_sep_self_att_sep_vv1_RCSLS_en/model-best.pth'\
                  --init_path_zh='save_for_joint/20201111_134807_gtssg_sep_self_att_sep_vv1_RCSLS_submap_v2_add_wordmap_add_global_naacl_self_gate/model-best.pth'
                  }