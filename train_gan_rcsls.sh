#!/usr/bin/env bash
#export PYTHONPATH=PYTHONPATH:$PWD
export PYTHONPATH=PYTHONPATH:unpaired_image_caption_revise/
 # word leverl
 # no self-critical, lr5e-5
{
    python train_gan_only.py --caption_model sep_self_att_sep_gan_only --batch_size 50 --lr_policy linear --learning_rate 5e-5 --lr_decay_iters 10000 --gan_type 0 --freeze_i2t 1 --enable_i2t 0 \
                    --use_orthogonal 0 --use_spectral_norm 1 --gan_mode vanilla --gan_d_type 3 --gan_g_type 1 --lambda_A 10.0 --lambda_B 10.0 --pool_size 50 \
                    --save_checkpoint_every 1000 --beam_size 5 --checkpoint_path save_gan_rcsls --self_critical_after 1000000 --train_split train --gpu 0 --seq_per_img=1 --num_images=1000 \
                    --input_isg_dir = data/coco_graph_extract_ft_isg_joint_rcsls  --input_ssg_dir = data/coco_graph_extract_ft_ssg_joint_rcsls \
                    --caption_model_to_replace = up_gtssg_sep_self_att_sep_vv1_RCSLS_p2 \
                    --learning_rate_decay_start -1  --learning_rate_decay_every 5
}

# word+subgraph
{
    python train_gan_only.py --caption_model sep_self_att_sep_gan_only --batch_size 50 --lr_policy linear --learning_rate 5e-5 --lr_decay_iters 10 --gan_type 0 --freeze_i2t 1 --enable_i2t 0 \
                    --use_orthogonal 0 --use_spectral_norm 1 --gan_mode vanilla --gan_d_type 3 --gan_g_type 1 --lambda_A 10.0 --lambda_B 10.0 --pool_size 50 \
                    --save_checkpoint_every 1000 --beam_size 5 --checkpoint_path save_gan_rcsls_s --self_critical_after 10 --train_split train --gpu 0 --seq_per_img=1 --num_images=1000 \
                    --input_isg_dir data/coco_graph_extract_ft_isg_joint_rcsls_submap  --input_ssg_dir data/coco_graph_extract_ft_ssg_joint_rcsls_submap \
                    --caption_model_to_replace up_gtssg_sep_self_att_sep_vv1_RCSLS_add_wordmap_p2 \
                    --learning_rate_decay_start -1  --learning_rate_decay_every 5
}


# HGM(base)
{
    python train_gan_only.py --caption_model sep_self_att_sep_gan_only --batch_size 50 --lr_policy linear --learning_rate 5e-5 --lr_decay_iters 30 --gan_type 0 --freeze_i2t 1 --enable_i2t 0 \
                    --use_orthogonal 0 --use_spectral_norm 1 --gan_mode vanilla --gan_d_type 3 --gan_g_type 1 --lambda_A 10.0 --lambda_B 10.0 --pool_size 50 \
                    --save_checkpoint_every 1000 --beam_size 5 --checkpoint_path save_gan_rcsls_s_g --self_critical_after 1000000 --train_split train --gpu 0 --seq_per_img=1 --num_images=1000 \
                    --input_isg_dir data/coco_graph_extract_ft_isg_joint_rcsls_submap_global  --input_ssg_dir data/coco_graph_extract_ft_ssg_joint_rcsls_submap_global \
                    --caption_model_to_replace up_gtssg_sep_self_att_sep_vv1_RCSLS_add_wordmap_add_global_p2 \
                    --learning_rate_decay_start -1  --learning_rate_decay_every 20
}


# HGM(self_gate)

{
  python train_gan_only.py --caption_model sep_self_att_sep_gan_only --batch_size 50 --lr_policy linear --learning_rate 5e-5 --lr_decay_iters 50 --gan_type 0 --freeze_i2t 1 --enable_i2t 0 \
                    --use_orthogonal 0 --use_spectral_norm 1 --gan_mode vanilla --gan_d_type 3 --gan_g_type 1 --lambda_A 10.0 --lambda_B 10.0 --pool_size 50 \
                    --save_checkpoint_every 1000 --beam_size 5 --checkpoint_path save_gan_rcsls_s_g_naacl --self_critical_after 1000000 --train_split train --gpu 1 --seq_per_img=1 --num_images=1000 \
                    --input_isg_dir data/coco_graph_extract_ft_isg_joint_rcsls_submap_global_naacl_self_gate  --input_ssg_dir data/coco_graph_extract_ft_ssg_joint_rcsls_submap_global_naacl_self_gate \
                    --caption_model_to_replace up_gtssg_sep_self_att_sep_vv1_RCSLS_submap_v2_add_wordmap_add_global_naacl_self_gate_p2 \
                    --learning_rate_decay_start -1  --learning_rate_decay_every 20 \
                    --init_path_zh unpaired_image_caption_revise/save_final_naacl/20201116_203753_gtssg_sep_self_att_sep_vv1_RCSLS_submap_v2_add_wordmap_add_global_naacl_self_gate_p2/model-best.pth\
}