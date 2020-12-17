#!/usr/bin/env bash

#!/usr/bin/env bash
#!/usr/bin/env bash
export PYTHONPATH=PYTHONPATH:$PWD
# independent training
{ python eval_mt_naacl.py --caption_model gtssg_sep_self_att_sep_vv1_RCSLS_submap_v2_add_wordmap_add_global_naacl_self_gate \
                          --model unpaired_image_caption_revise/save_naacl/20201111_134807_gtssg_sep_self_att_sep_vv1_RCSLS_submap_v2_add_wordmap_add_global_naacl_self_gate/model-best.pth \
                          --infos_path unpaired_image_caption_revise/save_naacl/20201111_134807_gtssg_sep_self_att_sep_vv1_RCSLS_submap_v2_add_wordmap_add_global_naacl_self_gate/infos-best.pkl \
                          --start_from save_naacl/20201111_134807_gtssg_sep_self_att_sep_vv1_RCSLS_submap_v2_add_wordmap_add_global_naacl_self_gate/ \
                          --gpu 0
}

# joint training
{ python eval_mt_naacl.py --caption_model gtssg_sep_self_att_sep_vv1_RCSLS_submap_v2_add_wordmap_add_global_naacl_self_gate_p2 \
                          --model unpaired_image_caption_revise/save_naacl/20201116_203753_gtssg_sep_self_att_sep_vv1_RCSLS_submap_v2_add_wordmap_add_global_naacl_self_gate_p2/model-best.pth \
                          --infos_path unpaired_image_caption_revise/save_naacl/20201116_203753_gtssg_sep_self_att_sep_vv1_RCSLS_submap_v2_add_wordmap_add_global_naacl_self_gate_p2/infos-best.pkl \
                          --start_from save_naacl/20201116_203753_gtssg_sep_self_att_sep_vv1_RCSLS_submap_v2_add_wordmap_add_global_naacl_self_gate_p2/ \
                          --gpu 0
}

