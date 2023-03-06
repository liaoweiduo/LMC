
export WANDB_MODE=offline
CUDA_VISIBLE_DEVICES=0 python3 lmc_cgqa.py --exp_name lmc_resnet18_cgqa  --exp_dir /apdcephfs/share_1364275/lwd/LMC-experiments --data_dir /apdcephfs/share_1364275/lwd/datasets --module_type resnet_block --activate_after_str_oh=0 --momentum_bn 0.1 --track_running_stats_bn 1 --pr_name lmc_resnet18_cgqa --shuffle_test 0 --init_oh=none --momentum_bn_decoder=0.1 --activation_structural=sigmoid --deviation_threshold=4 --depth=4 --epochs=100 --fix_layers_below_on_addition=0 --hidden_size=512 --lr=0.001 --mask_str_loss=1 --module_init=mean --multihead=gated_linear --normalize_oh=1 --optmize_structure_only_free_modules=1 --projection_layer_oh=0 --projection_phase_length=20 --reg_factor=10 --running_stats_steps=100 --str_prior_factor=1 --str_prior_temp=0.1 --structure_inv=ae --structure_inv_oh=linear_no_act --task_agnostic_test=1 --temp=0.1 --wdecay=0.001
