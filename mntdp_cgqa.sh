
export WANDB_MODE=offline
CUDA_VISIBLE_DEVICES=0 python3 mntdp_cgqa.py --exp_name mntdp_resnet18_cgqa_lr1e-3_ --module_type resnet_block --batch_size=64 --momentum_bn 0.1 --pr_name mntdp_resnet18_cgqa --copy_batchstats 1 --track_running_stats_bn 1 --task_sequence s_minus --gating MNTDP --shuffle_test 0 --epochs 100 --lr 1e-3 --wdecay 1e-3
