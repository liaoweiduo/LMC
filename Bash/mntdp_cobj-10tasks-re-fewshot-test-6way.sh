
export WANDB_MODE=offline
CUDA_VISIBLE_DEVICES=4 python3 mntdp_cgqa.py --exp_name mntdp_resnet18_cobj_10tasks_lr4e-4 --lr 4e-4 --n_tasks 10 --test_ways 6 --dataset cobj --seed 1234 --module_type resnet_block --batch_size=100 --depth 5 --redo_final_test 0 --epochs 100 --momentum_bn 0.1 --pr_name mntdp_resnet18_cobj --copy_batchstats 1 --track_running_stats_bn 1 --task_sequence s_minus --gating MNTDP --shuffle_test 0 --wdecay 0
