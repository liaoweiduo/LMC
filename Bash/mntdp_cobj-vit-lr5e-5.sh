
export WANDB_MODE=offline
CUDA_VISIBLE_DEVICES=0 python3 mntdp_cgqa.py --exp_name mntdp_vit_cobj_3tasks_lr5e-5 --lr 5e-5 --n_tasks 5 --dataset cobj --seed 1234 --module_type vit_block --hidden_size=384 --batch_size=100 --depth 5 --redo_final_test 1 --epochs 200 --image_size 224 --momentum_bn 0.1 --pr_name mntdp_vit_cobj --copy_batchstats 1 --track_running_stats_bn 1 --task_sequence s_minus --gating MNTDP --shuffle_test 0 --wdecay 0
