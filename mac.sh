TORCH_DISTRIBUTED_DEBUG=INFO RAY_ENABLE_MAC_LARGE_OBJECT_STORE=1 python mac-main.py --data_type synthetic --lr 0.0001 --decay 100-200 --model_level S --swinfeature --batch_size 16 --burst_size 14 --patch_size 256 --data_dir ~/working/datasets/zurich-raw-to-rgb --use_checkpoint --loss L1