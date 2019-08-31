python train_CMC.py --data_folder /data5/chengxuz/Dataset/imagenet_raw \
    --model_path /mnt/fs4/chengxuz/cmc_models/res18_half/models \
    --tb_path /mnt/fs4/chengxuz/cmc_models/res18_half/recs \
    --model resnet18 --batch_size 128 --crop_low 0.08 --is_half "$@"
