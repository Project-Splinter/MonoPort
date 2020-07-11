python apps/train_netG.py \
    --dataset "dynamic" \
    -- \
    name "Yolov4" \
    batch_size 12 \
    num_threads 6 \
    netG.mean "(0.0, 0.0, 0.0)" \
    netG.std "(1.0, 1.0, 1.0)" \
    netG.backbone.IMF "Yolov4Filters" \
    netG.ckpt_path "./data/Yolov4/netG_epoch_latest"