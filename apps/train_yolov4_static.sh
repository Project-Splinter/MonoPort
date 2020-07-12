python apps/train_netG.py \
    --dataset "static" \
    -- \
    name "Yolov4_static" \
    batch_size 12 \
    num_threads 12 \
    learning_rate 0.0001 \
    netG.mean "(0.0, 0.0, 0.0)" \
    netG.std "(1.0, 1.0, 1.0)" \
    netG.backbone.IMF "Yolov4Filters" \
    netG.ckpt_path "./data/Yolov4/netG_epoch_latest"