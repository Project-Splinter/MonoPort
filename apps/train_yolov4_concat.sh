python apps/train_netG.py \
    --dataset "concat" \
    -- \
    name "Yolov4_concat" \
    batch_size 12 \
    num_threads 12 \
    learning_rate 0.0001 \
    dataset.score_filter 0.02 \
    netG.mean "(0.0, 0.0, 0.0)" \
    netG.std "(1.0, 1.0, 1.0)" \
    netG.backbone.IMF "Yolov4Filters" \
    netG.ckpt_path "./data/Yolov4/netG_epoch_latest"