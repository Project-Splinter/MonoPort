python apps/train_netG.py \
    --dataset "concat" \
    -- \
    name "Yolov4_scale" \
    batch_size 12 \
    num_threads 12 \
    learning_rate 0.0001 \
    dataset.score_filter 0.02 \
    dataset.scale_uniform True \
    netG.mean "(0.0, 0.0, 0.0)" \
    netG.std "(1.0, 1.0, 1.0)" \
    netG.backbone.IMF "Yolov4Filters" \
    netG.ckpt_path "./data/checkpoints/Yolov4_scale/ckpt_1.pth" \
    resume True