python apps/train_netG.py \
    --dataset "dynamic" \
    -- \
    name "Yolov4_hip" \
    batch_size 12 \
    num_threads 6 \
    netG.mean "(0.0, 0.0, 0.0)" \
    netG.std "(1.0, 1.0, 1.0)" \
    netG.backbone.IMF "Yolov4Filters" \
    dataset.align_hip True \
    netG.ckpt_path "./data/checkpoints/Yolov4_hip/ckpt_0.pth" \
    resume True