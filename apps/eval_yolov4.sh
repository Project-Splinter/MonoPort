python apps/eval_netG.py \
    --split "train" \
    --dataset "static" \
    -- \
    netG.mean "(0.0, 0.0, 0.0)" \
    netG.std "(1.0, 1.0, 1.0)" \
    netG.backbone.IMF "Yolov4Filters" \
    netG.ckpt_path "./data/Yolov4/netG_epoch_latest"