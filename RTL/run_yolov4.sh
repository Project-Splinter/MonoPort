python RTL/main.py \
    --image_folder "/home/rui/local/projects/PIFu-RealTime/zenTelePort/data/recording/test" \
    -- \
    netG.mean "(0.0, 0.0, 0.0)" \
    netG.std "(1.0, 1.0, 1.0)" \
    netG.projection "orthogonal" \
    netG.backbone.IMF "Yolov4Filters" \
    netG.normalizer.IMF "PIFuNomalizer" \
    netG.head.IMF "PIFuNetGMLP" \
    netG.ckpt_path "./data/Yolov4/netG_epoch_latest" \
