python RTL/main.py \
    --image_folder "/home/rui/local/projects/PIFu-RealTime/zenTelePort/data/recording/test" \
    --use_server \
    -- \
    netG.projection orthogonal \
    netG.backbone.IMF PIFuHGFilters \
    netG.normalizer.IMF PIFuNomalizer \
    netG.head.IMF PIFuNetGMLP \
    netG.ckpt_path ./data/PIFu/net_G \
    netC.projection orthogonal \
    netC.backbone.IMF PIFuResBlkFilters \
    netC.normalizer.IMF PIFuNomalizer \
    netC.head.IMF PIFuNetCMLP \
    netC.ckpt_path ./data/PIFu/net_C