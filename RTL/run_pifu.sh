python RTL/main.py \
    --camera \
    --use_server \
    --loop \
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