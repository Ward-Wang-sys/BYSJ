# model settings
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='VIT_MLA',
        model_name='vit_base_patch16_384',
        img_size=384,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        num_classes=2,
        drop_rate=0.1,
        norm_cfg=norm_cfg,
        pos_embed_interp=True,
        align_corners=False,
        mla_channels=256,
        mla_index=(2,5,8,11)
    ),
    decode_head=dict(
        type='VIT_MLAHead',
        in_channels=1024,
        channels=512,
        img_size=384,
        mla_channels=256,
        use_edge_loss=True,
        mlahead_channels=128,
        num_classes=2,
        norm_cfg=norm_cfg,
        num_conv=2,
        upsampling_method='bilinear',
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0,class_weight=[1,1]),
        loss_edge_decode = dict(
             type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0,class_weight=[1,1])))
# model training and testing settings
train_cfg = dict()
test_cfg = dict(mode='whole')
