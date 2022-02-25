_base_ = [
    '../_base_/models/setr_mla.py',
    '../_base_/datasets/our.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]
model = dict(
    backbone=dict(img_size=384, pos_embed_interp=True, drop_rate=0.,
                  mla_channels=256, mla_index=(2,5,8,11)),
    decode_head=dict(img_size=384, mla_channels=256,
                     mlahead_channels=128, num_classes=2),
    # auxiliary_head=[
    #     dict(
    #         type='VIT_MLA_AUXIHead',
    #         in_channels=256,
    #         channels=512,
    #         in_index=0,
    #         img_size=384,
    #         num_classes=2,
    #         align_corners=False,
    #         loss_decode=dict(
    #             type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4,class_weight=[0.4,1.6])),
    #     dict(
    #         type='VIT_MLA_AUXIHead',
    #         in_channels=256,
    #         channels=512,
    #         in_index=1,
    #         img_size=384,
    #         num_classes=2,
    #         align_corners=False,
    #         loss_decode=dict(
    #             type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4,class_weight=[0.4,1.6])),
    #     dict(
    #         type='VIT_MLA_AUXIHead',
    #         in_channels=256,
    #         channels=512,
    #         in_index=2,
    #         img_size=384,
    #         num_classes=2,
    #         align_corners=False,
    #         loss_decode=dict(
    #             type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4,class_weight=[0.4,1.6])),
    #     dict(
    #         type='VIT_MLA_AUXIHead',
    #         in_channels=256,
    #         channels=512,
    #         in_index=3,
    #         img_size=384,
    #         num_classes=2,
    #         align_corners=False,
    #         loss_decode=dict(
    #             type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4,class_weight=[0.4,1.6])),
    # ])
    )

optimizer = dict(lr=0.002, weight_decay=0.0,
                 paramwise_cfg=dict(custom_keys={'head': dict(lr_mult=10.)})
                 )

crop_size = (384,384)
test_cfg = dict(mode='slide', crop_size=crop_size, stride=(250,250))
find_unused_parameters = True
data = dict(samples_per_gpu=16)