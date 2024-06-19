model = dict(
    type='FRNet',
    data_preprocessor=dict(type='FrustumRangePreprocessor'),
    voxel_encoder=dict(
        type='FrustumFeatureEncoder',
        in_channels=4,
        feat_channels=(64, 128, 256, 256),
        with_distance=True,
        with_cluster_center=True,
        norm_cfg=dict(type='SyncBN', eps=1e-3, momentum=0.01),
        with_pre_norm=True,
        feat_compression=16),
        # feat_compression=5),
    backbone=dict(
        type='FRNetBackbone',
        in_channels=16,
        point_in_channels=384, # (256 + stem_channels)
        # point_in_channels=640, # (256 + stem_channels)
        depth=34,
        stem_channels=128,
        # stem_channels=384,
        num_stages=4,
        out_channels=(128, 128, 128, 128),
        strides=(1, 2, 2, 2),
        dilations=(1, 1, 1, 1),
        fuse_channels=(256, 128),
        norm_cfg=dict(type='SyncBN', eps=1e-3, momentum=0.01),
        point_norm_cfg=dict(type='SyncBN', eps=1e-3, momentum=0.01),
        act_cfg=dict(type='HSwish', inplace=True),
        # # config for VIT pretrained
        embed_dim=384, 
        vit_depth=12,
        num_heads=6,
        decoder_embed_dim=512,
        decoder_depth=4,
        decoder_num_heads=8,
        mlp_ratio=4,
        n_clss=20,
        skip_filters=128
        ),

    decode_head=dict(
        type='FRHead',
        in_channels=128,
        middle_channels=(128, 256, 128, 64),
        norm_cfg=dict(type='SyncBN', eps=1e-3, momentum=0.01),
        channels=64,
        dropout_ratio=0,
        loss_ce=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            class_weight=None,
            loss_weight=1.0),
        conv_seg_kernel_size=1))
