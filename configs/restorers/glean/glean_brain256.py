exp_name = 'glean_brain_256_256'

scale = 8
# model settings
model = dict(
    type='GLEAN',
    generator=dict(
        type='GLEANStyleGANv2FuC',
        in_size=256,
        out_size=256,
        style_channels=512,
        pretrained=dict(
            ckpt_path='./pretrained_model/glean/best_fid_iter_80000.pth',
            prefix='generator_ema')),
    discriminator=dict(
        type='StyleGAN2Discriminator',
        in_size=256,
        pretrained=dict(
            ckpt_path='./pretrained_model/glean/best_fid_iter_80000.pth',
            prefix='discriminator')),
    pixel_loss=dict(type='MSELoss', loss_weight=1.0, reduction='mean'),
    perceptual_loss=dict(
        type='PerceptualLoss',
        layer_weights={'21': 1.0},
        vgg_type='vgg16',
        perceptual_weight=1e-2,
        style_weight=0,
        norm_img=False,
        criterion='mse',
        pretrained='torchvision://vgg16'),
    gan_loss=dict(
        type='GANLoss',
        gan_type='vanilla',
        loss_weight=1e-2,
        real_label_val=1.0,
        fake_label_val=0),
    pretrained=None,
)
# model training and testing settings
train_cfg = None
test_cfg = dict(metrics=['PSNR', 'SSIM'], crop_border=0)

# dataset settings
train_dataset_type = 'SRFolderDataset'
val_dataset_type = 'SRFolderDataset'
test_dataset_type = 'SRH5pyDataset'
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='lq',
        flag='color',
        channel_order='rgb'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='gt',
        flag='color',
        channel_order='rgb'),
    dict(type='MATLABLikeResize', keys=['gt'], output_shape=(256, 256)),
    dict(type='MATLABLikeResize', keys=['lq'], output_shape=(256, 256)),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(
        type='Flip', keys=['lq', 'gt'], flip_ratio=0.5,
        direction='horizontal'),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['lq', 'gt'], transpose_ratio=0.5),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path']),
    dict(type='ImageToTensor', keys=['lq', 'gt'])
]
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='lq',
        flag='color',
        channel_order='rgb'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='gt',
        flag='color',
        channel_order='rgb'),
    dict(type='MATLABLikeResize', keys=['gt'], output_shape=(256, 256)),
    dict(type='MATLABLikeResize', keys=['lq'], output_shape=(256, 256)),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path']),
    dict(type='ImageToTensor', keys=['lq', 'gt'])
]
real_pipeline=[
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='lq',
        flag='color',
        channel_order='rgb'),
    dict(type='MATLABLikeResize', keys=['lq'], output_shape=(256, 256)),
    dict(type='RescaleToZeroOne', keys=['lq']),
    dict(type='Collect', keys=['lq'], meta_keys=['lq_path']),
    dict(type='ImageToTensor', keys=['lq'])
]


data = dict(
    workers_per_gpu=2,
    train_dataloader=dict(samples_per_gpu=2, drop_last=True),
    val_dataloader=dict(samples_per_gpu=2),
    test_dataloader=dict(samples_per_gpu=2),
    train=dict(
        type='RepeatDataset',
        times=1000,
        dataset=dict(
            type=train_dataset_type,
            lq_folder='/home3/huangshan/reconstruction/SRCNN/SRCNN/mri_data/data_x1_y1_z1/imgs_rot/1/train_lr3',
            gt_folder='/home3/huangshan/reconstruction/SRCNN/SRCNN/mri_data/data_x1_y1_z1/imgs_rot/1/train_hr3',
            # ann_file='data/DIV2K/meta_info_DIV2K800sub_GT.txt',
            pipeline=train_pipeline,
            scale=scale,
            ratio1=0,
            ratio2=0.3)),
    val=dict(
        type=val_dataset_type,
        lq_folder='/home3/huangshan/reconstruction/SRCNN/SRCNN/mri_data/data_x1_y1_z1/imgs_rot/1/val_lr3',
        gt_folder='/home3/huangshan/reconstruction/SRCNN/SRCNN/mri_data/data_x1_y1_z1/imgs_rot/1/val_hr3',
        pipeline=test_pipeline,
        scale=scale,
        filename_tmpl='{}',
        ratio1=0,
        ratio2=0.15),
    test=dict(
        type=val_dataset_type,
        lq_folder='/home3/huangshan/reconstruction/SRCNN/SRCNN/mri_data/data_x1_y1_z1/imgs_rot/1/val_lr3',
        gt_folder='/home3/huangshan/reconstruction/SRCNN/SRCNN/mri_data/data_x1_y1_z1/imgs_rot/1/val_hr3',
        # lq_folder='/home3/huangshan/dataset/hcp_imgs/3/2/lr',
        # gt_folder='/home3/huangshan/dataset/hcp_imgs/3/2/hr',
        pipeline=test_pipeline,
        scale=scale,
        filename_tmpl='{}',
        ratio1=0.90,
        ratio2=0.95))


# data = dict(
#     workers_per_gpu=4,
#     train_dataloader=dict(samples_per_gpu=2, drop_last=True),
#     val_dataloader=dict(samples_per_gpu=2),
#     test_dataloader=dict(samples_per_gpu=2),
#     train=dict(
#         type='RepeatDataset',
#         times=1000,
#         dataset=dict(
#             type=train_dataset_type,
#             lq_folder='/home3/huangshan/reconstruction/SRCNN/SRCNN/mri_data/data_x1_y1_z1/imgs_rot/1/train_lr3',
#             gt_folder='/home3/huangshan/reconstruction/SRCNN/SRCNN/mri_data/data_x1_y1_z1/imgs_rot/1/train_hr3',
#             # ann_file='data/DIV2K/meta_info_DIV2K800sub_GT.txt',
#             pipeline=train_pipeline,
#             scale=scale,
#             ratio1=0,
#             ratio2=0.003)),
#     val=dict(
#         type=val_dataset_type,
#         lq_folder='/home3/huangshan/reconstruction/SRCNN/SRCNN/mri_data/data_x1_y1_z1/imgs_rot/1/val_lr3',
#         gt_folder='/home3/huangshan/reconstruction/SRCNN/SRCNN/mri_data/data_x1_y1_z1/imgs_rot/1/val_hr3',
#         pipeline=test_pipeline,
#         scale=scale,
#         filename_tmpl='{}',
#         ratio1=0,
#         ratio2=0.0015),
#     test=dict(
#         type=val_dataset_type,
#         lq_folder='/home3/huangshan/reconstruction/SRCNN/SRCNN/mri_data/data_x1_y1_z1/imgs_rot/1/val_lr3',
#         gt_folder='/home3/huangshan/reconstruction/SRCNN/SRCNN/mri_data/data_x1_y1_z1/imgs_rot/1/val_hr3',
#         # lq_folder='/home3/huangshan/dataset/hcp_imgs/3/2/lr',
#         # gt_folder='/home3/huangshan/dataset/hcp_imgs/3/2/hr',
#         pipeline=test_pipeline,
#         scale=scale,
#         filename_tmpl='{}',
#         ratio1=0.90,
#         ratio2=0.95))


# optimizer
optimizers = dict(
    generator=dict(type='Adam', lr=1e-4, betas=(0.9, 0.99)),
    discriminator=dict(type='Adam', lr=1e-4, betas=(0.9, 0.99)))

# learning policy
total_iters = 300000
lr_config = dict(
    policy='CosineRestart',
    by_epoch=False,
    periods=[300000],
    restart_weights=[1],
    min_lr=1e-7)

checkpoint_config = dict(interval=3000, save_optimizer=True, by_epoch=False)
evaluation = dict(interval=30000, save_image=False)
log_config = dict(
    interval=2,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook'),
    ])
visual_config = None

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'./work_dirs/{exp_name}'
load_from = None
resume_from = None
workflow = [('train', 1)]
find_unused_parameters = True
