exp_name = 'edvrm_demo'

# model settings
model = dict(
    type='EDVR',
    generator=dict(
        type='EDVRNet',
        in_channels=3,
        out_channels=3,
        mid_channels=64,
        num_frames=5,
        deform_groups=8,
        num_blocks_extraction=5,
        num_blocks_reconstruction=10,
        center_frame_idx=2,
        with_tsa=False),
    pixel_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='sum'))
# model training and testing settings
train_cfg = None
test_cfg = dict(metrics=['PSNR', 'SSIM'], crop_border=0)

# dataset settings
train_dataset_type = 'SRFolderVideoDataset'
val_dataset_type = 'SRFolderVideoDataset'
train_pipeline = [
    dict(type='GenerateFrameIndices', interval_list=[1], frames_per_clip=99),
    dict(type='TemporalReverse', keys='lq_path', reverse_ratio=0),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        flag='unchanged'),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='gt',
        flag='unchanged'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(
        type='Normalize',
        keys=['lq', 'gt'],
        mean=[0, 0, 0],
        std=[1, 1, 1],
        to_rgb=True),
    dict(type='PairedRandomCrop', gt_patch_size=256),
    dict(
        type='Flip', keys=['lq', 'gt'], flip_ratio=0.5,
        direction='horizontal'),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['lq', 'gt'], transpose_ratio=0.5),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path']),
    dict(type='FramesToTensor', keys=['lq', 'gt'])
]

test_pipeline = [
    dict(type='GenerateFrameIndiceswithPadding', padding='reflection_circle'),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        flag='unchanged'),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='gt',
        flag='unchanged'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(
        type='Normalize',
        keys=['lq', 'gt'],
        mean=[0, 0, 0],
        std=[1, 1, 1],
        to_rgb=True),
    dict(
        type='Collect',
        keys=['lq', 'gt'],
        meta_keys=['lq_path', 'gt_path', 'key']),
    dict(type='FramesToTensor', keys=['lq', 'gt'])
]

data = dict(
    workers_per_gpu=4,
    train_dataloader=dict(samples_per_gpu=4, drop_last=True),
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=1000,
        dataset=dict(
            type=train_dataset_type,
            lq_folder='./demo_files/lq_sequences',
            gt_folder='./demo_files/gt_sequences',
            num_input_frames=5,
            pipeline=train_pipeline,
            scale=4,
            test_mode=False)),
    val=dict(
        type=val_dataset_type,
        lq_folder='./demo_files/lq_sequences',
        gt_folder='./demo_files/gt_sequences',
        num_input_frames=5,
        pipeline=test_pipeline,
        scale=4,
        test_mode=True),
    test=dict(
        type=val_dataset_type,
        lq_folder='./demo_files/lq_sequences',
        gt_folder='./demo_files/gt_sequences',
        num_input_frames=5,
        pipeline=test_pipeline,
        scale=4,
        test_mode=True),
)

# optimizer
optimizers = dict(generator=dict(type='Adam', lr=4e-4, betas=(0.9, 0.999)))

# learning policy
total_iters = 100
lr_config = dict(
    policy='CosineRestart',
    by_epoch=False,
    periods=[150000, 150000, 150000, 150000],
    restart_weights=[1, 0.5, 0.5, 0.5],
    min_lr=1e-7)

checkpoint_config = dict(interval=5000, save_optimizer=True, by_epoch=False)
# remove gpu_collect=True in non distributed training
evaluation = dict(interval=50, save_image=False, gpu_collect=True)
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook'),
        # dict(type='PaviLoggerHook', init_kwargs=dict(project='mmedit-sr'))
    ])
visual_config = None

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'./work_dirs/{exp_name}'
load_from = None
resume_from = None
workflow = [('train', 1)]
