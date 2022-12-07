/home3/huangshan/open-mmlab/mmediting/mmedit/utils/setup_env.py:33: UserWarning: Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed.
  f'Setting OMP_NUM_THREADS environment variable for each process '
/home3/huangshan/open-mmlab/mmediting/mmedit/utils/setup_env.py:43: UserWarning: Setting MKL_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed.
  f'Setting MKL_NUM_THREADS environment variable for each process '
2022-11-28 14:51:05,208 - mmedit - INFO - Environment info:
------------------------------------------------------------
sys.platform: linux
Python: 3.7.13 (default, Oct 18 2022, 18:57:03) [GCC 11.2.0]
CUDA available: True
GPU 0: GeForce RTX 2080 Ti
CUDA_HOME: /usr/local/cuda
NVCC: Cuda compilation tools, release 11.2, V11.2.142
GCC: gcc (GCC) 12.2.0
PyTorch: 1.10.0+cu102
PyTorch compiling details: PyTorch built with:
  - GCC 7.3
  - C++ Version: 201402
  - Intel(R) oneAPI Math Kernel Library Version 2021.4-Product Build 20210904 for Intel(R) 64 architecture applications
  - Intel(R) MKL-DNN v2.2.3 (Git Hash 7336ca9f055cf1bfa13efb658fe15dc9b41f0740)
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - LAPACK is enabled (usually provided by MKL)
  - NNPACK is enabled
  - CPU capability usage: AVX2
  - CUDA Runtime 10.2
  - NVCC architecture flags: -gencode;arch=compute_37,code=sm_37;-gencode;arch=compute_50,code=sm_50;-gencode;arch=compute_60,code=sm_60;-gencode;arch=compute_70,code=sm_70
  - CuDNN 7.6.5
  - Magma 2.5.2
  - Build settings: BLAS_INFO=mkl, BUILD_TYPE=Release, CUDA_VERSION=10.2, CUDNN_VERSION=7.6.5, CXX_COMPILER=/opt/rh/devtoolset-7/root/usr/bin/c++, CXX_FLAGS= -Wno-deprecated -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -fopenmp -DNDEBUG -DUSE_KINETO -DUSE_FBGEMM -DUSE_QNNPACK -DUSE_PYTORCH_QNNPACK -DUSE_XNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -DEDGE_PROFILER_USE_KINETO -O2 -fPIC -Wno-narrowing -Wall -Wextra -Werror=return-type -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function -Wno-unused-result -Wno-unused-local-typedefs -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=redundant-decls -Wno-error=old-style-cast -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Wno-stringop-overflow, LAPACK_INFO=mkl, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_VERSION=1.10.0, USE_CUDA=ON, USE_CUDNN=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=ON, USE_MKLDNN=ON, USE_MPI=OFF, USE_NCCL=ON, USE_NNPACK=ON, USE_OPENMP=ON, 

TorchVision: 0.11.0+cu102
OpenCV: 4.6.0
MMCV: 1.6.2
MMCV Compiler: GCC 7.3
MMCV CUDA Compiler: 10.2
MMEditing: 0.16.0+556a1e0
------------------------------------------------------------

2022-11-28 14:51:05,209 - mmedit - INFO - Distributed training: False
2022-11-28 14:51:05,209 - mmedit - INFO - mmedit Version: 0.16.0
2022-11-28 14:51:05,209 - mmedit - INFO - Config:
/home3/huangshan/open-mmlab/mmediting/configs/restorers/edsr/edsr_brainhualiao_1000k.py
exp_name = 'edsr_brainhualiao_1000k.py'

scale = 1
# model settings
model = dict(
    type='BasicRestorer',
    generator=dict(
        type='EDSR',
        in_channels=3,
        out_channels=3,
        mid_channels=64,
        num_blocks=16,
        upscale_factor=scale,
        res_scale=1,
        rgb_mean=[0.4488, 0.4371, 0.4040],
        rgb_std=[1.0, 1.0, 1.0]),
    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'))
# model training and testing settings
train_cfg = None
test_cfg = dict(metrics=['PSNR', 'SSIM'], crop_border=scale)

# dataset settings
train_dataset_type = 'SRFolderDataset'
val_dataset_type = 'SRFolderDataset'
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
    
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='PairedRandomCrop', gt_patch_size=32),
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
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path']),
    dict(type='ImageToTensor', keys=['lq', 'gt'])
]

data = dict(
    workers_per_gpu=4,
    train_dataloader=dict(samples_per_gpu=16, drop_last=True),
    val_dataloader=dict(samples_per_gpu=16),
    test_dataloader=dict(samples_per_gpu=16),
    train=dict(
        type='RepeatDataset',
        times=1000,
        dataset=dict(
            type=train_dataset_type,
            lq_folder='/home3/huangshan/reconstruction/SRCNN/SRCNN/mri_data/data_x1_y1_z1/imgs_rot/1/train_lr3',
            gt_folder='/home3/huangshan/reconstruction/SRCNN/SRCNN/mri_data/data_x1_y1_z1/imgs_rot/1/train_hr3',
            # ann_file='data/DIV2K/meta_info_DIV2K800sub_GT.txt',
            pipeline=train_pipeline,
            scale=scale)),
    val=dict(
        type=val_dataset_type,
        lq_folder='/home3/huangshan/reconstruction/SRCNN/SRCNN/mri_data/data_x1_y1_z1/imgs_rot/1/val_lr3',
        gt_folder='/home3/huangshan/reconstruction/SRCNN/SRCNN/mri_data/data_x1_y1_z1/imgs_rot/1/val_hr3',
        pipeline=test_pipeline,
        scale=scale,
        filename_tmpl='{}'),
    test=dict(
        type=val_dataset_type,
        lq_folder='/home3/huangshan/reconstruction/SRCNN/SRCNN/mri_data/data_x1_y1_z1/imgs_rot/1/test_lr3',
        gt_folder='/home3/huangshan/reconstruction/SRCNN/SRCNN/mri_data/data_x1_y1_z1/imgs_rot/1/test_hr3',
        pipeline=test_pipeline,
        scale=scale,
        filename_tmpl='{}'))

# optimizer
optimizers = dict(generator=dict(type='Adam', lr=1e-4, betas=(0.9, 0.999)))

# learning policy
total_iters = 1000000
lr_config = dict(
    policy='Step',
    by_epoch=False,
    step=[200000, 400000, 600000, 800000],
    gamma=0.5)

checkpoint_config = dict(interval=5000, save_optimizer=True, by_epoch=False)
evaluation = dict(interval=5000, save_image=True, gpu_collect=True)
log_config = dict(
    interval=100, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
visual_config = None

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'./work_dirs/{exp_name}'
load_from = None
resume_from = None
workflow = [('train', 1)]

2022-11-28 14:51:05,210 - mmedit - INFO - Set random seed to 1213489088, deterministic: False
2022-11-28 14:51:54,273 - mmedit - INFO - Start running, host: huangshan@node2, work_dir: /home3/huangshan/open-mmlab/mmediting/work_dirs/edsr_brainhualiao_1000k.py
2022-11-28 14:51:54,274 - mmedit - INFO - Hooks will be executed in the following order:
before_run:
(VERY_HIGH   ) StepLrUpdaterHook                  
(NORMAL      ) CheckpointHook                     
(VERY_LOW    ) TextLoggerHook                     
 -------------------- 
before_train_epoch:
(VERY_HIGH   ) StepLrUpdaterHook                  
(LOW         ) IterTimerHook                      
(VERY_LOW    ) TextLoggerHook                     
 -------------------- 
before_train_iter:
(VERY_HIGH   ) StepLrUpdaterHook                  
(LOW         ) IterTimerHook                      
 -------------------- 
after_train_iter:
(NORMAL      ) CheckpointHook                     
(LOW         ) IterTimerHook                      
(LOW         ) EvalIterHook                       
(VERY_LOW    ) TextLoggerHook                     
 -------------------- 
after_train_epoch:
(NORMAL      ) CheckpointHook                     
(VERY_LOW    ) TextLoggerHook                     
 -------------------- 
before_val_epoch:
(LOW         ) IterTimerHook                      
(VERY_LOW    ) TextLoggerHook                     
 -------------------- 
before_val_iter:
(LOW         ) IterTimerHook                      
 -------------------- 
after_val_iter:
(LOW         ) IterTimerHook                      
 -------------------- 
after_val_epoch:
(VERY_LOW    ) TextLoggerHook                     
 -------------------- 
after_run:
(VERY_LOW    ) TextLoggerHook                     
 -------------------- 
2022-11-28 14:51:54,275 - mmedit - INFO - workflow: [('train', 1)], max: 1000000 iters
2022-11-28 14:51:54,275 - mmedit - INFO - Checkpoints will be saved to /home3/huangshan/open-mmlab/mmediting/work_dirs/edsr_brainhualiao_1000k.py by HardDiskBackend.
2022-11-28 14:52:00,352 - mmedit - INFO - Iter [100/1000000]	lr_generator: 1.000e-04, eta: 13:52:26, time: 0.050, data_time: 0.003, memory: 181, loss_pix: 0.0658, loss: 0.0658
2022-11-28 14:52:04,680 - mmedit - INFO - Iter [200/1000000]	lr_generator: 1.000e-04, eta: 12:58:51, time: 0.044, data_time: 0.002, memory: 181, loss_pix: 0.0218, loss: 0.0218
2022-11-28 14:52:08,793 - mmedit - INFO - Iter [300/1000000]	lr_generator: 1.000e-04, eta: 12:27:33, time: 0.041, data_time: 0.002, memory: 181, loss_pix: 0.0146, loss: 0.0146
2022-11-28 14:52:12,928 - mmedit - INFO - Iter [400/1000000]	lr_generator: 1.000e-04, eta: 12:12:46, time: 0.041, data_time: 0.001, memory: 181, loss_pix: 0.0146, loss: 0.0146
2022-11-28 14:52:16,854 - mmedit - INFO - Iter [500/1000000]	lr_generator: 1.000e-04, eta: 11:56:56, time: 0.039, data_time: 0.001, memory: 181, loss_pix: 0.0130, loss: 0.0130
2022-11-28 14:52:20,920 - mmedit - INFO - Iter [600/1000000]	lr_generator: 1.000e-04, eta: 11:50:12, time: 0.041, data_time: 0.001, memory: 181, loss_pix: 0.0128, loss: 0.0128
2022-11-28 14:52:25,189 - mmedit - INFO - Iter [700/1000000]	lr_generator: 1.000e-04, eta: 11:50:14, time: 0.043, data_time: 0.002, memory: 181, loss_pix: 0.0128, loss: 0.0128
2022-11-28 14:52:29,263 - mmedit - INFO - Iter [800/1000000]	lr_generator: 1.000e-04, eta: 11:46:11, time: 0.041, data_time: 0.001, memory: 181, loss_pix: 0.0119, loss: 0.0119
2022-11-28 14:52:33,240 - mmedit - INFO - Iter [900/1000000]	lr_generator: 1.000e-04, eta: 11:41:13, time: 0.040, data_time: 0.001, memory: 181, loss_pix: 0.0127, loss: 0.0127
2022-11-28 14:52:37,376 - mmedit - INFO - Exp name: edsr_brainhualiao_1000k.py
2022-11-28 14:52:37,376 - mmedit - INFO - Iter [1000/1000000]	lr_generator: 1.000e-04, eta: 11:39:53, time: 0.041, data_time: 0.001, memory: 181, loss_pix: 0.0123, loss: 0.0123
2022-11-28 14:52:41,435 - mmedit - INFO - Iter [1100/1000000]	lr_generator: 1.000e-04, eta: 11:37:36, time: 0.041, data_time: 0.001, memory: 181, loss_pix: 0.0116, loss: 0.0116
2022-11-28 14:52:45,495 - mmedit - INFO - Iter [1200/1000000]	lr_generator: 1.000e-04, eta: 11:35:43, time: 0.041, data_time: 0.001, memory: 181, loss_pix: 0.0114, loss: 0.0114
2022-11-28 14:52:49,484 - mmedit - INFO - Iter [1300/1000000]	lr_generator: 1.000e-04, eta: 11:33:12, time: 0.040, data_time: 0.001, memory: 181, loss_pix: 0.0112, loss: 0.0112
2022-11-28 14:52:53,372 - mmedit - INFO - Iter [1400/1000000]	lr_generator: 1.000e-04, eta: 11:29:50, time: 0.039, data_time: 0.001, memory: 181, loss_pix: 0.0115, loss: 0.0115
2022-11-28 14:52:57,233 - mmedit - INFO - Iter [1500/1000000]	lr_generator: 1.000e-04, eta: 11:26:36, time: 0.039, data_time: 0.001, memory: 181, loss_pix: 0.0120, loss: 0.0120
2022-11-28 14:53:01,251 - mmedit - INFO - Iter [1600/1000000]	lr_generator: 1.000e-04, eta: 11:25:24, time: 0.040, data_time: 0.001, memory: 181, loss_pix: 0.0103, loss: 0.0103
2022-11-28 14:53:05,398 - mmedit - INFO - Iter [1700/1000000]	lr_generator: 1.000e-04, eta: 11:25:36, time: 0.041, data_time: 0.001, memory: 181, loss_pix: 0.0119, loss: 0.0119
2022-11-28 14:53:09,452 - mmedit - INFO - Iter [1800/1000000]	lr_generator: 1.000e-04, eta: 11:24:54, time: 0.041, data_time: 0.001, memory: 181, loss_pix: 0.0111, loss: 0.0111
2022-11-28 14:53:13,966 - mmedit - INFO - Iter [1900/1000000]	lr_generator: 1.000e-04, eta: 11:28:18, time: 0.045, data_time: 0.001, memory: 181, loss_pix: 0.0108, loss: 0.0108
2022-11-28 14:53:17,964 - mmedit - INFO - Exp name: edsr_brainhualiao_1000k.py
2022-11-28 14:53:17,964 - mmedit - INFO - Iter [2000/1000000]	lr_generator: 1.000e-04, eta: 11:27:03, time: 0.040, data_time: 0.001, memory: 181, loss_pix: 0.0107, loss: 0.0107
2022-11-28 14:53:21,997 - mmedit - INFO - Iter [2100/1000000]	lr_generator: 1.000e-04, eta: 11:26:13, time: 0.040, data_time: 0.001, memory: 181, loss_pix: 0.0098, loss: 0.0098
2022-11-28 14:53:26,005 - mmedit - INFO - Iter [2200/1000000]	lr_generator: 1.000e-04, eta: 11:25:15, time: 0.040, data_time: 0.001, memory: 181, loss_pix: 0.0112, loss: 0.0112
2022-11-28 14:53:30,006 - mmedit - INFO - Iter [2300/1000000]	lr_generator: 1.000e-04, eta: 11:24:18, time: 0.040, data_time: 0.001, memory: 181, loss_pix: 0.0109, loss: 0.0109
2022-11-28 14:53:34,041 - mmedit - INFO - Iter [2400/1000000]	lr_generator: 1.000e-04, eta: 11:23:40, time: 0.040, data_time: 0.001, memory: 181, loss_pix: 0.0117, loss: 0.0117
2022-11-28 14:53:38,074 - mmedit - INFO - Iter [2500/1000000]	lr_generator: 1.000e-04, eta: 11:23:04, time: 0.040, data_time: 0.001, memory: 181, loss_pix: 0.0109, loss: 0.0109
2022-11-28 14:53:42,256 - mmedit - INFO - Iter [2600/1000000]	lr_generator: 1.000e-04, eta: 11:23:28, time: 0.042, data_time: 0.001, memory: 181, loss_pix: 0.0110, loss: 0.0110
2022-11-28 14:53:46,219 - mmedit - INFO - Iter [2700/1000000]	lr_generator: 1.000e-04, eta: 11:22:28, time: 0.040, data_time: 0.001, memory: 181, loss_pix: 0.0111, loss: 0.0111
2022-11-28 14:53:49,957 - mmedit - INFO - Iter [2800/1000000]	lr_generator: 1.000e-04, eta: 11:20:13, time: 0.037, data_time: 0.001, memory: 181, loss_pix: 0.0108, loss: 0.0108
2022-11-28 14:53:53,765 - mmedit - INFO - Iter [2900/1000000]	lr_generator: 1.000e-04, eta: 11:18:30, time: 0.038, data_time: 0.001, memory: 181, loss_pix: 0.0105, loss: 0.0105
2022-11-28 14:53:57,902 - mmedit - INFO - Exp name: edsr_brainhualiao_1000k.py
2022-11-28 14:53:57,903 - mmedit - INFO - Iter [3000/1000000]	lr_generator: 1.000e-04, eta: 11:18:44, time: 0.041, data_time: 0.001, memory: 181, loss_pix: 0.0104, loss: 0.0104
2022-11-28 14:54:01,753 - mmedit - INFO - Iter [3100/1000000]	lr_generator: 1.000e-04, eta: 11:17:24, time: 0.038, data_time: 0.001, memory: 181, loss_pix: 0.0107, loss: 0.0107
2022-11-28 14:54:06,077 - mmedit - INFO - Iter [3200/1000000]	lr_generator: 1.000e-04, eta: 11:18:37, time: 0.043, data_time: 0.001, memory: 181, loss_pix: 0.0107, loss: 0.0107
2022-11-28 14:54:10,152 - mmedit - INFO - Iter [3300/1000000]	lr_generator: 1.000e-04, eta: 11:18:29, time: 0.041, data_time: 0.001, memory: 181, loss_pix: 0.0102, loss: 0.0102
2022-11-28 14:54:14,438 - mmedit - INFO - Iter [3400/1000000]	lr_generator: 1.000e-04, eta: 11:19:24, time: 0.043, data_time: 0.001, memory: 181, loss_pix: 0.0114, loss: 0.0114
2022-11-28 14:54:18,913 - mmedit - INFO - Iter [3500/1000000]	lr_generator: 1.000e-04, eta: 11:21:09, time: 0.045, data_time: 0.001, memory: 181, loss_pix: 0.0102, loss: 0.0102
2022-11-28 14:54:23,056 - mmedit - INFO - Iter [3600/1000000]	lr_generator: 1.000e-04, eta: 11:21:16, time: 0.041, data_time: 0.001, memory: 181, loss_pix: 0.0104, loss: 0.0104
2022-11-28 14:54:27,076 - mmedit - INFO - Iter [3700/1000000]	lr_generator: 1.000e-04, eta: 11:20:50, time: 0.040, data_time: 0.001, memory: 181, loss_pix: 0.0099, loss: 0.0099
2022-11-28 14:54:31,236 - mmedit - INFO - Iter [3800/1000000]	lr_generator: 1.000e-04, eta: 11:21:01, time: 0.042, data_time: 0.001, memory: 181, loss_pix: 0.0102, loss: 0.0102
2022-11-28 14:54:35,318 - mmedit - INFO - Iter [3900/1000000]	lr_generator: 1.000e-04, eta: 11:20:52, time: 0.041, data_time: 0.001, memory: 181, loss_pix: 0.0107, loss: 0.0107
2022-11-28 14:54:39,495 - mmedit - INFO - Exp name: edsr_brainhualiao_1000k.py
2022-11-28 14:54:39,496 - mmedit - INFO - Iter [4000/1000000]	lr_generator: 1.000e-04, eta: 11:21:06, time: 0.042, data_time: 0.001, memory: 181, loss_pix: 0.0100, loss: 0.0100
2022-11-28 14:54:43,314 - mmedit - INFO - Iter [4100/1000000]	lr_generator: 1.000e-04, eta: 11:19:53, time: 0.038, data_time: 0.001, memory: 181, loss_pix: 0.0107, loss: 0.0107
2022-11-28 14:54:47,388 - mmedit - INFO - Iter [4200/1000000]	lr_generator: 1.000e-04, eta: 11:19:43, time: 0.041, data_time: 0.001, memory: 181, loss_pix: 0.0102, loss: 0.0102
2022-11-28 14:54:51,404 - mmedit - INFO - Iter [4300/1000000]	lr_generator: 1.000e-04, eta: 11:19:21, time: 0.040, data_time: 0.001, memory: 181, loss_pix: 0.0103, loss: 0.0103
2022-11-28 14:54:55,375 - mmedit - INFO - Iter [4400/1000000]	lr_generator: 1.000e-04, eta: 11:18:49, time: 0.040, data_time: 0.001, memory: 181, loss_pix: 0.0101, loss: 0.0101
2022-11-28 14:54:59,487 - mmedit - INFO - Iter [4500/1000000]	lr_generator: 1.000e-04, eta: 11:18:49, time: 0.041, data_time: 0.001, memory: 181, loss_pix: 0.0096, loss: 0.0096
2022-11-28 14:55:03,378 - mmedit - INFO - Iter [4600/1000000]	lr_generator: 1.000e-04, eta: 11:18:01, time: 0.039, data_time: 0.001, memory: 181, loss_pix: 0.0091, loss: 0.0091
2022-11-28 14:55:07,471 - mmedit - INFO - Iter [4700/1000000]	lr_generator: 1.000e-04, eta: 11:17:58, time: 0.041, data_time: 0.001, memory: 181, loss_pix: 0.0095, loss: 0.0095
2022-11-28 14:55:11,270 - mmedit - INFO - Iter [4800/1000000]	lr_generator: 1.000e-04, eta: 11:16:54, time: 0.038, data_time: 0.001, memory: 181, loss_pix: 0.0099, loss: 0.0099
2022-11-28 14:55:15,071 - mmedit - INFO - Iter [4900/1000000]	lr_generator: 1.000e-04, eta: 11:15:53, time: 0.038, data_time: 0.001, memory: 181, loss_pix: 0.0104, loss: 0.0104
2022-11-28 14:55:19,082 - mmedit - INFO - Saving checkpoint at 5000 iterations
[                                                  ] 0/1000, elapsed: 0s, ETA:
  File "/home3/huangshan/open-mmlab/mmediting/tools/train.py", line 170, in <module>
    main()
  File "/home3/huangshan/open-mmlab/mmediting/tools/train.py", line 166, in main
    meta=meta)
  File "/home3/huangshan/open-mmlab/mmediting/mmedit/apis/train.py", line 113, in train_model
    meta=meta)
  File "/home3/huangshan/open-mmlab/mmediting/mmedit/apis/train.py", line 362, in _non_dist_train
    runner.run(data_loaders, cfg.workflow, cfg.total_iters)
  File "/home/huangshan/anaconda3/envs/open-mmlab/lib/python3.7/site-packages/mmcv/runner/iter_based_runner.py", line 144, in run
    iter_runner(iter_loaders[i], **kwargs)
  File "/home/huangshan/anaconda3/envs/open-mmlab/lib/python3.7/site-packages/mmcv/runner/iter_based_runner.py", line 70, in train
    self.call_hook('after_train_iter')
  File "/home/huangshan/anaconda3/envs/open-mmlab/lib/python3.7/site-packages/mmcv/runner/base_runner.py", line 317, in call_hook
    getattr(hook, fn_name)(self)
  File "/home3/huangshan/open-mmlab/mmediting/mmedit/core/evaluation/eval_hooks.py", line 47, in after_train_iter
    iteration=runner.iter)
  File "/home3/huangshan/open-mmlab/mmediting/mmedit/apis/test.py", line 41, in single_gpu_test
    for data in data_loader:
  File "/home/huangshan/anaconda3/envs/open-mmlab/lib/python3.7/site-packages/torch/utils/data/dataloader.py", line 521, in __next__
    data = self._next_data()
  File "/home/huangshan/anaconda3/envs/open-mmlab/lib/python3.7/site-packages/torch/utils/data/dataloader.py", line 1203, in _next_data
    return self._process_data(data)
  File "/home/huangshan/anaconda3/envs/open-mmlab/lib/python3.7/site-packages/torch/utils/data/dataloader.py", line 1229, in _process_data
    data.reraise()
  File "/home/huangshan/anaconda3/envs/open-mmlab/lib/python3.7/site-packages/torch/_utils.py", line 434, in reraise
    raise exception
RuntimeError: Caught RuntimeError in DataLoader worker process 1.
Original Traceback (most recent call last):
  File "/home/huangshan/anaconda3/envs/open-mmlab/lib/python3.7/site-packages/torch/utils/data/_utils/worker.py", line 287, in _worker_loop
    data = fetcher.fetch(index)
  File "/home/huangshan/anaconda3/envs/open-mmlab/lib/python3.7/site-packages/torch/utils/data/_utils/fetch.py", line 52, in fetch
    return self.collate_fn(data)
  File "/home/huangshan/anaconda3/envs/open-mmlab/lib/python3.7/site-packages/mmcv/parallel/collate.py", line 81, in collate
    for key in batch[0]
  File "/home/huangshan/anaconda3/envs/open-mmlab/lib/python3.7/site-packages/mmcv/parallel/collate.py", line 81, in <dictcomp>
    for key in batch[0]
  File "/home/huangshan/anaconda3/envs/open-mmlab/lib/python3.7/site-packages/mmcv/parallel/collate.py", line 84, in collate
    return default_collate(batch)
  File "/home/huangshan/anaconda3/envs/open-mmlab/lib/python3.7/site-packages/torch/utils/data/_utils/collate.py", line 56, in default_collate
    return torch.stack(batch, 0, out=out)
RuntimeError: stack expects each tensor to be equal size, but got [3, 256, 256] at entry 0 and [3, 68, 86] at entry 10
