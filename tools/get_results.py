# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os

import mmcv
from mmedit import datasets
import torch
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from mmedit.apis import multi_gpu_test, set_random_seed, single_gpu_test
from mmedit.core.distributed_wrapper import DistributedDataParallelWrapper
from mmedit.datasets import build_dataloader, build_dataset
from mmedit.models import build_model
from mmedit.utils import setup_multi_processes
import numpy as np
import SimpleITK as sitk
import h5py

def parse_args():
    parser = argparse.ArgumentParser(description='mmediting tester')
    parser.add_argument('--config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument('--out', help='output result pickle file')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument(
        '--save-path',
        default=None,
        type=str,
        help='path to store images and if not given, will not save image')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    rank, _ = get_dist_info()

    # set random seeds
    if args.seed is not None:
        if rank == 0:
            print('set random seed to', args.seed)
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)

    loader_cfg = {
        **dict((k, cfg.data[k]) for k in ['workers_per_gpu'] if k in cfg.data),
        **dict(
            samples_per_gpu=1,
            drop_last=False,
            shuffle=False,
            dist=distributed),
        **cfg.data.get('test_dataloader', {})
    }
    data_loader = build_dataloader(dataset, **loader_cfg)
    # build the model and load checkpoint
    model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    data_dir = './mri_data/data_x1_y1_z1/mri1_1_1_newmodal'
    h5_path = '/home3/huangshan/reconstruction/SRCNN/SRCNN/mri_data/data_x1_y1_z1/low_data1.h5'
    
    result_dir = './result_hualiao1'
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    planes = ['yz', 'xz']
    for patient in os.listdir(data_dir):
        print('--------process {} -----------'.format(patient))
        patient_path = os.path.join(data_dir, patient)
        new_patient_path = os.path.join(result_dir, patient)
        if not os.path.exists(new_patient_path):
            os.mkdir(new_patient_path)
        for date in os.listdir(patient_path):
            date_path = os.path.join(patient_path, date)
            new_date_path = os.path.join(result_dir, patient)
            if not os.path.exists(new_date_path):
                os.mkdir(new_date_path)
            for modal in os.listdir(date_path):
                modal_path = os.path.join(date_path, modal) #原图路径
                new_modal_path = os.path.join(new_date_path, modal)
                volume = sitk.ReadImage(modal_path)
                sr_aver = None
                i = 0
                for plane in planes:
                    
                    args.info = (patient, date, modal, plane)
                    
                    args.filename = h5_path
                    # loader = data.Data(args)
                    
                    sr_vol = None
                    
                    with h5py.File(h5_path, 'r') as f:
                        zeronum1 =  int(np.array(f[patient][date][modal.split('.')[0]][plane]['zero_numend1']))
                        zeronum2 = int(np.array(f[patient][date][modal.split('.')[0]][plane]['zero_numend2']))
                    
                    args.save_image = args.save_path is not None
                    empty_cache = cfg.get('empty_cache', False)
                    if not distributed:
                        _ = load_checkpoint(model, args.checkpoint, map_location='cpu')
                        model = MMDataParallel(model)
                        outputs = single_gpu_test(
                            model,
                            data_loader,
                            save_path=args.save_path,
                            save_image=args.save_image)
                    else:
                        find_unused_parameters = cfg.get('find_unused_parameters', False)
                        model = DistributedDataParallelWrapper(
                            model,
                            device_ids=[torch.cuda.current_device()],
                            broadcast_buffers=False,
                            find_unused_parameters=find_unused_parameters)

                        device_id = torch.cuda.current_device()
                        _ = load_checkpoint(
                            model,
                            args.checkpoint,
                            map_location=lambda storage, loc: storage.cuda(device_id))
                        outputs = multi_gpu_test(
                            model,
                            data_loader,
                            args.tmpdir,
                            args.gpu_collect,
                            save_path=args.save_path,
                            save_image=args.save_image,
                            empty_cache=empty_cache)
                    sr_vol = outputs
                    if plane == 'yz':
                        sr_vol = sr_vol.transpose(1,0,2)
                        h, _, w = sr_vol.shape
                        sr_vol1 = np.zeros((h, zeronum1+1, w))
                        sr_vol = np.concatenate([sr_vol1, sr_vol], axis=1)
                        sr_vol2 = np.zeros((h, 255-zeronum2+1, w))
                        sr_vol = np.concatenate([sr_vol, sr_vol2], axis=1)
                    else:
                        sr_vol = sr_vol.transpose(2,1,0)
                        h, w, _ = sr_vol.shape
                        sr_vol1 = np.zeros((h, w, zeronum1+1))
                        sr_vol = np.concatenate([sr_vol1, sr_vol], axis=2)
                        # sr_vol2 = np.zeros((h, 255-zeronum2+1, w))
                        sr_vol2 = np.zeros((h, w, 255-zeronum2+1))
                        sr_vol = np.concatenate([sr_vol, sr_vol2], axis=2)
                    
                    high_vol = sitk.GetImageFromArray(sr_vol)
                    high_vol = sitk.Cast(sitk.RescaleIntensity(high_vol), sitk.sitkUInt16)
                    high_vol.SetDirection(volume.GetDirection())
                    high_vol.SetSpacing(volume.GetSpacing())
                    high_vol.SetOrigin(volume.GetOrigin())
                    sitk.WriteImage(high_vol, str(i) + '.nii.gz')
                    if sr_aver is  None:
                        sr_aver = sr_vol
                        
                    else:
                        sr_aver = (sr_aver + sr_vol) / 2
                    i += 1
                high_vol = sitk.GetImageFromArray(sr_aver)
                high_vol = sitk.Cast(sitk.RescaleIntensity(high_vol), sitk.sitkUInt16)
                high_vol.SetDirection(volume.GetDirection())
                high_vol.SetSpacing(volume.GetSpacing())
                high_vol.SetOrigin(volume.GetOrigin())
                
                
                if rank == 0 and 'eval_result' in outputs[0]:
                    print('')
                    # print metrics
                    stats = datasets.evaluate(outputs)
                    for stat in stats:
                        print('Eval-{}: {}'.format(stat, stats[stat]))

                    # save result pickle
                    if args.out:
                        print('writing results to {}'.format(args.out))
                        mmcv.dump(outputs, args.out)


if __name__ == '__main__':
    main()
