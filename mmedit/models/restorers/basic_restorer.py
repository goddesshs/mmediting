# Copyright (c) OpenMMLab. All rights reserved.
import numbers
import os
import os.path as osp

import mmcv
from mmcv.runner import auto_fp16

from mmedit.core import psnr, ssim, tensor2img
from ..base import BaseModel
from ..builder import build_backbone, build_loss
from ..registry import MODELS
import numpy as np

@MODELS.register_module()
class BasicRestorer(BaseModel):
    """Basic model for image restoration.

    It must contain a generator that takes an image as inputs and outputs a
    restored image. It also has a pixel-wise loss for training.

    The subclasses should overwrite the function `forward_train`,
    `forward_test` and `train_step`.

    Args:
        generator (dict): Config for the generator structure.
        pixel_loss (dict): Config for pixel-wise loss.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path for pretrained model. Default: None.
    """
    allowed_metrics = {'PSNR': psnr, 'SSIM': ssim}

    def __init__(self,
                 generator,
                 pixel_loss,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super().__init__()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # support fp16
        self.fp16_enabled = False

        # generator
        self.generator = build_backbone(generator)
        self.init_weights(pretrained)

        # loss
        self.pixel_loss = build_loss(pixel_loss)

    def init_weights(self, pretrained=None):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
        """
        self.generator.init_weights(pretrained)

    @auto_fp16(apply_to=('lq', ))
    def forward(self, lq, gt=None, test_mode=False, **kwargs):
        """Forward function.

        Args:
            lq (Tensor): Input lq images.
            gt (Tensor): Ground-truth image. Default: None.
            test_mode (bool): Whether in test mode or not. Default: False.
            kwargs (dict): Other arguments.
        """

        if test_mode:
            return self.forward_test(lq, gt, **kwargs)

        return self.forward_train(lq, gt)

    def forward_train(self, lq, gt):
        """Training forward function.

        Args:
            lq (Tensor): LQ Tensor with shape (n, c, h, w).
            gt (Tensor): GT Tensor with shape (n, c, h, w).

        Returns:
            Tensor: Output tensor.
        """
        losses = dict()
        output = self.generator(lq)
        loss_pix = self.pixel_loss(output, gt)
        losses['loss_pix'] = loss_pix
        outputs = dict(
            losses=losses,
            num_samples=len(gt.data),
            results=dict(lq=lq.cpu(), gt=gt.cpu(), output=output.cpu()))
        return outputs

    def evaluate(self, output, gt):
        """Evaluation function.

        Args:
            output (Tensor): Model output with shape (n, c, h, w).
            gt (Tensor): GT Tensor with shape (n, c, h, w).

        Returns:
            dict: Evaluation results.
        """
        resutls = []
        crop_border = self.test_cfg.crop_border
        for i in range(gt.shape[0]):
            o = tensor2img(output[i,:,:,:])
            g = tensor2img(gt[i,:,:,:])
            if np.any(o):
                eval_result = dict()
                for metric in self.test_cfg.metrics:
                    eval_result[metric] = self.allowed_metrics[metric](o, g,
                                                                    crop_border)
                resutls.append(dict(eval_result=eval_result))
        return resutls

    def forward_test(self,
                     lq,
                     gt=None,
                     meta=None,
                     save_image=False,
                     save_path=None,
                     iteration=None):
        """Testing forward function.

        Args:
            lq (Tensor): LQ Tensor with shape (n, c, h, w).
            gt (Tensor): GT Tensor with shape (n, c, h, w). Default: None.
            save_image (bool): Whether to save image. Default: False.
            save_path (str): Path to save image. Default: None.
            iteration (int): Iteration for the saving image name.
                Default: None.

        Returns:
            dict: Output results.
        """
        output = self.generator(lq)
        # normalize from [-1, 1] to [0, 1]
        # output = (output + 1) / 2.0
        if self.test_cfg is not None and self.test_cfg.get('metrics', None):
            assert gt is not None, (
                'evaluation with metrics must have gt images.')
            # results = dict(eval_result=self.evaluate(output, gt))
            results = self.evaluate(output, gt)
        else:
            results = []
            result = dict(lq=lq.cpu(), output=output.cpu())
            if gt is not None:
                result['gt'] = gt.cpu()
            results.append(result)

        # save image
        if save_image:
            # if save_image:
            for i in range(output.shape[0]):
                o = tensor2img(output[i, :, :, :])
                l = tensor2img(lq[i, :, :, :])
                f= save_path.rsplit('_', 1)[0]
                lq_folder_name = f+ '_lr'
                gt_folder_name = f + '_gthr'
                # lq_path = os.path.join()
                if not os.path.exists(lq_folder_name):
                    os.makedirs(lq_folder_name)
                if not os.path.exists(gt_folder_name):
                    os.makedirs(gt_folder_name)
                lq_path = meta[i]['lq_path']
                folder_name = osp.splitext(osp.basename(lq_path))[0]
                # if isinstance(iteration, numbers.Number):
                #     save_path = osp.join(save_path, folder_name,
                #                         f'{folder_name}-{iteration + 1:06d}.png')
                # elif iteration is None:
                #     save_path = osp.join(save_path, f'{folder_name}.png')
                # else:
                #     raise ValueError('iteration should be number or None, '
                #                     f'but got {type(iteration)}')
                mmcv.imwrite(o, osp.join(save_path, f'{folder_name}.png'))
                if gt is not None:
                    g = tensor2img(gt[i, :, :, :])
                    mmcv.imwrite(g, osp.join(gt_folder_name, f'{folder_name}.png'))
                mmcv.imwrite(l, osp.join(lq_folder_name, f'{folder_name}.png'))
                
            # if isinstance(iteration, numbers.Number):
            #     save_path = osp.join(save_path, folder_name,
            #                          f'{folder_name}-{iteration + 1:06d}.png')
            # elif iteration is None:
            #     save_path = osp.join(save_path, f'{folder_name}.png')
            # else:
            #     raise ValueError('iteration should be number or None, '
            #                      f'but got {type(iteration)}')
            # mmcv.imwrite(tensor2img(output), save_path)

        return results

    def forward_dummy(self, img):
        """Used for computing network FLOPs.

        Args:
            img (Tensor): Input image.

        Returns:
            Tensor: Output image.
        """
        out = self.generator(img)
        return out

    def train_step(self, data_batch, optimizer):
        """Train step.

        Args:
            data_batch (dict): A batch of data.
            optimizer (obj): Optimizer.

        Returns:
            dict: Returned output.
        """
        outputs = self(**data_batch, test_mode=False)
        loss, log_vars = self.parse_losses(outputs.pop('losses'))

        # optimize
        optimizer['generator'].zero_grad()
        loss.backward()
        optimizer['generator'].step()

        outputs.update({'log_vars': log_vars})
        return outputs

    def val_step(self, data_batch, **kwargs):
        """Validation step.

        Args:
            data_batch (dict): A batch of data.
            kwargs (dict): Other arguments for ``val_step``.

        Returns:
            dict: Returned output.
        """
        output = self.forward_test(**data_batch, **kwargs)
        return output
