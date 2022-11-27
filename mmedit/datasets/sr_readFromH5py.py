# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from torch.utils.data import Dataset

from .base_sr_dataset import BaseSRDataset
from .registry import DATASETS
from .pipelines import Compose

@DATASETS.register_module()
class SRH5pyDataset(Dataset):
    """General paired image folder dataset for image restoration.

    The dataset loads lq (Low Quality) and gt (Ground-Truth) image pairs,
    applies specified transforms and finally returns a dict containing paired
    data and other information.

    This is the "folder mode", which needs to specify the lq folder path and gt
    folder path, each folder containing the corresponding images.
    Image lists will be generated automatically. You can also specify the
    filename template to match the lq and gt pairs.

    For example, we have two folders with the following structures:

    ::

        data_root
        ├── lq
        │   ├── 0001_x4.png
        │   ├── 0002_x4.png
        ├── gt
        │   ├── 0001.png
        │   ├── 0002.png

    then, you need to set:

    .. code-block:: python

        lq_folder = data_root/lq
        gt_folder = data_root/gt
        filename_tmpl = '{}_x4'

    Args:
        lq_folder (str | :obj:`Path`): Path to a lq folder.
        gt_folder (str | :obj:`Path`): Path to a gt folder.
        pipeline (List[dict | callable]): A sequence of data transformations.
        scale (int): Upsampling scale ratio.
        test_mode (bool): Store `True` when building test dataset.
            Default: `False`.
        filename_tmpl (str): Template for each filename. Note that the
            template excludes the file extension. Default: '{}'.
    """

    def __init__(self,
                 filename,
                 pipeline,
                 scale,
                 test_mode=False,
                 filename_tmpl='{}'):
        # super().__init__(pipeline, scale, test_mode)
        self.filename = filename
        self.pipeline = Compose(pipeline)


    def __len__(self):
        """Length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        with h5py.File(self.filename, 'r') as f:
            
            return len(f[self.info[0]][self.info[1]][self.info[2].split('.')[0]][self.info[3]])-3

    def __getitem__(self, index):
        with h5py.File(self.filename, 'r') as f:
            vol = f[self.info[0]][self.info[1]][self.info[2].split('.')[0]][self.info[3]]
            lr = np.array(vol[str(index)])
            return self.pipeline(dict(lr=lr))
    # def __getitem__(self, idx):
    #     """Get item at each call.

    #     Args:
    #         idx (int): Index for getting each item.
    #     """
    #     if self.test_mode:
    #         return self.prepare_test_data(idx)

    #     return self.prepare_train_data(idx)
