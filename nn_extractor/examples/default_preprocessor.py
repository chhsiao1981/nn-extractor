#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import math
import multiprocessing
import shutil
from time import sleep
from typing import Self, Tuple

import SimpleITK
import numpy as np
import pandas as pd
from batchgenerators.utilities.file_and_folder_operations import *
from tqdm import tqdm

import nnunetv2
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw
from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero
from nnunetv2.preprocessing.resampling.default_resampling import compute_new_shape
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.utils import get_filenames_of_train_images_and_targets
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor as orig_DefaultPreprocessor

from nn_extractor import utils
from nn_extractor.nnextractor import NNExtractor
from nn_extractor.ops.crop import Crop
from nn_extractor.ops.spacing import Spacing
import copy


class DefaultPreprocessor(orig_DefaultPreprocessor):
    extractor: Optional[NNExtractor] = None

    def __init__(self, verbose: bool = True, extractor: Optional[NNExtractor] = None):
        self.verbose = verbose

        self.extractor = extractor
        """
        Everything we need is in the plans. Those are given when run() is called
        """

    def run_case_npy(
        self: Self,
        data: np.ndarray,
        seg: Union[np.ndarray, None],
        properties: dict,
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        dataset_json: Union[dict, str],
    ):
        # let's not mess up the inputs!
        data = data.astype(np.float32)  # this creates a copy
        if seg is not None:
            assert data.shape[1:] == seg.shape[1:], "Shape mismatch between image and segmentation. Please fix your dataset and make use of the --verify_dataset_integrity flag to ensure everything is correct"
            seg = np.copy(seg)

        has_seg = seg is not None

        # apply transpose_forward, this also needs to be applied to the spacing!
        data = data.transpose([0, *[i + 1 for i in plans_manager.transpose_forward]])
        if seg is not None:
            seg = seg.transpose([0, *[i + 1 for i in plans_manager.transpose_forward]])
        original_spacing = [properties['spacing'][i] for i in plans_manager.transpose_forward]

        # crop, remember to store size before cropping!
        shape_before_cropping = data.shape[1:]
        properties['shape_before_cropping'] = shape_before_cropping
        # this command will generate a segmentation. This is important because of the nonzero mask which we may need
        orig_data = data
        orig_seg = seg
        data, seg, bbox = crop_to_nonzero(data, seg)
        properties['bbox_used_for_cropping'] = bbox
        # print(data.shape, seg.shape)
        properties['shape_after_cropping_and_before_resampling'] = data.shape[1:]

        # self.extractor: add crop data in preprocess.
        region_sar = utils.slice_spl_to_sar(bbox, orig_data.shape)
        crop_data = {
            'img': Crop(img=data, region_sar=region_sar),
            'props': properties,
            'region_sar': region_sar,
        }
        if has_seg:
            crop_data['seg'] = Crop(
                img=seg,
                region_sar=region_sar,
            )

        self.extractor.add_preprocess(
            name=f'crop-img',
            data=crop_data,
        )

        # resample
        target_spacing = configuration_manager.spacing  # this should already be transposed

        if len(target_spacing) < len(data.shape[1:]):
            # target spacing for 2d has 2 entries but the data and original_spacing have three because everything is 3d
            # in 2d configuration we do not change the spacing between slices
            target_spacing = [original_spacing[0]] + target_spacing
        new_shape = compute_new_shape(data.shape[1:], original_spacing, target_spacing)

        # normalize
        # normalization MUST happen before resampling or we get huge problems with resampled nonzero masks no
        # longer fitting the images perfectly!
        data = self._normalize(data, seg, configuration_manager,
                               plans_manager.foreground_intensity_properties_per_channel)

        # print('current shape', data.shape[1:], 'current_spacing', original_spacing,
        #       '\ntarget shape', new_shape, 'target_spacing', target_spacing)
        old_shape = data.shape[1:]
        data = configuration_manager.resampling_fn_data(
            data, new_shape, original_spacing, target_spacing)
        seg = configuration_manager.resampling_fn_seg(
            seg, new_shape, original_spacing, target_spacing)

        original_spacing_ras = copy.deepcopy(original_spacing)
        original_spacing_ras.reverse()

        target_spacing_ras = copy.deepcopy(target_spacing)
        target_spacing_ras.reverse()

        # self.extractor: add spacing data
        spacing_data = {
            'img': Spacing(img=data, spacing_ras=target_spacing_ras),
            'orig_spacing_ras': original_spacing_ras,
            'orig_shape_sar': old_shape,
        }
        if has_seg:
            spacing_data['seg'] = Spacing(img=seg, spacing_ras=target_spacing_ras)

        # self.extractor: add scaling in preprocess.
        self.extractor.add_preprocess(
            name=f'spacing-img',
            data=spacing_data,
        )

        if self.verbose:
            print(f'old shape: {old_shape}, new_shape: {new_shape}, old_spacing: {original_spacing}, '
                  f'new_spacing: {target_spacing}, fn_data: {configuration_manager.resampling_fn_data}')

        # if we have a segmentation, sample foreground locations for oversampling and add those to properties
        if has_seg:
            # reinstantiating LabelManager for each case is not ideal. We could replace the dataset_json argument
            # with a LabelManager Instance in this function because that's all its used for. Dunno what's better.
            # LabelManager is pretty light computation-wise.
            label_manager = plans_manager.get_label_manager(dataset_json)
            collect_for_this = label_manager.foreground_regions if label_manager.has_regions \
                else label_manager.foreground_labels

            # when using the ignore label we want to sample only from annotated regions. Therefore we also need to
            # collect samples uniformly from all classes (incl background)
            if label_manager.has_ignore_label:
                collect_for_this.append([-1] + label_manager.all_labels)

            # no need to filter background in regions because it is already filtered in handle_labels
            # print(all_labels, regions)
            properties['class_locations'] = self._sample_foreground_locations(seg, collect_for_this,
                                                                              verbose=self.verbose)
            seg = self.modify_seg_fn(seg, plans_manager, dataset_json, configuration_manager)
        if np.max(seg) > 127:
            seg = seg.astype(np.int16)
        else:
            seg = seg.astype(np.int8)
        return data, seg, properties
