from typing import Optional, Self, Union, List

import numpy as np

from batchgenerators.dataloading.data_loader import DataLoader

from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor as orig_DefaultPreprocessor
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.inference.data_iterators import PreprocessAdapterFromNpy as orig_PreprocessAdapterFromNpy

from nn_extractor import cfg
from nn_extractor.nnextractor import NNExtractor
from .default_preprocessor import DefaultPreprocessor


class PreprocessAdapterFromNpy(orig_PreprocessAdapterFromNpy):
    extractor: Optional[NNExtractor] = None

    def __init__(
        self: Self,
        list_of_images: List[np.ndarray],
        list_of_segs_from_prev_stage: Union[List[np.ndarray], None],
        list_of_image_properties: List[dict],
        truncated_ofnames: Union[List[str], None],
        plans_manager: PlansManager,
        dataset_json: dict,
        configuration_manager: ConfigurationManager,
        num_threads_in_multithreaded: int = 1,
        verbose: bool = False,

        extractor: Optional[NNExtractor] = None,
    ):
        self.extractor = extractor

        # extractor: replace nnunetv2.DefaultPreprocessor to DefaultPreprocessor with extractor.
        preprocessor = None
        if configuration_manager.preprocessor_class == orig_DefaultPreprocessor:
            cfg.logger.info('PreprocessAdapterFromNpy: to replace DefaultPreprocessor to DefaultPreprocessor with extractor.')  # noqa
            preprocessor = DefaultPreprocessor(verbose=verbose, extractor=extractor)
        else:
            cfg.logger.info(f'PreprocessAdapterFromNpy: to remain original configuration_manager.preprocessor_class: {configuration_manager.preprocessor_class}')  # noqa
            preprocessor = configuration_manager.preprocessor_class(
                verbose=verbose,
                extractor=extractor,
            )

        self.preprocessor, self.plans_manager, self.configuration_manager, self.dataset_json, self.truncated_ofnames = \
            preprocessor, plans_manager, configuration_manager, dataset_json, truncated_ofnames

        self.label_manager = plans_manager.get_label_manager(dataset_json)

        if list_of_segs_from_prev_stage is None:
            list_of_segs_from_prev_stage = [None] * len(list_of_images)
        if truncated_ofnames is None:
            truncated_ofnames = [None] * len(list_of_images)

        DataLoader.__init__(
            self,
            list(zip(list_of_images, list_of_segs_from_prev_stage,
                 list_of_image_properties, truncated_ofnames)),
            1, num_threads_in_multithreaded,
            seed_for_shuffle=1, return_incomplete=True,
            shuffle=False, infinite=False, sampling_probabilities=None)

        self.indices = list(range(len(list_of_images)))
