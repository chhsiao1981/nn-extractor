from typing import Any, Optional, Union

from nn_extractor import nii
from nn_extractor.nii import NII
from nn_extractor.nnextractor import NNExtractor
from nn_extractor.ops.pad import Pad
from nn_extractor.types import NNTensor
import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import load_json, save_pickle

from nnunetv2.configuration import default_num_processes
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager

from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape  # noqa


def export_prediction_from_logits(predicted_array_or_file: Union[np.ndarray, torch.Tensor], properties_dict: dict,  # noqa
                                  configuration_manager: ConfigurationManager,
                                  plans_manager: PlansManager,
                                  dataset_json_dict_or_file: Union[dict, str], output_file_truncated: str,  # noqa
                                  save_probabilities: bool = False,
                                  num_threads_torch: int = default_num_processes,

                                  extractor: Optional[NNExtractor] = None,
                                  label_image: Optional[NNTensor] = None,
                                  label_image_properties: Optional[dict] = None,
                                  ):
    # if isinstance(predicted_array_or_file, str):
    #     tmp = deepcopy(predicted_array_or_file)
    #     if predicted_array_or_file.endswith('.npy'):
    #         predicted_array_or_file = np.load(predicted_array_or_file)
    #     elif predicted_array_or_file.endswith('.npz'):
    #         predicted_array_or_file = np.load(predicted_array_or_file)['softmax']
    #     os.remove(tmp)

    if isinstance(dataset_json_dict_or_file, str):
        dataset_json_dict_or_file = load_json(dataset_json_dict_or_file)

    label_manager = plans_manager.get_label_manager(dataset_json_dict_or_file)
    ret = convert_predicted_logits_to_segmentation_with_correct_shape(
        predicted_array_or_file, plans_manager, configuration_manager, label_manager, properties_dict,  # noqa
        return_probabilities=save_probabilities, num_threads_torch=num_threads_torch
    )

    # extractor add postprocess
    _extractor_add_postprocess(
        predicted_array_or_file,
        properties_dict,
        plans_manager,
        save_probabilities,
        ret,

        extractor,
    )

    del predicted_array_or_file

    # save
    if save_probabilities:
        segmentation_final, probabilities_final = ret
        np.savez_compressed(output_file_truncated + '.npz', probabilities=probabilities_final)
        save_pickle(properties_dict, output_file_truncated + '.pkl')

        # extractor add outputs
        _extractor_add_outputs(
            segmentation_final,
            probabilities_final,
            label_image,
            label_image_properties,

            extractor,
        )

        del probabilities_final, ret
    else:
        segmentation_final = ret

        # extractor add outputs
        _extractor_add_outputs(
            segmentation_final,
            None,
            label_image,
            label_image_properties,

            extractor,
        )

        del ret

    rw = plans_manager.image_reader_writer_class()
    rw.write_seg(segmentation_final, output_file_truncated + dataset_json_dict_or_file['file_ending'],  # noqa
                 properties_dict)


def _extractor_add_postprocess(
    predicted_array_or_file: np.ndarray | torch.Tensor,
    properties_dict: dict,
    plans_manager: PlansManager,
    save_probabilities: bool,
    ret: np.ndarray | tuple[np.ndarray, np.ndarray | Any],

    extractor: Optional[NNExtractor],
):
    if extractor is None:
        return

    segmentation: Optional[np.ndarray] = None
    probability: Optional[np.ndarray] = None
    if save_probabilities:
        segmentation, probability = ret[0], ret[1]
    else:
        segmentation = ret

    slicer_revert_padding = properties_dict['bbox_used_for_cropping']
    pad_segmentation = Pad(img=segmentation, slicer_revert_padding=slicer_revert_padding)
    pad_probability = None
    if probability is not None:
        pad_probability = Pad(img=probability, slicer_revert_padding=slicer_revert_padding)

    extractor.add_postprocess(
            name='correct-shape',
            data={
                'predicted_logits': predicted_array_or_file,
                'segmentation': pad_segmentation,
                'probability': pad_probability,
                'spacing': properties_dict['spacing'],
                'transpose_forward': plans_manager.transpose_forward,
                'shape_after_cropping_and_before_resampling': properties_dict['shape_after_cropping_and_before_resampling'],  # noqa
                'shape_before_cropping': properties_dict['shape_before_cropping'],
                'bbox_used_for_cropping': properties_dict['bbox_used_for_cropping'],
            })


def _extractor_add_outputs(
    segmentation_final: np.ndarray,
    probabilities_final: Optional[np.ndarray],
    label_image: NNTensor,
    label_image_properties: NNTensor,

    extractor: Optional[NNExtractor],
):
    if extractor is None:
        return

    label_nii: Optional[NII] = None
    if label_image is not None and label_image_properties is not None:
        label_nii = nii.from_sitk_image_props(label_image, label_image_properties)

    outputs_data = {'segmentation': segmentation_final}
    if probabilities_final is not None:
        outputs_data['probability'] = probabilities_final
    if label_nii is not None:
        outputs_data['ground_truth'] = label_nii

    extractor.add_outputs(
        data=outputs_data,
        name='final',
    )
