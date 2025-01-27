# -*- coding: utf-8 -*-

# An example of adopting nn_extractor into the programming workflow.
#
# Code adopted from https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/inference/predict_from_raw_data.py
from typing import Optional

import argparse

import numpy as np
import torch
from torch._dynamo import OptimizedModule
from tqdm import tqdm
import itertools

from acvl_utils.cropping_and_padding.padding import pad_nd_image

import os
import os.path

from batchgenerators.utilities.file_and_folder_operations import load_json, save_pickle


from nnunetv2.configuration import default_num_processes

from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor as baseNNUNetPredictor
from nnunetv2.inference.data_iterators import PreprocessAdapterFromNpy
from nnunetv2.inference import export_prediction
from nnunetv2.inference.sliding_window_prediction import compute_gaussian

from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.file_path_utilities import get_output_folder
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager

import nn_extractor
from nn_extractor import NNExtractor

from nn_extractor import profile
from nn_extractor import cfg


def export_prediction_from_logits(predicted_array_or_file: np.ndarray | torch.Tensor, properties_dict: dict,
                                  configuration_manager: ConfigurationManager,
                                  plans_manager: PlansManager,
                                  dataset_json_dict_or_file: dict | str, output_file_truncated: str,
                                  save_probabilities: bool = False, extractor: NNExtractor = None):

    if isinstance(dataset_json_dict_or_file, str):
        dataset_json_dict_or_file = load_json(dataset_json_dict_or_file)

    label_manager = plans_manager.get_label_manager(dataset_json_dict_or_file)
    ret = export_prediction.convert_predicted_logits_to_segmentation_with_correct_shape(
        predicted_array_or_file, plans_manager, configuration_manager, label_manager, properties_dict,
        return_probabilities=save_probabilities
    )
    del predicted_array_or_file

    # save
    if save_probabilities:
        segmentation_final, probabilities_final = ret
        np.savez_compressed(output_file_truncated + '.npz', probabilities=probabilities_final)
        save_pickle(properties_dict, output_file_truncated + '.pkl')
        if extractor is not None:
            extractor.add_outputs(
                data={'segmentation': segmentation_final, 'probability': probabilities_final},
                name='final',
            )
        del probabilities_final, ret
    else:
        segmentation_final = ret
        if extractor is not None:
            extractor.add_outputs(data={'segmentation': segmentation_final}, name='final')
        del ret

    rw = plans_manager.image_reader_writer_class()
    rw.write_seg(segmentation_final, output_file_truncated + dataset_json_dict_or_file['file_ending'],
                 properties_dict)


class nnUNetPredictor(baseNNUNetPredictor):
    '''
    nnUNetPredictor with NNExtractor
    '''

    '''
    extractor

    The procedure of prediction in nnUNetPredictor is sliding-window based prediction.
    Therefore, instead of directly having nnUNetPredictor.extractor registers forward_hooks.
    There are several sub-extractors registering forward_hooks.
    nnUNetPredictor.extractor only adds inputs, postprocess, outputs, and the sub-extractors.
    '''
    extractor: Optional[NNExtractor] = None

    '''
    nn_extractor_name

    name of the extractor (in)
    '''
    nn_extractor_name: str = ''

    def __init__(
            self,
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            device: torch.device = torch.device('cuda'),
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=True,
            nn_extractor_name: str = '',
    ):
        super().__init__(
            tile_step_size,
            use_gaussian,
            use_mirroring,
            perform_everything_on_device,
            device,
            verbose,
            verbose_preprocessing,
            allow_tqdm,
        )
        self.nn_extractor_name = nn_extractor_name

    def predict_single_npy_array(
            self,
            input_image: np.ndarray,
            image_properties: dict,
            segmentation_previous_stage: Optional[np.ndarray] = None,
            output_file_truncated: str = None,
            save_or_return_probabilities: bool = False,
            prompt: str = ''):
        '''
        predict single npy array

        * always return results with probability.
        '''
        # self.extractor add image as input.
        self.extractor = NNExtractor(name=prompt)
        self.extractor.add_inputs(
            name=prompt,
            data={'image': input_image, 'props': image_properties, 'previous_stage': segmentation_previous_stage},
        )

        ppa = PreprocessAdapterFromNpy(
            [input_image],
            [segmentation_previous_stage],
            [image_properties],
            [output_file_truncated],
            self.plans_manager,
            self.dataset_json,
            self.configuration_manager,
            num_threads_in_multithreaded=1,
            verbose=self.verbose,
        )

        if self.verbose:
            print('preprocessing')
        dct = next(ppa)

        # self.extractor add dct.data and dct.data_properties in preprocess.
        properties_dict = dct['data_properties']
        self.extractor.add_preprocess(
            name=prompt,
            data={'image': dct['data'].detach().to('cpu').numpy(), 'props': properties_dict},
        )

        if self.verbose:
            print('predicting')

        predicted_logits = self.predict_logits_from_preprocessed_data(dct['data'], prompt).cpu()

        if output_file_truncated is not None:
            export_prediction_from_logits(predicted_logits, dct['data_properties'], self.configuration_manager,
                                          self.plans_manager, self.dataset_json, output_file_truncated,
                                          save_or_return_probabilities, self.extractor)
            self.extractor.remove_hook()
            self.extractor.save()
        else:
            ret = export_prediction.convert_predicted_logits_to_segmentation_with_correct_shape(
                predicted_logits, self.plans_manager,
                self.configuration_manager,
                self.label_manager,
                dct['data_properties'],

                return_probabilities=save_or_return_probabilities)

            segmentation = None
            probability = None
            if save_or_return_probabilities:
                segmentation, probability = ret[0], ret[1]
            else:
                segmentation = ret

            # self.extractor add postprocess.
            self.extractor.add_postprocess(
                name='correct-shape',
                data={
                    'predicted_logits': predicted_logits,
                    'segmentation': segmentation,
                    'probability': probability,
                    'spacing': properties_dict['spacing'],
                    'transpose_forward': self.plans_manager.transpose_forward,
                    'shape_after_cropping_and_before_resampling': properties_dict['shape_after_cropping_and_before_resampling'],  # noqa
                    'shape_before_cropping': properties_dict['shape_before_cropping'],
                    'bbox_used_for_cropping': properties_dict['bbox_used_for_cropping'],
                })

            # self.extractor add ret as outputs.
            self.extractor.add_outputs(name=prompt, data={'segmentation': segmentation, 'probability': probability})

            self.extractor.remove_hook()
            self.extractor.save()

            return ret

    def predict_logits_from_preprocessed_data(self, data: torch.Tensor, prompt: str) -> torch.Tensor:
        """
        IMPORTANT! IF YOU ARE RUNNING THE CASCADE, THE SEGMENTATION FROM THE PREVIOUS STAGE MUST ALREADY BE STACKED ON
        TOP OF THE IMAGE AS ONE-HOT REPRESENTATION! SEE PreprocessAdapter ON HOW THIS SHOULD BE DONE!

        RETURNED LOGITS HAVE THE SHAPE OF THE INPUT. THEY MUST BE CONVERTED BACK TO THE ORIGINAL IMAGE SIZE.
        SEE convert_predicted_logits_to_segmentation_with_correct_shape
        """
        n_threads = torch.get_num_threads()
        torch.set_num_threads(default_num_processes if default_num_processes < n_threads else n_threads)
        prediction = None

        for idx2, params in enumerate(self.list_of_parameters):

            # messing with state dict names...
            if not isinstance(self.network, OptimizedModule):
                self.network.load_state_dict(params)
            else:
                self.network._orig_mod.load_state_dict(params)

            # why not leave prediction on device if perform_everything_on_device? Because this may cause the
            # second iteration to crash due to OOM. Grabbing that with try except cause way more bloated code than
            # this actually saves computation time
            if prediction is None:
                prediction = self.predict_sliding_window_return_logits(data, f'{prompt}-{idx2}').to('cpu')
            else:
                prediction += self.predict_sliding_window_return_logits(data, f'{prompt}-{idx2}').to('cpu')

        if len(self.list_of_parameters) > 1:
            prediction /= len(self.list_of_parameters)

        if self.verbose:
            print('Prediction done')
        torch.set_num_threads(n_threads)
        return prediction

    def predict_sliding_window_return_logits(self, input_image: torch.Tensor, prompt: str) -> np.ndarray | torch.Tensor:
        with torch.no_grad():
            assert isinstance(input_image, torch.Tensor)
            self.network = self.network.to(self.device)
            self.network.eval()

            empty_cache(self.device)

            # Autocast can be annoying
            # If the device_type is 'cpu' then it's slow as heck on some CPUs (no auto bfloat16 support detection)
            # and needs to be disabled.
            # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False
            # is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
            # So autocast will only be active if we have a cuda device.
            with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
                assert input_image.ndim == 4, 'input_image must be a 4D np.ndarray or torch.Tensor (c, x, y, z)'

                if self.verbose:
                    print(f'Input shape: {input_image.shape}')
                    print("step_size:", self.tile_step_size)
                    print("mirror_axes:", self.allowed_mirroring_axes if self.use_mirroring else None)

                # if input_image is smaller than tile_size we need to pad it to tile_size.
                data, slicer_revert_padding = pad_nd_image(input_image, self.configuration_manager.patch_size,
                                                           'constant', {'value': 0}, True,
                                                           None)

                slicers: list[slice] = self._internal_get_sliding_window_slicers(data.shape[1:])
                cfg.logger.info(f'data: {data.shape} slicers: ({slicers}/{len(slicers)})')

                if self.perform_everything_on_device and self.device != 'cpu':
                    # we need to try except here because we can run OOM in which case we need to fall back to CPU
                    # as a results device
                    try:
                        predicted_logits = self._internal_predict_sliding_window_return_logits(
                            data, slicers,
                            self.perform_everything_on_device,
                            prompt=prompt)
                    except RuntimeError:
                        print('Prediction on device was unsuccessful, probably due to a lack of memory. Moving results arrays to CPU')  # noqa
                        empty_cache(self.device)
                        predicted_logits = self._internal_predict_sliding_window_return_logits(
                            data,
                            slicers,
                            False,
                            prompt=prompt,
                        )
                else:
                    predicted_logits = self._internal_predict_sliding_window_return_logits(
                        data,
                        slicers,
                        self.perform_everything_on_device,
                        prompt=prompt,
                    )

                empty_cache(self.device)
                # revert padding
                predicted_logits = predicted_logits[(slice(None), *slicer_revert_padding[1:])]

                # self.extractor revert padding.
                self.extractor.add_postprocess(name='revert-padding', data={'predicted_logits': predicted_logits})

        return predicted_logits

    def _internal_predict_sliding_window_return_logits(
        self,
        data: torch.Tensor,
        slicers: list[slice],
        do_on_device: bool = True,
        prompt: str = '',
    ):
        predicted_logits = n_predictions = prediction = gaussian = workon = None
        results_device = self.device if do_on_device else torch.device('cpu')

        try:
            empty_cache(self.device)

            # move data to device
            if self.verbose:
                print(f'move image to device {results_device}')
            data = data.to(results_device)

            # preallocate arrays
            if self.verbose:
                print(f'preallocating results arrays on device {results_device}')
            predicted_logits = torch.zeros((self.label_manager.num_segmentation_heads, *data.shape[1:]),
                                           dtype=torch.half,
                                           device=results_device)
            n_predictions = torch.zeros(data.shape[1:], dtype=torch.half, device=results_device)

            if self.use_gaussian:
                gaussian = compute_gaussian(tuple(self.configuration_manager.patch_size), sigma_scale=1. / 8,
                                            value_scaling_factor=10,
                                            device=results_device)
            else:
                gaussian = 1

            if not self.allow_tqdm and self.verbose:
                print(f'running prediction: {len(slicers)} steps')

            # sub-extractor
            for idx, sl in tqdm(list(enumerate(slicers)), disable=not self.allow_tqdm):
                workon = data[sl][None]
                workon = workon.to(self.device)

                # sub-extractor add workon and slicer as inputs.
                sub_extractor = NNExtractor(f'{prompt}-workon-{idx}')
                sub_extractor.add_inputs(name=f'workon-{idx}', data={'workon': workon, 'slicer': list(sl)})

                prediction = self._internal_maybe_mirror_and_predict(workon, sub_extractor)[0].to(results_device)

                # extractor add sub-extractor
                self.extractor.add_extractor(extractor=sub_extractor)

                if self.use_gaussian:
                    prediction *= gaussian
                predicted_logits[sl] += prediction
                n_predictions[sl[1:]] += gaussian

                # self.extractor add postprocess for predicted-logits and n-predictions
                self.extractor.add_postprocess(
                    name=f'workon-{idx}',
                    data={
                        'predicted_logits': predicted_logits,
                        'n_predictions': n_predictions,
                        'prediction': prediction,
                        'gaussian': gaussian,
                    },
                )

                sub_extractor.remove_hook()

            predicted_logits /= n_predictions

            # self.extractor add postprocess for normalize
            self.extractor.add_postprocess(
                name='normalize',
                data={'predicted_logits': predicted_logits, 'n_predictions': n_predictions},
            )
            # check for infs
            if torch.any(torch.isinf(predicted_logits)):
                raise RuntimeError('Encountered inf in predicted array. Aborting... If this problem persists, '
                                   'reduce value_scaling_factor in compute_gaussian or increase the dtype of '
                                   'predicted_logits to fp32')
        except Exception as e:
            del predicted_logits, n_predictions, prediction, gaussian, workon
            predicted_logits = None
            empty_cache(self.device)
            empty_cache(results_device)
            raise e

        return predicted_logits

    def _internal_maybe_mirror_and_predict(self, x: torch.Tensor, sub_extractor: NNExtractor) -> torch.Tensor:
        # sub-extractor register forward hooks.
        sub_extractor.register_forward_hook(self.network)

        mirror_axes = self.allowed_mirroring_axes if self.use_mirroring else None
        profile.profile_start('network')
        prediction = self.network(x)
        cfg.logger.info(f'x: {x.device} prediction: {prediction.device}')
        profile.profile_stop('network')
        sub_extractor.forward_snapshot()
        profile.report('forward')

        if mirror_axes is not None:
            # check for invalid numbers in mirror_axes
            # x should be 5d for 3d images and 4d for 2d. so the max value of mirror_axes cannot exceed len(x.shape) - 3
            assert max(mirror_axes) <= x.ndim - 3, 'mirror_axes does not match the dimension of the input!'

            mirror_axes = [m + 2 for m in mirror_axes]
            axes_combinations = [
                c for i in range(len(mirror_axes)) for c in itertools.combinations(mirror_axes, i + 1)
            ]
            for idx, axes in enumerate(axes_combinations):
                # prediction += torch.flip(self.network(torch.flip(x, axes)), axes)
                flipped_x = torch.flip(x, axes)
                # sub-extractor add mirroring preprocess
                sub_extractor.add_preprocess(name=f'mirror-{idx}', data={'flipped': flipped_x, 'axes': axes})
                profile.profile_start('network')
                each_prediction = self.network(flipped_x)
                cfg.logger.info(f'flipped_x: {flipped_x.device} each_prediction: {each_prediction.device}')
                profile.profile_stop('network')
                sub_extractor.forward_snapshot()

                profile.report(f'mirror-{idx}')

                unflipped_prediction = torch.flip(each_prediction, axes)
                prediction += unflipped_prediction

                # sub-extractor add mirroring postrocess
                sub_extractor.add_postprocess(
                    name=f'mirror-{idx}',
                    data={'unflipped': unflipped_prediction, 'axes': axes, 'prediction': prediction},
                )

            n_axes_combinations_plus_1 = len(axes_combinations) + 1
            prediction /= n_axes_combinations_plus_1
            sub_extractor.add_postprocess(
                name='mirror-normalize',
                data={'prediction': prediction, 'n_axes_combinations_plus_1': n_axes_combinations_plus_1},
            )

        return prediction


def _determine_filenames(filename: str) -> tuple[str, list[str]]:
    the_basename = os.path.basename(filename)
    filename_list = the_basename.split('.')
    filename0_list = filename_list[0].split('_')
    filename_prefix = '_'.join(filename0_list[:-1])
    the_dirname = os.path.dirname(filename)

    filenames = list(filter(lambda x: x.startswith(filename_prefix), os.listdir(the_dirname)))
    full_filenames = list(map(lambda x: os.sep.join([the_dirname, x]), filenames))

    cfg.logger.info(f'filename: {filename} filename_prefix: {filename_prefix}, filenames: {filenames}')

    return filename_prefix, full_filenames


def main():
    parser = argparse.ArgumentParser(description='Use this to run inference with nnU-Net. This function is used when '
                                     'you want to manually specify a folder containing a trained nnU-Net '
                                     'model. This is useful when the nnunet environment variables '
                                                 '(nnUNet_results) are not set.')
    parser.add_argument('-i', type=str, required=True,
                        help='input filename with _0000.nii.gz. Remember to use the correct channel numberings for your files (_0000 etc). '  # noqa
                             'File endings must be the same as the training dataset!')
    parser.add_argument('-o', type=str, required=True,
                        help='Output folder. If it does not exist it will be created. Predicted segmentations will '
                             'have the same name as their source images.')
    parser.add_argument('-d', type=str, required=True,
                        help='Dataset with which you would like to predict. You can specify either dataset name or id')
    parser.add_argument('-p', type=str, required=False, default='nnUNetPlans',
                        help='Plans identifier. Specify the plans in which the desired configuration is located. '
                             'Default: nnUNetPlans')
    parser.add_argument('-tr', type=str, required=False, default='nnUNetTrainer',
                        help='What nnU-Net trainer class was used for training? Default: nnUNetTrainer')
    parser.add_argument('-c', type=str, required=True,
                        help='nnU-Net configuration that should be used for prediction. Config must be located '
                             'in the plans specified with -p')
    parser.add_argument('-f', nargs='+', type=str, required=False, default=(0, 1, 2, 3, 4),
                        help='Specify the folds of the trained model that should be used for prediction. '
                             'Default: (0, 1, 2, 3, 4)')
    parser.add_argument('-step_size', type=float, required=False, default=0.5,
                        help='Step size for sliding window prediction. The larger it is the faster but less accurate '
                             'the prediction. Default: 0.5. Cannot be larger than 1. We recommend the default.')
    parser.add_argument('--disable_tta', action='store_true', required=False, default=False,
                        help='Set this flag to disable test time data augmentation in the form of mirroring. Faster, '
                             'but less accurate inference. Not recommended.')
    parser.add_argument('--verbose', action='store_true', help="Set this if you like being talked to. You will have "
                                                               "to be a good listener/reader.")
    parser.add_argument('--save_probabilities', action='store_true',
                        help='Set this to export predicted class "probabilities". Required if you want to ensemble '
                             'multiple configurations.')
    parser.add_argument('--continue_prediction', action='store_true',
                        help='Continue an aborted previous prediction (will not overwrite existing files)')
    parser.add_argument('-chk', type=str, required=False, default='checkpoint_final.pth',
                        help='Name of the checkpoint you want to use. Default: checkpoint_final.pth')
    parser.add_argument('-npp', type=int, required=False, default=3,
                        help='Number of processes used for preprocessing. More is not always better. Beware of '
                             'out-of-RAM issues. Default: 3')
    parser.add_argument('-nps', type=int, required=False, default=3,
                        help='Number of processes used for segmentation export. More is not always better. Beware of '
                             'out-of-RAM issues. Default: 3')
    parser.add_argument('-prev_stage_predictions', type=str, required=False, default=None,
                        help='Folder containing the predictions of the previous stage. Required for cascaded models.')
    parser.add_argument('-device', type=str, default='cuda', required=False,
                        help="Use this to set the device the inference should run with. Available options are 'cuda' "
                             "(GPU), 'cpu' (CPU) and 'mps' (Apple M1/M2). Do NOT use this to set which GPU ID! "
                             "Use CUDA_VISIBLE_DEVICES=X nnUNetv2_predict [...] instead!")
    parser.add_argument('--disable_progress_bar', action='store_true', required=False, default=False,
                        help='Set this flag to disable progress bar. Recommended for HPC environments (non interactive '
                             'jobs)')

    parser.add_argument('-N', '--nn-extractor-name', type=str, required=False, default='',
                        help='nn-extractor name.')

    parser.add_argument('-C', '--nn-extractor-config-file', type=str, required=False, default='',
                        help='nn-extractor config filename.')

    print(
        "\n#######################################################################\nPlease cite the following paper "
        "when using nnU-Net:\n"
        "Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). "
        "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. "
        "Nature methods, 18(2), 203-211.\n#######################################################################\n")
    args = parser.parse_args()
    args.f = [i if i == 'all' else int(i) for i in args.f]

    model_folder = get_output_folder(args.d, args.tr, args.p, args.c)

    os.makedirs(args.o, exist_ok=True)

    assert args.device in ['cpu', 'cuda',
                           'mps'], f'-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {args.device}.'  # noqa

    if args.device == 'cpu':
        # let's allow torch to use hella threads
        import multiprocessing
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device('cpu')
    elif args.device == 'cuda':
        # multithreading in torch doesn't help nnU-Net if run on GPU
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device('cuda')
    else:
        device = torch.device('mps')

    profile.reset()

    # init nn-extractor
    extra_params: nn_extractor.Config = {}
    if args.nn_extractor_name:
        extra_params['name'] = args.nn_extractor_name
    nn_extractor.init(args.nn_extractor_config_file, extra_params=extra_params)
    nn_extractor_name = nn_extractor.config['name']

    assert nn_extractor_name, 'nn-extract-name cannot be empty.'

    predictor = nnUNetPredictor(tile_step_size=args.step_size,
                                use_gaussian=True,
                                use_mirroring=not args.disable_tta,
                                perform_everything_on_device=True,
                                device=device,
                                verbose=args.verbose,
                                verbose_preprocessing=args.verbose,
                                allow_tqdm=not args.disable_progress_bar,
                                nn_extractor_name=nn_extractor_name)
    predictor.initialize_from_trained_model_folder(
        model_folder,
        args.f,
        checkpoint_name=args.chk
    )

    filename_prefix, filenames = _determine_filenames(args.i)

    output_filename = os.sep.join([args.o, f'{filename_prefix}.nii.gz'])

    img, props = SimpleITKIO().read_images(filenames)
    _ = predictor.predict_single_npy_array(
        img,
        props,
        None,
        output_filename,
        args.save_probabilities,
        f'{nn_extractor_name}-{filename_prefix}',
    )

    profile.report()


if __name__ == '__main__':
    main()
