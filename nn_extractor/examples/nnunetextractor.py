import itertools
import os
from queue import Queue
import re
from threading import Thread
from typing import Optional, Self, Union

from pydantic import BaseModel, Field

from nn_extractor import argparse, cfg, nii
import nn_extractor
from nn_extractor.nnextractor import NNExtractor
from nn_extractor.ops.crop import Crop
from nn_extractor.ops.flip import Flip
from nn_extractor.ops.pad import Pad
import numpy as np
import torch
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, isdir
from torch._dynamo import OptimizedModule
from tqdm import tqdm

from nnunetv2.configuration import default_num_processes
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from nnunetv2.inference.data_iterators import PreprocessAdapterFromNpy
from nnunetv2.inference.export_prediction import \
    convert_predicted_logits_to_segmentation_with_correct_shape
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.utilities.file_path_utilities import get_output_folder
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.utils import create_lists_from_splitted_dataset_folder

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor as baseNNUNetPredictor

from . import export_prediction


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

    def predict_single_npy_array(self, input_image: np.ndarray, image_properties: dict,
                                 segmentation_previous_stage: np.ndarray = None,
                                 output_file_truncated: str = None,
                                 save_or_return_probabilities: bool = False,

                                 nnextractor_name: str = '',
                                 label_image: Optional[np.ndarray] = None,
                                 label_image_properties: Optional[dict] = None,
                                 ):
        """
        WARNING: SLOW. ONLY USE THIS IF YOU CANNOT GIVE NNUNET MULTIPLE IMAGES AT ONCE FOR SOME REASON.


        input_image: Make sure to load the image in the way nnU-Net expects! nnU-Net is trained on a certain axis
                     ordering which cannot be disturbed in inference,
                     otherwise you will get bad results. The easiest way to achieve that is to use the same I/O class
                     for loading images as was used during nnU-Net preprocessing! You can find that class in your
                     plans.json file under the key "image_reader_writer". If you decide to freestyle, know that the
                     default axis ordering for medical images is the one from SimpleITK. If you load with nibabel,  # noqa
                     you need to transpose your axes AND your spacing from [x,y,z] to [z,y,x]!
        image_properties must only have a 'spacing' key!
        """
        # init self.extractor
        self.extractor = NNExtractor(name=nnextractor_name)

        # self.extractor add image as input.
        self.extractor.add_inputs(
            name=nnextractor_name,
            data={
                'nii': nii.from_sitk_image_props(input_image, image_properties),
                'previous_stage': segmentation_previous_stage,
            },
        )

        ppa = PreprocessAdapterFromNpy([input_image], [segmentation_previous_stage], [image_properties],  # noqa
                                       [output_file_truncated],
                                       self.plans_manager, self.dataset_json, self.configuration_manager,  # noqa
                                       num_threads_in_multithreaded=1, verbose=self.verbose)
        if self.verbose:
            print('preprocessing')
        dct = next(ppa)

        # self.extractor add dct.data and dct.data_properties in preprocess.
        properties_dict = dct['data_properties']
        img = dct['data'].detach().to('cpu').numpy()
        crop_region = properties_dict['bbox_used_for_cropping']
        self.extractor.add_preprocess(
            name=f'{nnextractor_name}-crop',
            data={'img': Crop(img=img, region=crop_region), 'props': properties_dict},
        )

        if self.verbose:
            print('predicting')
        predicted_logits = self.predict_logits_from_preprocessed_data(
            data=dct['data'],
            nnextractor_name=nnextractor_name,
        ).cpu()

        if self.verbose:
            print('resampling to original shape')
        if output_file_truncated is not None:
            export_prediction.export_prediction_from_logits(
                predicted_logits, dct['data_properties'],
                self.configuration_manager,
                self.plans_manager,
                self.dataset_json,
                output_file_truncated,
                save_or_return_probabilities,

                extractor=self.extractor,
                label_image=label_image,
                label_image_properties=label_image_properties,
            )

            # remove hooks and save.
            self.extractor.remove_hook()
            self.extractor.save()

        else:
            ret = convert_predicted_logits_to_segmentation_with_correct_shape(predicted_logits, self.plans_manager,  # noqa
                                                                              self.configuration_manager,  # noqa
                                                                              self.label_manager,
                                                                              dct['data_properties'],  # noqa
                                                                              return_probabilities=  # fmt: off  # noqa
                                                                              save_or_return_probabilities)  # noqa
            export_prediction.extractor_add_postprocess(
                predicted_array_or_file=predicted_logits,
                properties_dict=properties_dict,
                plans_manager=self.plans_manager,
                save_probabilities=save_or_return_probabilities,
                ret=ret,

                extractor=self.extractor,
            )

            export_prediction.extractor_add_outputs(
                save_probabilities=save_or_return_probabilities,
                ret=ret,
                label_image=label_image,
                label_image_properties=label_image_properties,
            )

            # remove hooks and save.
            self.extractor.remove_hook()
            self.extractor.save()

            if save_or_return_probabilities:
                return ret[0], ret[1]
            else:
                return ret

    @torch.inference_mode()
    def predict_logits_from_preprocessed_data(self, data: torch.Tensor, nnextractor_name: str) -> torch.Tensor:  # noqa
        """
        IMPORTANT! IF YOU ARE RUNNING THE CASCADE, THE SEGMENTATION FROM THE PREVIOUS STAGE MUST ALREADY BE STACKED ON
        TOP OF THE IMAGE AS ONE-HOT REPRESENTATION! SEE PreprocessAdapter ON HOW THIS SHOULD BE DONE!

        RETURNED LOGITS HAVE THE SHAPE OF THE INPUT. THEY MUST BE CONVERTED BACK TO THE ORIGINAL IMAGE SIZE.  # noqa
        SEE convert_predicted_logits_to_segmentation_with_correct_shape
        """
        n_threads = torch.get_num_threads()
        torch.set_num_threads(default_num_processes if default_num_processes < n_threads else n_threads)  # noqa
        prediction = None

        for idx, params in enumerate(self.list_of_parameters):

            # messing with state dict names...
            if not isinstance(self.network, OptimizedModule):
                self.network.load_state_dict(params)
            else:
                self.network._orig_mod.load_state_dict(params)

            # why not leave prediction on device if perform_everything_on_device? Because this may cause the  # noqa
            # second iteration to crash due to OOM. Grabbing that with try except cause way more bloated code than  # noqa
            # this actually saves computation time
            prompt = f'{nnextractor_name}-{idx}'
            if prediction is None:
                prediction = self.predict_sliding_window_return_logits(data, prompt).to('cpu')
            else:
                prediction += self.predict_sliding_window_return_logits(data, prompt).to('cpu')

        if len(self.list_of_parameters) > 1:
            prediction /= len(self.list_of_parameters)

        if self.verbose: print('Prediction done')  # fmt: off  # noqa
        torch.set_num_threads(n_threads)
        return prediction

    @torch.inference_mode()
    def _internal_maybe_mirror_and_predict(
        self: Self,
        x: torch.Tensor,
        sub_extractor: NNExtractor,
    ) -> torch.Tensor:
        # sub-extractor register forward hooks.
        sub_extractor.register_forward_hook(self.network)

        mirror_axes = self.allowed_mirroring_axes if self.use_mirroring else None
        prediction = self.network(x)

        # sub-extractor forward snapshot
        sub_extractor.forward_snapshot()

        if mirror_axes is not None:
            # check for invalid numbers in mirror_axes
            # x should be 5d for 3d images and 4d for 2d. so the max value of mirror_axes cannot exceed len(x.shape) - 3  # noqa
            assert max(mirror_axes) <= x.ndim - 3, 'mirror_axes does not match the dimension of the input!'  # noqa

            mirror_axes = [m + 2 for m in mirror_axes]
            axes_combinations = [
                c for i in range(len(mirror_axes)) for c in itertools.combinations(mirror_axes, i + 1)  # noqa
            ]
            for idx, axes in enumerate(axes_combinations):
                # add each_flipped for sub-extractor.
                each_flipped_x = torch.flip(x, axes)

                # sub-extractor add-preprocess
                sub_extractor.add_preprocess(
                    name=f'mirror-flip-{idx}',
                    data={'flipped': Flip(img=each_flipped_x, axes=axes)},
                )

                # add each-prediction for sub-extractor
                each_prediction = self.network(each_flipped_x)

                # sub-extractor forward snapshot
                sub_extractor.forward_snapshot()

                unflipped_prediction = torch.flip(each_prediction, axes)
                prediction += unflipped_prediction

                # sub-extractor add mirroring postrocess: unflip
                sub_extractor.add_postprocess(
                    name=f'mirror-unflip-{idx}',
                    data={
                        'each_prediction': each_prediction,
                        'unflipped': Flip(img=unflipped_prediction, axes=axes),
                        'prediction': prediction,
                    },
                )

            n_axes_combinations_plus_1 = len(axes_combinations) + 1
            prediction /= n_axes_combinations_plus_1

            # sub-extractor add mirroring postprocess: normalize
            sub_extractor.add_postprocess(
                name='mirror-normalize',
                data={
                    'prediction': prediction,
                    'n_axes_combinations_plus_1': n_axes_combinations_plus_1,
                },
            )
        return prediction

    @torch.inference_mode()
    def _internal_predict_sliding_window_return_logits(self,
                                                       data: torch.Tensor,
                                                       slicers,
                                                       do_on_device: bool = True,
                                                       prompt: str = ''
                                                       ):
        predicted_logits = n_predictions = prediction = gaussian = workon = None
        results_device = self.device if do_on_device else torch.device('cpu')

        def producer(d, slh, q):
            for s in slh:
                q.put((torch.clone(d[s][None], memory_format=torch.contiguous_format).to(self.device), s))  # noqa
            q.put('end')

        try:
            empty_cache(self.device)

            # move data to device
            if self.verbose:
                print(f'move image to device {results_device}')
            data = data.to(results_device)
            queue = Queue(maxsize=2)
            t = Thread(target=producer, args=(data, slicers, queue))
            t.start()

            # preallocate arrays
            if self.verbose:
                print(f'preallocating results arrays on device {results_device}')
            predicted_logits = torch.zeros((self.label_manager.num_segmentation_heads, *data.shape[1:]),  # noqa
                                           dtype=torch.half,
                                           device=results_device)
            n_predictions = torch.zeros(data.shape[1:], dtype=torch.half, device=results_device)

            if self.use_gaussian:
                gaussian = compute_gaussian(tuple(self.configuration_manager.patch_size), sigma_scale=1. / 8,  # noqa
                                            value_scaling_factor=10,
                                            device=results_device)
            else:
                gaussian = 1

            if not self.allow_tqdm and self.verbose:
                print(f'running prediction: {len(slicers)} steps')

            with tqdm(desc=None, total=len(slicers), disable=not self.allow_tqdm) as pbar:
                while True:
                    item = queue.get()
                    if item == 'end':
                        queue.task_done()
                        break
                    workon, sl = item

                    # sub-extractor add workon and slicer as inputs.
                    slicer_idx = pbar.n
                    sub_extractor = NNExtractor(f'{prompt}-workon-{slicer_idx}')
                    sub_extractor.add_inputs(
                        name=f'workon-{slicer_idx}',
                        data={'workon': Crop(img=workon, region=list(sl)), 'slicer': list(sl)},
                    )

                    prediction = self._internal_maybe_mirror_and_predict(workon, sub_extractor)[0].to(results_device)  # noqa

                    # extractor add sub-extractor
                    self.extractor.add_extractor(extractor=sub_extractor)

                    if self.use_gaussian:
                        prediction *= gaussian
                    predicted_logits[sl] += prediction
                    n_predictions[sl[1:]] += gaussian

                    # self.extractor add postprocess for predicted-logits and n-predictions
                    self.extractor.add_postprocess(
                        name=f'workon-{slicer_idx}',
                        data={
                            # requiring prediction and gaussian before predicted_logits as correct order of affine.
                            'prediction': prediction,
                            'gaussian': gaussian,
                            'predicted_logits': Pad(img=predicted_logits, slicer_revert_padding=list(sl)),
                            'n_predictions': n_predictions,
                        },
                    )

                    sub_extractor.remove_hook()

                    queue.task_done()
                    pbar.update()
            queue.join()

            # predicted_logits /= n_predictions
            torch.div(predicted_logits, n_predictions, out=predicted_logits)
            # check for infs
            if torch.any(torch.isinf(predicted_logits)):
                raise RuntimeError('Encountered inf in predicted array. Aborting... If this problem persists, '  # noqa
                                   'reduce value_scaling_factor in compute_gaussian or increase the dtype of '  # noqa
                                   'predicted_logits to fp32')
        except Exception as e:
            del predicted_logits, n_predictions, prediction, gaussian, workon
            empty_cache(self.device)
            empty_cache(results_device)
            raise e
        return predicted_logits  # noqa

    @torch.inference_mode()
    def predict_sliding_window_return_logits(self, input_image: torch.Tensor, prompt: str) \
            -> Union[np.ndarray, torch.Tensor]:
        assert isinstance(input_image, torch.Tensor)
        self.network = self.network.to(self.device)
        self.network.eval()

        empty_cache(self.device)

        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck on some CPUs (no auto bfloat16 support detection)  # noqa
        # and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False  # noqa
        # is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():  # noqa
            assert input_image.ndim == 4, 'input_image must be a 4D np.ndarray or torch.Tensor (c, x, y, z)'  # noqa

            if self.verbose:
                print(f'Input shape: {input_image.shape}')
                print("step_size:", self.tile_step_size)
                print("mirror_axes:", self.allowed_mirroring_axes if self.use_mirroring else None)

            # if input_image is smaller than tile_size we need to pad it to tile_size.
            data, slicer_revert_padding = pad_nd_image(input_image, self.configuration_manager.patch_size,  # noqa
                                                       'constant', {'value': 0}, True,
                                                       None)

            # extractor add preprocess: Pad
            self.extractor.add_preprocess(
                name=f'{prompt}-pad',
                data={
                    'img': Pad(img=data, slicer_revert_padding=slicer_revert_padding)
                },
            )

            slicers = self._internal_get_sliding_window_slicers(data.shape[1:])

            if self.perform_everything_on_device and self.device != 'cpu':
                # we need to try except here because we can run OOM in which case we need to fall back to CPU as a results device  # noqa
                try:
                    predicted_logits = self._internal_predict_sliding_window_return_logits(
                        data,
                        slicers,
                        self.perform_everything_on_device,

                        prompt=prompt,
                    )
                except RuntimeError:
                    print(
                        'Prediction on device was unsuccessful, probably due to a lack of memory. Moving results arrays to CPU')  # noqa
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
            the_slice = (slice(None), *slicer_revert_padding[1:])
            predicted_logits = predicted_logits[the_slice]

            # self.extractor revert padding.
            self.extractor.add_postprocess(
                name='revert-padding',
                data={'predicted_logits': Crop(img=predicted_logits, region=the_slice)},
            )

        return predicted_logits


class Args(BaseModel):
    # https://stackoverflow.com/questions/72741663/argument-parser-from-a-pydantic-model
    i: str = Field(description='input filename with _0000.nii.gz.')
    o: str = Field(description='Output folder.')
    d: str = Field(description='Dataset with which you trained.')
    p: str = Field(default='nnUNetPlans', description='Plans identifier.')
    tr: str = Field(alias='tr', default='nnUNetTrainer', description='nnUNet trainer.')
    c: str = Field(
        description='nnUNet config (2d, 3d_lowres, 3d_fullres, 3d_cascade_fullres)')
    f: str = Field(
        nargs='+',
        default=(0, 1, 2, 3, 4),
        description='folds of the trained model')
    step_size: float = Field(
        alias='step_size',
        default=0.5, description='Step size for sliding window prediction.')
    disable_tta: bool = Field(
        default=False,
        action='store_true',
        description='Set this flag to disable test time data augmentationin the form of mirroring.')
    verbose: bool = Field(
        default=False,
        action='store_true',
        description='debugging')
    save_probabilities: bool = Field(
        default=False,
        action='store_true',
        description='export predicted class "probabilities".')
    continue_prediction: bool = Field(
        default=False,
        action='store_true',
        description='Continue an aborted previous prediction.')
    chk: str = Field(
        alias='chk',
        default='checkpoint_final.pth',
        description='Name of the checkpoint.')
    npp: int = Field(
        alias='npp',
        default=3,
        description='Number of processes used for preprocessing.')
    nps: int = Field(
        alias='nps',
        default=3,
        description='Number of processes used for segmentation export.')
    prev_stage_predictions: str = Field(
        alias='prev_stage_predictions',
        default=None,
        description='Folder containing the predictions of the previous stage.'
        'Required for cascaded models.')
    device: str = Field(
        alias='device',
        default='cuda',
        description='set the device the inference should run with. (cuda, cpu, mps).'
        'Set GPU ID as environment variable CUDA_VISIBLE_DEVICES=X if needed.')
    disable_progress_bar: bool = Field(
        default=False,
        action='store_true',
        description='Set this flag to disable progress bar.')

    nn_extractor_name: str = Field(
        default='nnunetv2',
        description='nn-extractor name',
        alias='N')

    nn_extractor_config_file: str = Field(
        default='',
        description='nn-extractor config filename',
        alias='C')


def _determine_filenames(filename: str) -> tuple[str, list[str], str]:
    the_basename = os.path.basename(filename)
    filename_list = the_basename.split('.')
    filename0_list = filename_list[0].split('_')
    identifier = '_'.join(filename0_list[:-1])
    filename_postfix = '.'.join(filename_list[1:])

    the_dirname = os.path.dirname(filename)

    # https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/inference/predict_from_raw_data.py#L171
    list_of_full_filenames = create_lists_from_splitted_dataset_folder(
        the_dirname, file_ending=f'.{filename_postfix}', identifiers=[identifier], num_processes=1)
    if len(list_of_full_filenames) != 1:
        raise Exception(f'_determine_filenames: invalid list_of_full_filenames: filename_prefix: {identifier} full_filenames: {list_of_full_filenames}')  # noqa

    full_filenames = list_of_full_filenames[0]

    # label filename
    the_dirname_basename = os.path.basename(the_dirname)
    the_dirname_label_basename = re.sub(r'images', 'labels', the_dirname_basename)
    the_dirname_dirname = os.path.dirname(the_dirname)
    label_dirname = os.sep.join([the_dirname_dirname, the_dirname_label_basename])

    base_label_filename = f'{identifier}.{filename_postfix}'
    full_label_filename = os.sep.join([label_dirname, base_label_filename])
    if not os.path.exists(full_label_filename):
        full_label_filename = None

    cfg.logger.info(f'filename: {filename} identifier: {identifier}, full_filenames: {full_filenames} full_label_filename: {full_label_filename}')  # noqa

    return identifier, full_filenames, full_label_filename


def parse_args():
    parser = argparse.ArgumentParser(description='Use this to run inference with nnU-Net. This function is used when '  # noqa
                                                 'you want to manually specify a folder containing a trained nnU-Net '  # noqa
                                                 'model. This is useful when the nnunet environment variables '  # noqa
                                                 '(nnUNet_results) are not set.')
    argparse.add_args(parser, Args)

    return parser.parse_args()


def predict_entry_point():
    print(
        "\n#######################################################################\nPlease cite the following paper "  # noqa
        "when using nnU-Net:\n"
        "Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). "
        "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. "
        "Nature methods, 18(2), 203-211.\n#######################################################################\n")  # noqa

    args = parse_args()
    args.f = [i if i == 'all' else int(i) for i in args.f]

    model_folder = get_output_folder(args.d, args.tr, args.p, args.c)

    if not isdir(args.o):
        maybe_mkdir_p(args.o)

    assert args.device in ['cpu', 'cuda',
                           'mps'], f'-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {args.device}.'  # noqa
    if args.device == 'cpu':
        # let's allow torch to use hella threads
        import multiprocessing  # noqa
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device('cpu')
    elif args.device == 'cuda':
        # multithreading in torch doesn't help nnU-Net if run on GPU
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device('cuda')
    else:
        device = torch.device('mps')

    # init nn-extractor
    extra_params: nn_extractor.Config = {}
    if args.nn_extractor_name:
        extra_params['name'] = args.nn_extractor_name
    nn_extractor.init(args.nn_extractor_config_file, extra_params=extra_params)
    nnextractor_name = nn_extractor.config['name']

    predictor = nnUNetPredictor(tile_step_size=args.step_size,
                                use_gaussian=True,
                                use_mirroring=not args.disable_tta,
                                perform_everything_on_device=True,
                                device=device,
                                verbose=args.verbose,
                                verbose_preprocessing=args.verbose,
                                allow_tqdm=not args.disable_progress_bar)
    predictor.initialize_from_trained_model_folder(
        model_folder,
        args.f,
        checkpoint_name=args.chk
    )

    filename_prefix, filenames, label_filename = _determine_filenames(args.i)

    output_filename = os.sep.join([args.o, f'{filename_prefix}.nii.gz'])

    img, props = SimpleITKIO().read_images(filenames)

    label_img: Optional[np.ndarray] = None
    label_props: Optional[dict] = None
    if label_filename:
        label_img, label_props = SimpleITKIO().read_images([label_filename])
        label_img = label_img[0]

    _ = predictor.predict_single_npy_array(
        input_image=img,
        image_properties=props,
        segmentation_previous_stage=None,
        output_file_truncated=output_filename,
        save_or_return_probabilities=args.save_probabilities,

        nnextractor_name=f'{nnextractor_name}-{filename_prefix}',
        label_image=label_img,
        label_image_properties=label_props,
    )
