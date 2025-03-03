from queue import Queue
from threading import Thread
from typing import Optional, Union

from nn_extractor import nii
from nn_extractor.nnextractor import NNExtractor
from nn_extractor.ops.crop import Crop
from nn_extractor.ops.pad import Pad
import numpy as np
import torch
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, isdir
from torch._dynamo import OptimizedModule
from tqdm import tqdm

from nnunetv2.configuration import default_num_processes
from nnunetv2.inference.data_iterators import PreprocessAdapterFromNpy
from nnunetv2.inference.export_prediction import \
    convert_predicted_logits_to_segmentation_with_correct_shape
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.utilities.file_path_utilities import get_output_folder
from nnunetv2.utilities.helpers import empty_cache, dummy_context

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
            name=nnextractor_name,
            data={'img': Crop(img=img, region=crop_region), 'props': properties_dict},
        )

        if self.verbose:
            print('predicting')
        predicted_logits = self.predict_logits_from_preprocessed_data(dct['data']).cpu()

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
                        data={'workon': workon, 'slicer': list(sl)},
                    )

                    prediction = self._internal_maybe_mirror_and_predict(workon)[0].to(results_device)  # noqa

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
                            'predicted_logits': predicted_logits,
                            'n_predictions': n_predictions,
                            'prediction': prediction,
                            'gaussian': gaussian,
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


def predict_entry_point():
    import argparse
    parser = argparse.ArgumentParser(description='Use this to run inference with nnU-Net. This function is used when '  # noqa
                                                 'you want to manually specify a folder containing a trained nnU-Net '  # noqa
                                                 'model. This is useful when the nnunet environment variables '  # noqa
                                                 '(nnUNet_results) are not set.')
    parser.add_argument('-i', type=str, required=True,
                        help='input folder. Remember to use the correct channel numberings for your files (_0000 etc). '  # noqa
                             'File endings must be the same as the training dataset!')
    parser.add_argument('-o', type=str, required=True,
                        help='Output folder. If it does not exist it will be created. Predicted segmentations will '  # noqa
                             'have the same name as their source images.')
    parser.add_argument('-d', type=str, required=True,
                        help='Dataset with which you would like to predict. You can specify either dataset name or id')  # noqa
    parser.add_argument('-p', type=str, required=False, default='nnUNetPlans',
                        help='Plans identifier. Specify the plans in which the desired configuration is located. '  # noqa
                             'Default: nnUNetPlans')
    parser.add_argument('-tr', type=str, required=False, default='nnUNetTrainer',
                        help='What nnU-Net trainer class was used for training? Default: nnUNetTrainer')  # noqa
    parser.add_argument('-c', type=str, required=True,
                        help='nnU-Net configuration that should be used for prediction. Config must be located '  # noqa
                             'in the plans specified with -p')
    parser.add_argument('-f', nargs='+', type=str, required=False, default=(0, 1, 2, 3, 4),
                        help='Specify the folds of the trained model that should be used for prediction. '  # noqa
                             'Default: (0, 1, 2, 3, 4)')
    parser.add_argument('-step_size', type=float, required=False, default=0.5,
                        help='Step size for sliding window prediction. The larger it is the faster but less accurate '  # noqa
                             'the prediction. Default: 0.5. Cannot be larger than 1. We recommend the default.')  # noqa
    parser.add_argument('--disable_tta', action='store_true', required=False, default=False,
                        help='Set this flag to disable test time data augmentation in the form of mirroring. Faster, '  # noqa
                             'but less accurate inference. Not recommended.')
    parser.add_argument('--verbose', action='store_true', help="Set this if you like being talked to. You will have "  # noqa
                                                               "to be a good listener/reader.")
    parser.add_argument('--save_probabilities', action='store_true',
                        help='Set this to export predicted class "probabilities". Required if you want to ensemble '  # noqa
                             'multiple configurations.')
    parser.add_argument('--continue_prediction', action='store_true',
                        help='Continue an aborted previous prediction (will not overwrite existing files)')  # noqa
    parser.add_argument('-chk', type=str, required=False, default='checkpoint_final.pth',
                        help='Name of the checkpoint you want to use. Default: checkpoint_final.pth')  # noqa
    parser.add_argument('-npp', type=int, required=False, default=3,
                        help='Number of processes used for preprocessing. More is not always better. Beware of '  # noqa
                             'out-of-RAM issues. Default: 3')
    parser.add_argument('-nps', type=int, required=False, default=3,
                        help='Number of processes used for segmentation export. More is not always better. Beware of '  # noqa
                             'out-of-RAM issues. Default: 3')
    parser.add_argument('-prev_stage_predictions', type=str, required=False, default=None,
                        help='Folder containing the predictions of the previous stage. Required for cascaded models.')  # noqa
    parser.add_argument('-num_parts', type=int, required=False, default=1,
                        help='Number of separate nnUNetv2_predict call that you will be making. Default: 1 (= this one '  # noqa
                             'call predicts everything)')
    parser.add_argument('-part_id', type=int, required=False, default=0,
                        help='If multiple nnUNetv2_predict exist, which one is this? IDs start with 0 can end with '  # noqa
                             'num_parts - 1. So when you submit 5 nnUNetv2_predict calls you need to set -num_parts '  # noqa
                             '5 and use -part_id 0, 1, 2, 3 and 4. Simple, right? Note: You are yourself responsible '  # noqa
                             'to make these run on separate GPUs! Use CUDA_VISIBLE_DEVICES (google, yo!)')  # noqa
    parser.add_argument('-device', type=str, default='cuda', required=False,
                        help="Use this to set the device the inference should run with. Available options are 'cuda' "  # noqa
                             "(GPU), 'cpu' (CPU) and 'mps' (Apple M1/M2). Do NOT use this to set which GPU ID! "  # noqa
                             "Use CUDA_VISIBLE_DEVICES=X nnUNetv2_predict [...] instead!")
    parser.add_argument('--disable_progress_bar', action='store_true', required=False, default=False,  # noqa
                        help='Set this flag to disable progress bar. Recommended for HPC environments (non interactive '  # noqa
                             'jobs)')

    print(
        "\n#######################################################################\nPlease cite the following paper "  # noqa
        "when using nnU-Net:\n"
        "Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). "
        "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. "
        "Nature methods, 18(2), 203-211.\n#######################################################################\n")  # noqa

    args = parser.parse_args()
    args.f = [i if i == 'all' else int(i) for i in args.f]

    model_folder = get_output_folder(args.d, args.tr, args.p, args.c)

    if not isdir(args.o):
        maybe_mkdir_p(args.o)

    # slightly passive aggressive haha
    assert args.part_id < args.num_parts, 'Do you even read the documentation? See nnUNetv2_predict -h.'  # noqa

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
    predictor.predict_from_files(args.i, args.o, save_probabilities=args.save_probabilities,
                                 overwrite=not args.continue_prediction,
                                 num_processes_preprocessing=args.npp,
                                 num_processes_segmentation_export=args.nps,
                                 folder_with_segs_from_prev_stage=args.prev_stage_predictions,
                                 num_parts=args.num_parts,
                                 part_id=args.part_id)
    # r = predict_from_raw_data(args.i,
    #                           args.o,
    #                           model_folder,
    #                           args.f,
    #                           args.step_size,
    #                           use_gaussian=True,
    #                           use_mirroring=not args.disable_tta,
    #                           perform_everything_on_device=True,
    #                           verbose=args.verbose,
    #                           save_probabilities=args.save_probabilities,
    #                           overwrite=not args.continue_prediction,
    #                           checkpoint_name=args.chk,
    #                           num_processes_preprocessing=args.npp,
    #                           num_processes_segmentation_export=args.nps,
    #                           folder_with_segs_from_prev_stage=args.prev_stage_predictions,
    #                           num_parts=args.num_parts,
    #                           part_id=args.part_id,
    #                           device=device)


if __name__ == '__main__':
    ########################## predict a bunch of files  # noqa
    from nnunetv2.paths import nnUNet_results, nnUNet_raw  # noqa

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device('cuda', 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    predictor.initialize_from_trained_model_folder(
        join(nnUNet_results, 'Dataset004_Hippocampus/nnUNetTrainer_5epochs__nnUNetPlans__3d_fullres'),  # noqa
        use_folds=(0,),
        checkpoint_name='checkpoint_final.pth',
    )
    # predictor.predict_from_files(join(nnUNet_raw, 'Dataset003_Liver/imagesTs'),
    #                              join(nnUNet_raw, 'Dataset003_Liver/imagesTs_predlowres'),
    #                              save_probabilities=False, overwrite=False,
    #                              num_processes_preprocessing=2, num_processes_segmentation_export=2,  # noqa
    #                              folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)
    #
    # # predict a numpy array
    # from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
    #
    # img, props = SimpleITKIO().read_images([join(nnUNet_raw, 'Dataset003_Liver/imagesTr/liver_63_0000.nii.gz')])  # noqa
    # ret = predictor.predict_single_npy_array(img, props, None, None, False)
    #
    # iterator = predictor.get_data_iterator_from_raw_npy_data([img], None, [props], None, 1)
    # ret = predictor.predict_from_data_iterator(iterator, False, 1)

    ret = predictor.predict_from_files_sequential(
        [['/media/isensee/raw_data/nnUNet_raw/Dataset004_Hippocampus/imagesTs/hippocampus_002_0000.nii.gz'], ['/media/isensee/raw_data/nnUNet_raw/Dataset004_Hippocampus/imagesTs/hippocampus_005_0000.nii.gz']],  # noqa
        '/home/isensee/temp/tmp', False, True, None
    )


# noqa