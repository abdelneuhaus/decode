import decode.utils
import torch
from parameters.fitting_parameters import *


torch.set_num_threads(threads)  # set num threads
param = decode.utils.param_io.load_params(param_path)
model = decode.neuralfitter.models.SigmaMUNet.parse(param)
model = decode.utils.model_io.LoadSaveModel(model,
                                            input_file=model_path,
                                            output_file=None).load_init(device=device)

# overwrite camera
param = decode.utils.param_io.autofill_dict(meta['Camera'], param.to_dict(), mode_missing='include')
param = decode.utils.param_io.RecursiveNamespace(**param)

# depends on your input, e.g. load a tiff
frames = decode.utils.frames_io.load_tif(frame_path)

camera = decode.simulation.camera.Photon2Camera.parse(param)
camera.device = 'cpu'


# setup frame processing as by the parameter with which the model was trained
frame_proc = decode.neuralfitter.utils.processing.TransformSequence([
    decode.neuralfitter.utils.processing.wrap_callable(camera.backward),
    decode.neuralfitter.frame_processing.AutoCenterCrop(8),
    decode.neuralfitter.frame_processing.Mirror2D(dims=-1),  # WARNING: You might need to comment this line out. see above
    decode.neuralfitter.scale_transform.AmplitudeRescale.parse(param)
])

# determine extent of frame and its dimension after frame_processing
size_procced = decode.neuralfitter.frame_processing.get_frame_extent(frames.unsqueeze(1).size(), frame_proc.forward)  # frame size after processing
frame_extent = ((-0.5, size_procced[-2] - 0.5), (-0.5, size_procced[-1] - 0.5))


# Setup post-processing
# It's a sequence of backscaling, relative to abs. coord conversion and frame2emitter conversion
post_proc = decode.neuralfitter.utils.processing.TransformSequence([

    decode.neuralfitter.scale_transform.InverseParamListRescale.parse(param),

    decode.neuralfitter.coord_transform.Offset2Coordinate(xextent=frame_extent[0],
                                                          yextent=frame_extent[1],
                                                          img_shape=size_procced[-2:]),

    decode.neuralfitter.post_processing.SpatialIntegration(raw_th=0.1,
                                                          xy_unit='px',
                                                          px_size=param.Camera.px_size)


])

infer = decode.neuralfitter.Infer(model=model, ch_in=param.HyperParameter.channels_in,
                                  frame_proc=frame_proc, post_proc=post_proc,
                                  device=device, num_workers=worker)

emitter = infer.forward(frames[:])

# check on the output
print(emitter)


######################
random_ix = torch.randint(frames.size(0), size=(1, )).item()
em_subset = emitter.get_subset_frame(random_ix, random_ix)

# Check if predictions look reasonables on predictions frames (compare the frame and localization)
generate_predicted_frames(param, emitter)

# compare the inferred distribution of the photon numbers and background values with the ranges used during training.
compare_photon_bckg_values(param, emitter)

# plot uncertainties in x, y and z coordinates
plot_coordinates_uncertainties(emitter)

# plot the x-y view with z as colordimension as well as the x-z view. 
# right side we instead color by the combined inferred uncertainty. 
plot_xy_image_and_uncertainties(emitter)

# remove all localizations with uncertainties that exceed 40 nm in x,y or 80 nm in z
filtering_by_sigmas(emitter, sigmaX=40, sigmaY=40, sigmaZ=80)

# plot all of them and account for their uncertainty by rendering every localization as 
# a Gaussian with a two dimensional standard deviation equal to the predicted uncertainty
show_image(emitter)


# save emitter
emitter.save('emitter.h5')  # or '.csv' or '.pt'