import decode
import decode.utils
import decode.neuralfitter.train.live_engine

import torch
from parameters.training_parameters import *


torch.set_num_threads(threads)  # set num threads
param.Hardware.device = device
param.Hardware.device_ix = device_ix
param.Hardware.device_simulation = device
param.Hardware.torch_threads = threads
param.Hardware.num_worker_train = worker

simulator, sim_test = decode.neuralfitter.train.live_engine.setup_random_simulation(param)
camera = decode.simulation.camera.Photon2Camera.parse(param)

# finally we derive some parameters automatically for easy use
param = decode.utils.param_io.autoset_scaling(param)

tar_em, sim_frames, bg_frames = simulator.sample()
sim_frames = sim_frames.cpu()

data_frames = decode.utils.frames_io.load_tif(frame_path).cpu()
true_vs_simulated(camera, sim_frames, data_frames)

if device != 'cpu':
    mem_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
    print(f"Your approximate total GPU memory size on the set device {device} is {mem_gb:.2f} GB.")

param.HyperParameter.batch_size = 16

param_out_path = 'param_friendly.yaml' # or an alternative path
decode.utils.param_io.save_params(param_out_path, param)
