import decode.utils
import torch
import matplotlib.pyplot as plt
from gui import gui

device = 'cuda:0'  # or 'cpu'
device_ix = 0  # possibly change device index (only for cuda)
threads = 4  #  number of threads, useful for CPU heavy computation. Change if you know what you are doing.
worker = 4  # number of workers for data loading. Change only if you know what you are doing.
print("Open calibration file\n")
calib_file = gui()   # calibration file (from SMAP)

print("Open param_friendly file\n")
param = decode.utils.param_io.load_params(gui())  # change path if you load custom file

# Camera parameters
param.Camera.baseline = 500
param.Camera.e_per_adu = 3.0
param.Camera.em_gain = 20
param.Camera.px_size =[160.0, 160.0]  # Pixel Size in nano meter
param.Camera.qe = 0.95                # Quantum efficiency
param.Camera.read_sigma = 45
param.Camera.spur_noise = 0

param.Simulation.bg_uniform = [20.0, 200.0]           # background range to sample from. You can also specify a const. value as 'bg_uniform = 100'
param.Simulation.emitter_av = 50                      # Average number of emitters per frame
param.Simulation.emitter_extent[2] = [-400, 400]    # Volume in which emitters are sampled. x,y values should not be changed. z-range (in nm) should be adjusted according to the PSF
param.Simulation.intensity_mu_sig = [7000.0, 3000.0]  # Average intensity and its standard deviation
param.Simulation.lifetime_avg = 1.                     # Average lifetime of each emitter in frames. A value between 1 and 2 works for most experiments

param.InOut.calibration_file = calib_file
param.InOut.experiment_out = ''

param.Camera.to_dict()
param.Simulation.to_dict()
param.InOut.to_dict()
print("Open image\n")
frame_path = gui()  # change if you load your own data


def true_vs_simulated(camera, sim_frames, data_frames):
    
    print(f'Data shapes, simulation: {sim_frames.shape}, real data: {data_frames.shape}')
    print(f'Average value, simulation: {sim_frames.mean().round()}, real data: {data_frames.mean().round()}')

    data_frames = camera.backward(data_frames, device='cpu')
    print(f'Average value, simulation: {sim_frames.mean().round()}, real data: {data_frames.mean().round()}')

    plt.figure(figsize=(12,5))

    plt.subplot(121)
    decode.plot.PlotFrame(sim_frames[torch.randint(0, len(sim_frames), size=(1, ))]).plot()
    plt.colorbar()

    plt.subplot(122)
    decode.plot.PlotFrame(data_frames[torch.randint(0, len(data_frames), size=(1, )), 30:70,-40:]).plot()
    plt.colorbar()

    plt.show()

