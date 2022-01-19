import decode

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from gui import gui

device = 'cuda:0'  # or 'cpu', or you change cuda device index
threads = 4  
worker = 0  

# here you need to specify the parameters with suffix _run.yaml in your model's output folder (not param_run_in.yaml)
print("Open parameters file\n")
param_path = gui()
print("Open model file\n")
model_path = gui()
print("Open image\n")
frame_path = gui()

# specify camera parameters of tiffs
meta = {
    'Camera': {
        'baseline': 500,
        'e_per_adu': 3.0,
        'em_gain': 20.0,
        'spur_noise': 0  # if you don't know, you can set this to 0.
    }
}


def generate_predicted_frames(param, emitter):
    plt.figure(figsize=(14, 4))

    plt.subplot(131)
    mu, sig = param.Simulation.intensity_mu_sig
    plt.axvspan(0, mu + sig * 3, color='green', alpha=0.1)
    sns.distplot(emitter.phot.numpy())
    plt.xlabel('Inferred number of photons')
    plt.xlim(0)

    plt.subplot(132)
    plt.axvspan(*param.Simulation.bg_uniform, color='green', alpha=0.1)
    sns.distplot(emitter.bg.numpy())
    plt.xlabel('Inferred background values')

    plt.show()


# compare the inferred distribution of the photon numbers and background values with the ranges used during training.
def compare_photon_bckg_values(param, emitter):
    plt.figure(figsize=(14, 4))

    plt.subplot(131)
    mu, sig = param.Simulation.intensity_mu_sig
    plt.axvspan(0, mu + sig * 3, color='green', alpha=0.1)
    sns.distplot(emitter.phot.numpy())
    plt.xlabel('Inferred number of photons')
    plt.xlim(0)

    plt.subplot(132)
    plt.axvspan(*param.Simulation.bg_uniform, color='green', alpha=0.1)
    sns.distplot(emitter.bg.numpy())
    plt.xlabel('Inferred background values')

    plt.show()


# plot uncertainties in x, y and z coordinates
def plot_coordinates_uncertainties(emitter):
    plt.figure(figsize=(18,4))
    plt.subplot(131)
    sns.distplot(emitter.xyz_sig_nm[:, 0].numpy())
    plt.xlabel('Sigma Estimate in X (nm)')

    plt.subplot(132)
    sns.distplot(emitter.xyz_sig_nm[:, 1].numpy())
    plt.xlabel('Sigma Estimate in Y (nm)')

    plt.subplot(133)
    sns.distplot(emitter.xyz_sig_nm[:, 2].numpy())
    plt.xlabel('Sigma Estimate in Z (nm)')

    plt.show()
    

# plot the x-y view with z as colordimension as well as the x-z view. 
# right side we instead color by the combined inferred uncertainty. 
def plot_xy_image_and_uncertainties(emitter):
    fig, axs = plt.subplots(2, 2, figsize=(24, 12), sharex='col',
                        gridspec_kw={'height_ratios': [1, 1200 / 20000]})

    decode.renderer.Renderer2D(px_size=10., sigma_blur=5., rel_clip=None, abs_clip=1,
                            zextent=[-600, 600], colextent=[-500, 500], plot_axis=(0, 1),
                            contrast=1.25).render(emitter, emitter.xyz_nm[:, 2], ax=axs[0, 0])
    decode.renderer.Renderer2D(px_size=10., sigma_blur=5., rel_clip=None, abs_clip=50,
                            zextent=[-600, 600], plot_axis=(0, 2)).render(emitter, ax=axs[1, 0])

    decode.renderer.Renderer2D(px_size=10., sigma_blur=5., rel_clip=None, abs_clip=1,
                            zextent=[-600, 600], colextent=[0, 75], plot_axis=(0, 1),
                            contrast=1.25).render(emitter, emitter.xyz_sig_weighted_tot_nm, ax=axs[0, 1])

    decode.renderer.Renderer2D(px_size=10., sigma_blur=5., rel_clip=None, abs_clip=50,
                            zextent=[-600, 600], colextent=[0, 75], plot_axis=(0, 2)).\
        render(emitter, emitter.xyz_sig_weighted_tot_nm, ax=axs[1, 1])

    plt.show()


# remove all localizations with uncertainties that exceed 40 nm in x,y or 80 nm in z
def filtering_by_sigmas(emitter, sigmaX, sigmaY, sigmaZ):
    sigma_x_high_threshold = sigmaX
    sigma_y_high_threshold = sigmaY
    sigma_z_high_threshold = sigmaZ

    em_sub = emitter[
        (emitter.xyz_sig_nm[:, 0] <= sigma_x_high_threshold)
        * (emitter.xyz_sig_nm[:, 1] <= sigma_y_high_threshold)
        * (emitter.xyz_sig_nm[:, 2] <= sigma_z_high_threshold)
        ]
    # em_sub = emitter.filter_by_sigma(0.67)  # alternatively

    plt.figure(figsize=(12, 12))
    decode.renderer.Renderer2D(px_size=10., sigma_blur=5., rel_clip=None, abs_clip=1,
                            zextent=[-600, 600], colextent=[-500, 500], plot_axis=(0, 1),
                            contrast=1.5).render(em_sub, em_sub.xyz_nm[:, 2])
    plt.title(
        f'Filtered Image {np.round(100 * len(em_sub) / len(emitter))} % of em_subs remaining', loc='right')
    plt.show()
    plt.figure(figsize=(12, 3))
    decode.renderer.Renderer2D(px_size=10., sigma_blur=5., rel_clip=None, abs_clip=50,
                            zextent=[-600, 600], plot_axis=(0, 2)).render(em_sub)

    plt.show()
    
    
def show_image(emitter):
    fig, axs = plt.subplots(2, 2, figsize=(24, 12), sharex='col',
                            gridspec_kw={'height_ratios': [1, 1200 / 7000]})
    extents = {
        'xextent': [14000, 22000],
        'yextent': [1000, 8000],
        'zextent': [-600, 600],
        'colextent': [-500, 500]
    }

    decode.renderer.Renderer2D(
        px_size=5., sigma_blur=5., rel_clip=None, abs_clip=3, **extents,
        plot_axis=(0, 1), contrast=3).render(emitter, emitter.xyz_nm[:, 2], ax=axs[0, 0])

    decode.renderer.Renderer2D(
        px_size=5., sigma_blur=5., rel_clip=None, abs_clip=15, **extents,
        plot_axis=(0, 2), contrast=2).render(emitter, ax=axs[1, 0])

    decode.renderer.RendererIndividual2D(
        px_size=5., filt_size=20, rel_clip=None, abs_clip=3, **extents,
        plot_axis=(0, 1), contrast=3).render(emitter, emitter.xyz_nm[:, 2], ax=axs[0, 1])

    decode.renderer.RendererIndividual2D(
        px_size=5., filt_size=20, rel_clip=None, abs_clip=3, **extents,
        plot_axis=(0, 2)).render(emitter, ax=axs[1, 1])

    axs[0, 0].set_title('Rendering with constant sigma blur 5 nm', fontsize=20)
    axs[0, 1].set_title('Rendering with individual sigmas', fontsize=20)

    plt.show()