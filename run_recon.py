import gc
import torch
import numpy as np
import sigpy as sp
import sigpy.mri as mr

import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt

from igrog.recon.reconstructor import reconstructor

# Params
sigma = 1e-5 * 0
plot = True
recon_types = [
    'subspace',
    'nonlinear']
figsize = (18, 10)
data_dir = f'./simulated_data/'

# Recon params
fast_AHA_os = 2.0
max_iter = 30
max_eigen = None 
device_idx = 6

if len(recon_types) > 0:
    # Load from simulated data
    keys = ['ksp', 'trj', 'dcf', 'mps', 'phi']
    data_dict = {}
    for key in keys:
        data_dict[key] = np.load(data_dir + key + '.npy')
        print(f'{key} shape = {data_dict[key].shape}')
    
    # data
    ksp = data_dict['ksp']
    trj = data_dict['trj']
    mps = data_dict['mps']
    phi = data_dict['phi']
    dcf = data_dict['dcf']
    
    # Add noise
    ksp += np.random.normal(0, sigma, ksp.shape) + 1j * np.random.normal(0, sigma, ksp.shape)

    for recon_type in recon_types:

        # Subspace recon
        if 'sub' in recon_type:
            # Make recon object
            rcnstr = reconstructor(im_size=mps.shape[1:],
                                trj=trj,
                                dcf=dcf,
                                phi=phi,
                                device_idx=device_idx,
                                verbose=True)
            
            
            A = rcnstr.build_linop(fast_AHA_os=fast_AHA_os)
            A.max_eigen = max_eigen
            coeffs = rcnstr.run_recon(A_linop=A,
                                    ksp=ksp,
                                    mps=mps,
                                    proxg=None,
                                    max_iter=max_iter,)
        # Non linear recon
        elif 'nonlinear' in recon_type:
            
            # Make linop 
            pass # TODO

    # Save
    np.save(f'{data_dir}/recons/{recon_type}.npy', coeffs)

    # Clear memory for next run
    gc.collect()
    with torch.cuda.device(device_idx):
        torch.cuda.empty_cache()

if plot:
    
    # Magnitude
    transform = lambda x : np.abs(x)


    # Load data
    coeffs = np.load(f'{data_dir}/recons/{recon_type}.npy')
    n_subspace = coeffs.shape[0]
    d = len(coeffs.shape[1:])

    # make figure 
    fig, axs = plt.subplots(nrows=1, ncols=n_subspace, figsize=figsize)
    if n_subspace == 1:
        axs = np.expand_dims(axs, axis=0)

    # Plot all subspace coeffs
    for i in range(n_subspace):

        # Regular coefficient
        img = transform(coeffs[i, :, :])
        vmax_reg = np.median(img) + 3 * np.std(img, dtype=np.float128)
        if i == 0:
            ref = vmax_reg
        im = axs[i].imshow(img / vmax_reg, vmin=0, vmax=1, cmap='gray')
        axs[i].set_title(r'$\alpha_{{{}}}$'.format(i+1) + f'({int(np.round(ref / vmax_reg))}X)')
        axs[i].axis('off')
    plt.tight_layout()
    plt.show()