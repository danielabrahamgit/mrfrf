import os
import gc
import torch
import numpy as np
import sigpy as sp
import sigpy.mri as mr
import torch.nn as nn
from einops import rearrange
from tqdm import tqdm

from igrog.recon.reconstructor import reconstructor
from torchkbnufft import KbNufft
from einops import rearrange
from optimized_mrf.sequences import FISP

import matplotlib
matplotlib.use('webagg')
import matplotlib.pyplot as plt

def dict_matching(signal, dct, tissues):
    norm_vals = np.linalg.norm(dct, ord=2, axis=-1, keepdims=True)
    norm_dict = dct / norm_vals
    corr = signal @ norm_dict.squeeze().T
    est_tissues_idx = np.argmax(corr, axis=-1)
    est_amp = (
        corr[np.arange(corr.shape[0]), est_tissues_idx]
        / norm_vals[est_tissues_idx].squeeze()
    )
    est_tissues = tissues[est_tissues_idx].squeeze()
    est_tissues[..., 0] = est_amp
    return est_tissues

# Params
recon_types = [
    # 'subspace',
    'nonlinear'
    ]
figsize = (18, 10)
data_dir = f'./simulated_data/'
plot = True
figsize = (18, 10)

# Recon params
sigma = 1e-5 * 0
fast_AHA_os = 2.0
max_iter = 30
max_eigen = None 
device_idx = 3

# GPU stuff
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
torch_dev = torch.device(device_idx)

if len(recon_types) > 0:

    # Load from simulated data
    keys = ['ksp', 'trj', 'dcf', 'mps', 'phi', 'dct', 'tissues']
    data_dict = {}
    for key in keys:
        data_dict[key] = np.load(data_dir + key + '.npy')
        print(f'{key} shape = {data_dict[key].shape}')
    
    # data
    ksp = torch.from_numpy(data_dict['ksp']).to(device_idx)
    trj = data_dict['trj']
    mps = data_dict['mps']
    phi = data_dict['phi']
    dcf = data_dict['dcf']
    dct = data_dict['dct']
    tissues = data_dict['tissues']
    seq = FISP.load(data_dir + 'seq').to(device_idx)
    
    # Add noise
    ksp += sigma * (torch.randn_like(ksp) + 1j * torch.randn_like(ksp))

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
            
            # Make linop
            A = rcnstr.build_linop(fast_AHA_os=fast_AHA_os)
            A.max_eigen = max_eigen

            # Run subspace recon
            coeffs = rcnstr.run_recon(A_linop=A,
                                    ksp=ksp,
                                    mps=mps,
                                    proxg=None,
                                    max_iter=max_iter,)
            # ceoffs = [nsub, Nx, Ny]
            compressed_dct = dct @ phi.T
            est_tissues = dict_matching(rearrange(coeffs, 'a b c -> (b c) a'), compressed_dct, tissues)

        # Non linear recon
        elif 'nonlinear' in recon_type:
            
            # Make torch linop
            mps_torch = torch.tensor(mps, device=torch_dev, dtype=torch.complex64)
            omega = torch.tensor(trj, device=torch_dev)
            omega = np.pi * omega / omega.max()
            omega_rs = rearrange(omega, 'nro npe ntr d -> ntr d (nro npe)')
            tkb_obj = KbNufft(mps.shape[1:], device=torch_dev)
            def A(x, time_batch_size=5):
                """
                x - (T, nx, ny)
                """
                T, _, K = omega_rs.shape
                nc = mps_torch.shape[0]
                ksp = torch.zeros((T, nc, K), dtype=torch.complex64, device=torch_dev)
                for t1 in range(0, T, time_batch_size):
                    t2 = min(t1 + time_batch_size, T)
                    ksp[t1:t2] = tkb_obj(x[t1:t2, None, ...], omega_rs[t1:t2], smaps=mps_torch[None, ...])
                return rearrange(ksp, 'ntr nc (nro npe) -> nc nro npe ntr',
                                 nro=trj.shape[0], npe=trj.shape[1])
            
            # Randomly intialize param maps
            im_size = mps.shape[1:]
            init_params = torch.ones((*im_size, 1, 3)).type(torch.float32)
            init_params[..., 0, 0] = torch.rand(im_size)
            init_params[..., 0, 1] = 100 + 1500 * torch.rand(im_size)
            init_params[..., 0, 2] = 10 + 1000 * torch.rand(im_size)
            params = init_params.to(device_idx).detach().clone()
            params.requires_grad = True
            # params = torch.rand((*im_size, 1, 3), requires_grad=True, device=next(seq.parameters()).device)

            # Gradient descent
            nsteps = 1000
            optim = torch.optim.SGD([params], lr=10)
            loss_func = torch.nn.MSELoss()
            losses = []
            for i in tqdm(range(nsteps)):
                ims = rearrange(seq(params).type(torch.complex64).squeeze(), 'h w t -> t h w')
                ksp_est = A(ims)
                loss = loss_func(ksp.real, ksp_est.real) + loss_func(ksp.imag, ksp_est.imag)
                loss.backward()
                optim.step()
                optim.zero_grad()
                losses.append(loss)
                print(loss)

            plt.imshow(params[:, :, 0, 0].detach().cpu())
            plt.colorbar()
            plt.savefig('amps.png')
            plt.close()
            plt.imshow(params[:, :, 0, 1].detach().cpu())
            plt.colorbar()
            plt.savefig('t1.png')
            plt.close()
            plt.imshow(params[:, :, 0, 2].detach().cpu())
            plt.colorbar()
            plt.savefig('t2.png')
            plt.close()
            plt.plot(loss)
            plt.savefig('loss.png')
            plt.close()
            quit()
            # Make it work!

    # Save
    np.save(f'{data_dir}/recons/{recon_type}.npy', coeffs)
    print(f'saved {recon_type}')

    # Clear memory for next run
    gc.collect()
    with torch.cuda.device(device_idx):
        torch.cuda.empty_cache()

if plot:
    
    # Magnitude
    transform = lambda x : np.abs(x)

    # Load data
    recon_type = 'subspace'
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