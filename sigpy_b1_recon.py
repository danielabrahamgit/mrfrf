import sigpy as sp
import numpy as np
from optimized_mrf.sequences import FISP
from mr_sim.data_sim import data_sim, quant_phantom, mrf_sequence
import torch
import time
import cupy as cp
import matplotlib.pyplot as plt
from einops import rearrange

def _dict_matching(signal, dct, tissues):
    norm_vals = np.linalg.norm(dct, ord=2, axis=-1, keepdims=True)
    norm_dict = (dct / norm_vals)

    corr = np.abs(signal @ norm_dict.squeeze().T)

    est_idx = np.argmax(corr, axis=-1)

    est_tissues = tissues[est_idx].squeeze()

    est_amp = (
        corr[np.arange(corr.shape[0]), est_idx]
        / norm_vals[est_idx].squeeze()
    )
    est_tissues[:, 0] = est_amp

    return est_tissues

def dict_matching(signal, dct, tissues, b1_map):
    est_tissues = sp.to_device(np.zeros(signal.shape[:2]+(3,)), signal.device)
    for i in range(int(b1_map.max()) + 1):
        mask = b1_map == i
        b1_sig = signal[mask]
        b1_dict = dct[i]
        est_tissues[mask] = _dict_matching(b1_sig, b1_dict, tissues[:, 0])
    return est_tissues

def est_error(gt_pd, gt_t1, gt_t2, est):
    mask = gt_pd > 1e-5
    pd_err = np.mean(np.abs(gt_pd[mask] - est[mask][:, 0]) / gt_pd[mask]) 
    t1_err = np.mean(np.abs(gt_t1[mask] - est[mask][:, 1]) / gt_t1[mask])
    t2_err = np.mean(np.abs(gt_t2[mask] - est[mask][:, 2]) / gt_t2[mask])

    return pd_err, t1_err, t2_err

# Load data
device_idx = 0

data_dir = f"./simulated_data/b1_data/"
keys = ["ksp", "trj", "dcf", "mps", "phi", "dct", "tissues", "b1_map", "imgs"]
data_dict = {}
for key in keys:
    data_dict[key] = np.load(data_dir + key + ".npy")
    print(f"{key} shape = {data_dict[key].shape}")

# data
ksp = sp.to_device(data_dict["ksp"], device_idx)
trj = sp.to_device(data_dict["trj"], device_idx)
mps = sp.to_device(data_dict["mps"], device_idx)
phi = sp.to_device(data_dict["phi"], device_idx)
dcf = sp.to_device(data_dict["dcf"], device_idx)
dct = sp.to_device(data_dict["dct"], device_idx)
imgs = sp.to_device(data_dict["imgs"], device_idx)
b1_map = sp.to_device(data_dict["b1_map"], device_idx)
tissues = sp.to_device(data_dict["tissues"], device_idx)
seq = FISP.load(data_dir + "seq").to(device_idx)

debug_im_undersampling = 220 // 32
debug_seq_undersampling = 500 // 100
data = quant_phantom()
proc_data = {}
keys = ["t1", "t2", "pd"]
for key in keys:
    proc_data[key] = sp.resize(data[key].astype(np.float32)[100], (220, 220))
    proc_data[key] = proc_data[key]
gt_t1, gt_t2, gt_pd = (
    sp.to_device(proc_data["t1"], device_idx),
    sp.to_device(proc_data["t2"], device_idx),
    sp.to_device(proc_data["pd"], device_idx),
)
gt_t1 = gt_t1[::debug_im_undersampling, ::debug_im_undersampling]
gt_t2 = gt_t2[::debug_im_undersampling, ::debug_im_undersampling]
gt_pd = gt_pd[::debug_im_undersampling, ::debug_im_undersampling]


# define sigpy forward model
# masks for different b1 types
# forward 
with sp.Device(device_idx):
    def multi_subspace_linop(trj, phis, mps, sqrt_dcf=None):
        masks = [np.where(b1_map == i, 1, 0) for i in range(3)]
        return sp.linop.Add([single_subspace_linop(trj, phi, mps, sqrt_dcf, mask) for phi, mask in zip(phis, masks)])

    def single_subspace_linop(trj, phi, mps, sqrt_dcf=None, mask=None):
        dev = sp.get_device(mps)
        assert sp.get_device(phi) == dev

        if type(sqrt_dcf) == type(None):
            F = sp.linop.NUFFT(mps.shape[1:], trj)
        else:
            assert sp.get_device(sqrt_dcf) == dev
            F = sp.linop.Multiply(trj.shape[:-1], sqrt_dcf) * \
                sp.linop.NUFFT(mps.shape[1:], trj)

        if mask is None:
            mask = np.ones(mps.shape[1:])

        outer_A = []
        for k in range(mps.shape[0]):
            S = sp.linop.Multiply(mps.shape[1:], mps[k, ...]) * \
                sp.linop.Multiply(mps.shape[1:], mask) * \
                sp.linop.Reshape( mps.shape[1:], [1] + list(mps.shape[1:]))
            lst_A = [sp.linop.Reshape([1] + list(F.oshape), F.oshape)   * \
                    sp.linop.Multiply(F.oshape, phi[None, None, k, :]) * \
                    F * S for k in range(phi.shape[0])]
            inner_A = sp.linop.Hstack(lst_A, axis=0)
            outer_A.append(inner_A) 
        A = sp.linop.Vstack(outer_A, axis=0)

        return A


    sub_space = sp.to_device(np.zeros((37, 37, 11)), device_idx)
    for i in range(len(phi)):
        mask = np.where(b1_map == i, 1, 0)
        sub_space += (imgs @ phi[i].T) * mask[..., None]

    cmprs_dicts = []
    for i in range(len(phi)):
        cmprs_dicts.append(dct[:, i] @ phi[i].T)

    est_tissues = dict_matching(sub_space, cmprs_dicts, tissues, b1_map)
    pd_err, t1_err, t2_err = est_error(gt_pd, gt_t1, gt_t2, est_tissues)
    print(
        f"pd mape: {pd_err} t1 mape: {t1_err} t2 mape: {t2_err}"
    )

    # recon
    A = multi_subspace_linop(trj, phi, mps, np.sqrt(dcf))
    print(f'Calculating initial solution')
    tic = time.time()
    ksp = np.sqrt(dcf) * ksp
    x = A.H * ksp
    toc = time.time()
    print(f'A.H * x took {toc - tic}')

    recon = sp.app.LinearLeastSquares(A=A, y=ksp, x=x, max_iter=100).run()

    est_tissues = dict_matching(rearrange(recon, "a b c -> b c a"), cmprs_dicts, tissues, b1_map)
    pd_err, t1_err, t2_err = est_error(gt_pd, gt_t1, gt_t2, est_tissues)
    print(
        f"pd mape: {pd_err} t1 mape: {t1_err} t2 mape: {t2_err}"
    )
    

    recon = sp.to_device(recon, -1)
    for i, sub_s in enumerate(recon):
        plt.imshow(np.abs(sub_s), cmap='gray')
        plt.colorbar()
        plt.savefig(f'subspace_{i}')
        plt.close()

    # cg = sp.alg.ConjugateGradient(A=A, b=ksp, x=x, P=None, max_iter=100, tol=0)
    # while not cg.done():
    #     cg.update()
    est_tissues = sp.to_device(est_tissues, -1)
    plt.imshow(est_tissues[:, :, 0])
    plt.colorbar()
    plt.savefig('pds.png')
    plt.close()

    plt.imshow(est_tissues[:, :, 1])
    plt.colorbar()
    plt.savefig('t1.png')
    plt.close()

    plt.imshow(est_tissues[:, :, 2])
    plt.colorbar()
    plt.savefig('t2.png')
    plt.close()

