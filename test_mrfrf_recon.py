import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import time
from pathlib import Path

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
import sigpy as sp
from einops import rearrange
from loguru import logger
from mpl_toolkits.axes_grid1 import make_axes_locatable
from optimized_mrf.utils import create_exp_dir

from mr_sim.data_sim import quant_phantom

device_idx = 0

def multi_subspace_linop(trj, phis, mps, b1_map, sqrt_dcf=None):
    masks = [np.where(b1_map == i, 1, 0) for i in range(int(b1_map.max()) + 1)]
    return sp.linop.Add(
        [
            single_subspace_linop(trj, phi, mps, sqrt_dcf, mask)
            for phi, mask in zip(phis, masks)
        ]
    )


def single_subspace_linop(trj, phi, mps, sqrt_dcf=None, mask=None):
    dev = sp.get_device(mps)
    assert sp.get_device(phi) == dev

    if type(sqrt_dcf) == type(None):
        F = sp.linop.NUFFT(mps.shape[1:], trj)
    else:
        assert sp.get_device(sqrt_dcf) == dev
        F = sp.linop.Multiply(trj.shape[:-1], sqrt_dcf) * sp.linop.NUFFT(
            mps.shape[1:], trj
        )

    if mask is None:
        mask = np.ones(mps.shape[1:])

    outer_A = []
    for k in range(mps.shape[0]):
        S = (
            sp.linop.Multiply(mps.shape[1:], mps[k, ...])
            * sp.linop.Multiply(mps.shape[1:], mask)
            * sp.linop.Reshape(mps.shape[1:], [1] + list(mps.shape[1:]))
        )
        lst_A = [
            sp.linop.Reshape([1] + list(F.oshape), F.oshape)
            * sp.linop.Multiply(F.oshape, phi[None, None, k, :])
            * F
            * S
            for k in range(phi.shape[0])
        ]
        inner_A = sp.linop.Hstack(lst_A, axis=0)
        outer_A.append(inner_A)
    A = sp.linop.Vstack(outer_A, axis=0)

    return A


def _dict_matching(signal, dct, tissues):
    norm_vals = np.linalg.norm(dct, ord=2, axis=-1, keepdims=True)
    norm_dict = dct / norm_vals

    corr = np.abs(signal @ norm_dict.squeeze().T)

    est_idx = np.argmax(corr, axis=-1)

    est_tissues = tissues[est_idx].squeeze()

    est_amp = corr[np.arange(corr.shape[0]), est_idx] / norm_vals[est_idx].squeeze()
    est_tissues[:, 0] = est_amp

    return est_tissues


def dict_matching(signal, dct, tissues, b1_map, brain_mask=None, avg_pd=None):
    est_tissues = sp.to_device(np.zeros(signal.shape[:2] + (3,)), signal.device)
    for i in range(int(b1_map.max()) + 1):
        mask = b1_map == i
        b1_sig = signal[mask]
        b1_dict = dct[i]
        est_tissues[mask] = _dict_matching(b1_sig, b1_dict, tissues[:, 0])
    if brain_mask is not None:
        # Normalize PD
        est_tissues[brain_mask == False, ...] = 0
        if avg_pd is not None:
            scale =  avg_pd / np.mean(est_tissues[brain_mask, 0])
            est_tissues[brain_mask, 0] *= scale
    return est_tissues


def est_error(gt_pd, gt_t1, gt_t2, est):
    mask = gt_pd > 1e-5
    pd_err = np.mean(np.abs(gt_pd[mask] - est[mask][:, 0]) / gt_pd[mask])
    t1_err = np.mean(np.abs(gt_t1[mask] - est[mask][:, 1]) / gt_t1[mask])
    t2_err = np.mean(np.abs(gt_t2[mask] - est[mask][:, 2]) / gt_t2[mask])

    return pd_err, t1_err, t2_err


def mrfrf_recon(
    ksp, trj, phis, mps, sqrt_dcf, dicts, tissues, b1_map, mask, n_iters=50
):
    A = multi_subspace_linop(trj, phis, mps, b1_map, sqrt_dcf)
    ksp = sqrt_dcf * ksp
    x = A.H * ksp
    recon = sp.app.LinearLeastSquares(A=A, y=ksp, x=x, max_iter=n_iters).run()
    est_tissues_recon = dict_matching(
        rearrange(recon, "a b c -> b c a"), dicts, tissues, b1_map, mask, avg_pd
    )

    return recon, est_tissues_recon


def plot_coeffs(image_list, titles, save_path):
    C = image_list[0].shape[0]
    num_columns = len(image_list)

    fig, axarr = plt.subplots(C, num_columns, figsize=(5 * num_columns, 5 * C))

    for c in range(C):
        # Determine the vmin and vmax for the current channel across all images in the list
        vmin = min([img[c].min() for img in image_list])
        vmax = max([img[c].max() for img in image_list])

        for col, img in enumerate(image_list):
            # Display the image
            im = axarr[c, col].imshow(img[c], cmap="gray", vmin=vmin, vmax=vmax)
            axarr[c, col].axis("off")

            if c == 0:
                axarr[c, col].set_title(titles[col])

            # If it's the last column, add a colorbar to the right
            if col == num_columns - 1:
                divider = make_axes_locatable(axarr[c, col])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def plot_recons(recon_list, ground_truth, titles, mask, save_path_prefix):
    # Ensure that the number of titles matches the number of images
    if len(recon_list) != len(titles):
        raise ValueError(
            "The number of titles must match the number of images in the list."
        )

    modalities = ["PD", "T1", "T2"]

    for index, modality in enumerate(modalities):
        fig, axarr = plt.subplots(2, len(recon_list), figsize=(5 * len(recon_list), 10))

        for col, img in enumerate(recon_list):
            original_image = img[:, :, index]
            gt_image = ground_truth[:, :, index]

            # Compute pixel-wise normalized loss
            normalized_loss = np.abs(original_image - gt_image) / (
                gt_image + 1e-8
            )  # added epsilon to avoid division by zero

            # Compute normalized RMSE
            avg_err = np.mean(normalized_loss[mask])

            # Plot original image
            im_orig = axarr[0, col].imshow(original_image, cmap="hot")
            axarr[0, col].axis("off")
            axarr[0, col].set_title(titles[col])

            # Plot normalized loss
            im_loss = axarr[1, col].imshow(normalized_loss, cmap="hot", vmin=0, vmax=1)
            axarr[1, col].axis("off")
            axarr[1, col].text(
                0.8 * original_image.shape[1],
                0.95 * original_image.shape[0],
                f"Avg.Err: {avg_err:.4f}",
                color="white",
                ha="center",
            )

            if col == len(recon_list) - 1:
                divider_orig = make_axes_locatable(axarr[0, col])
                cax_orig = divider_orig.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im_orig, cax=cax_orig)

                divider_loss = make_axes_locatable(axarr[1, col])
                cax_loss = divider_loss.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im_loss, cax=cax_loss)

        plt.savefig(
            f"{save_path_prefix}_{modality}.png", bbox_inches="tight", pad_inches=0.1
        )
        plt.close(fig)


sigma = 0.001
n_iters = 20

data_dirs = [
    Path("simulated_data/b1/im_size_torch.Size([220, 220])_coil_8_R_16.0_n_b1_vals_20_n_subspaces_6"),
    Path("simulated_data/b1/im_size_torch.Size([220, 220])_coil_8_R_16.0_n_b1_vals_20_n_subspaces_4"),
    Path("simulated_data/b1/im_size_torch.Size([220, 220])_coil_8_R_16.0_n_b1_vals_20_max_1.2772691799877638_min_0.41349801952278464"),
]
names = ["MRFRF-6", "MRFRF-4", "MRFRF-3"]

debug_im_undersampling = 1# 220 // 31
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

mask = gt_pd > 1e-5
avg_pd = np.mean(gt_pd[mask])
n_coils = 8

exp_name = f"b1_im_size_{gt_pd.shape}_noise_{sigma}_n_iters_{n_iters}_n_coils_{n_coils}"

log_dir = create_exp_dir(Path("./logs"), exp_name)
logger.add(log_dir / "log-{time}.log")

recon_coeffs = []
recon_maps = []

for i, (name, data_dir) in enumerate(zip(names, data_dirs)):
    logger.info(f"Loading {name} data from {data_dir}")
    keys = ["ksp", "trj", "dcf", "mps", "phi", "dct", "tissues", "b1_map", "imgs"]
    data_dict = {}
    for key in keys:
        data_dict[key] = np.load(data_dir / (key + ".npy"))
        logger.info(f"{key} shape = {data_dict[key].shape}")

    ksp = sp.to_device(data_dict["ksp"], device_idx)
    trj = sp.to_device(data_dict["trj"], device_idx)
    mps = sp.to_device(data_dict["mps"], device_idx)
    phis = sp.to_device(data_dict["phi"], device_idx)
    dcf = sp.to_device(data_dict["dcf"], device_idx)
    dcts = sp.to_device(data_dict["dct"], device_idx)
    imgs = sp.to_device(data_dict["imgs"], device_idx)
    b1_map = sp.to_device(data_dict["b1_map"], device_idx)
    tissues = sp.to_device(data_dict["tissues"], device_idx)

    # Add noise
    if i == 0 and sigma > 0:
        noise = sp.to_device(np.random.normal(0, sigma, ksp.shape) + 1j * np.random.normal(
            0, sigma, ksp.shape
        ), device_idx)

    if sigma > 0:
        ksp += noise

    cmprs_dicts = []
    for i in range(len(phis)):
        cmprs_dicts.append(dcts[:, i] @ phis[i].T)

    recon, est_tissues_recon = mrfrf_recon(
        ksp,
        trj,
        phis,
        mps,
        np.sqrt(dcf),
        cmprs_dicts,
        tissues,
        b1_map,
        mask,
        n_iters=n_iters,
    )

    recon_coeffs.append(np.abs(sp.to_device(recon, -1)))
    recon_maps.append(sp.to_device(est_tissues_recon, -1))

    pd_err, t1_err, t2_err = est_error(gt_pd, gt_t1, gt_t2, est_tissues_recon)
    logger.info(
        f"Average estimation error when using {name}: pd mape: {pd_err} t1 mape: {t1_err} t2 mape: {t2_err}"
    )

    # Plot B1:
    b1_map = sp.to_device(b1_map, -1)
    plt.imshow(b1_map)
    plt.colorbar()
    plt.savefig(log_dir / f"{name}_b1_map.png")
    plt.close()

# Plot subspace coeffs
plot_coeffs(recon_coeffs, names, log_dir / "coeffs_cmpr.png")


# Plot reconstructed maps and errors
gt = sp.to_device(np.concatenate((gt_pd[..., None], gt_t1[..., None], gt_t2[..., None]), axis=-1), -1)
plot_recons(recon_maps, gt, names, sp.to_device(mask, -1), log_dir / "recon_cmpr.png")