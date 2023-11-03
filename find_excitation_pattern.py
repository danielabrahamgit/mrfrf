import numpy as np
from tqdm import tqdm
from einops import rearrange
from optimized_mrf.sequences import FISP
import sigpy.mri as mr
import matplotlib

matplotlib.use("webagg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import sigpy as sp
from scipy.special import j1

from mr_sim.data_sim import simple_b1_map, quant_phantom, get_b1_map
import noise


def find_excitation_pattern_and_clusters(fa1, fa2, seq_map, smps, n_clusters, mask):
    assert len(fa1) == len(fa2), "fa are required to have the same length"
    c, h, w = smps.shape
    smps = np.abs(smps[:, mask])
    smps = rearrange(smps, "c k -> k c")
    seq_map = seq_map[mask]
    # organize maps in columns
    # smps = rearrange(smps, "c h w -> (h w) c")

    base_fas = []
    coil_amps = []

    actual_maps = []
    for i in tqdm(range(len(fa1))):
        wnated_fa = np.where(seq_map == 0, fa1[i], fa2[i])
        # wnated_fa = rearrange(wnated_fa_2d, "h w -> (h w)")
        amps = np.linalg.lstsq(smps, wnated_fa)[0]
        base_fa = (fa1[i] + fa2[i]) / 2  # FIXME: not sure if this the best choice
        amps = amps / base_fa

        base_fas.append(base_fa)
        coil_amps.append(amps)

        actual_map = np.zeros((h, w))
        actual_map[mask] = smps @ (amps * base_fa)
        assert (actual_map[mask] > 0).all()
        actual_maps.append(actual_map[None, ...])
        # print(f'Max degree offset: {np.rad2deg(np.max(np.abs(actual_map - wnated_fa_2d)))}')

    # find clusters
    actual_maps = np.concatenate(actual_maps, axis=0)
    actual_maps = actual_maps[:, mask]
    actual_maps = rearrange(actual_maps, "n k -> k n")
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(actual_maps)

    voxels_labels = kmeans.labels_
    clusters = kmeans.cluster_centers_

    # compute subspace for the clusters

    return base_fas, coil_amps, actual_maps


def birdcage_excitation(fa1, fa2, smps, n_clusters: int = 5):
    assert len(fa1) == len(fa2)
    c, h, w = smps.shape
    smps = np.abs(smps)
    # scale the sum to one
    smps = smps / np.average(np.sum(smps, axis=0))
    fa_map = np.zeros(smps.shape[1:] + (len(fa1),))
    for i, map in tqdm(enumerate(smps)):
        if i % 2 == 0:
            fa_map += map[..., None] * fa1
        else:
            fa_map += map[..., None] * fa2

    assert (fa_map > 0).all()

    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(rearrange(fa_map, "h w n -> (h w) n"))

    voxels_labels = rearrange(kmeans.labels_, "(h w) -> h w", h=h, w=w)
    clusters = kmeans.cluster_centers_
    clustered_fa = clusters[voxels_labels]

    return clusters, voxels_labels, fa_map, clustered_fa


def alternating_excitation(fa, modes, n_clusters: int = 5):
    n_modes, h, w = modes.shape
    assert modes.min() > 0
    # scale the sum to one
    fa_map = np.zeros(modes.shape[1:] + (len(fa),))
    modes = np.tile(modes, [int(np.ceil(len(fa) / n_modes)), 1, 1])
    modes = rearrange(modes, 'n h w -> h w n')
    fa_map = modes[..., :len(fa)] * fa

    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(rearrange(fa_map, "h w n -> (h w) n"))

    voxels_labels = rearrange(kmeans.labels_, "(h w) -> h w", h=h, w=w)
    clusters = kmeans.cluster_centers_
    clustered_fa = clusters[voxels_labels]

    return clusters, voxels_labels, fa_map, clustered_fa


def jinc(x):
    # Ensure we don't divide by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        result = j1(np.pi * x) / (np.pi * x)
        result[x == 0] = 0.5  # handle the singularity
    return result

def smooth_2d_map(size):
    seed = np.random.randint(0, 1000)
    # Create a grid of x and y values
    scale = 40.0  # Increase the scale for smoother noise
    octaves = 8  # Increase the number of octaves for more details
    persistence = 0.6  # Adjust the persistence to control smoothness
    lacunarity = 2.0  # Adjust the lacunarity for more complex patterns
    # Create an empty array
    smooth_values = np.empty((size, size))

    # Generate Perlin noise values
    for i in range(size):
        for j in range(size):
            smooth_values[i, j] = noise.snoise2(i / scale,
                                                j / scale,
                                                octaves=octaves,
                                                persistence=persistence,
                                                lacunarity=lacunarity,
                                                repeatx=1024,
                                                repeaty=1024,
                                                base=seed)

    # Rescale the values to be in the range [0.8, 1.2]
    min_value = 0.8
    max_value = 1.2
    smooth_values = (smooth_values + 1) * (max_value - min_value) / 2 + min_value
    return smooth_values

if __name__ == "__main__":
    data = quant_phantom()
    proc_data = {}
    keys = ["t1", "t2", "pd"]
    for key in keys:
        proc_data[key] = sp.resize(data[key].astype(np.float32)[100], (220, 220))

    mask = proc_data["pd"] > 1e-5

    seq_0 = FISP.load("./logs/two_seqs/2023-10-29_14-39/seq_0_7.898")
    seq_1 = FISP.load("./logs/two_seqs/2023-10-29_14-39/seq_1_7.902")
    im_size = (220, 220)
    n_coil = 8
    # b1_map = simple_b1_map(im_size)
    mode_1 = get_b1_map(im_size, 0, scale=1)
    mode_2 = get_b1_map(im_size, 2, scale=1)
    mode_3 = get_b1_map(im_size, 3, scale=1)
    # mode_1 = smooth_2d_map(im_size[0])
    # mode_2 = smooth_2d_map(im_size[0])
    # mode_3 = smooth_2d_map(im_size[0])
    # sum_modes = mode_1 + mode_2
    # scale = 1 / np.average(sum_modes)
    # mode_1 = scale * mode_1
    # mode_2 = scale * mode_2

    # map = simple_b1_map(im_size)
    # mode_1 = np.where(map == 0, 1, 0)
    # mode_2 = np.where(map == 1, 1, 0)

    mode_1 = np.ones_like(mode_1)
    clusters, voxel_labels, fa_map, clustered_fa_map = birdcage_excitation(
        seq_0.flip_angles.detach().cpu().numpy(),
        seq_1.flip_angles.detach().cpu().numpy(),
        mode_1[None, ...],
        5,
    )

    # clusters, voxel_labels, fa_map, clustered_fa_map = birdcage_excitation(
    #     seq_0.flip_angles.detach().cpu().numpy(),
    #     seq_1.flip_angles.detach().cpu().numpy(),
    #     np.concatenate((mode_1[None, ...], mode_2[None, ...]), axis=0),
    #     20,
    # )

    # modes = np.concatenate(
    #     (mode_1[None, ...], mode_2[None, ...], mode_3[None, ...]), axis=0
    # )
    # clusters, voxel_labels, fa_map, clustered_fa_map = alternating_excitation(
    #     seq_0.flip_angles.detach().cpu().numpy(),
    #     modes,
    #     10,
    # )

    clustering_err = np.mean(np.rad2deg(np.abs(clustered_fa_map - fa_map)), axis=-1)
    plt.imshow(clustering_err)
    plt.colorbar()
    plt.savefig('clustering_err.png')

    np.save("./clusters", clusters)
    np.save("./voxel_labels", voxel_labels)
    np.save("./fa_map", fa_map)
