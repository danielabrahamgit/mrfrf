import os
import torch
import numpy as np
import sigpy as sp
import sigpy.mri as mr

from mr_sim.trj_lib import trj_lib
from mr_sim.data_sim import data_sim, quant_phantom, mrf_sequence

# Tunable Params
n_coil = 12
R = 16.0
dt = 4e-6
nseg = 100
device_idx = 6
save_data_dir = f'./simulated_data/'

# Load t1/t2/pd maps
data = quant_phantom()
proc_data = {}
keys = ['t1', 't2', 'pd']
for key in keys:
    proc_data[key] = sp.resize(data[key].astype(np.float32)[100], (220, 220))
    proc_data[key] = torch.from_numpy(proc_data[key])
t1s, t2s, pds = proc_data['t1'], proc_data['t2'], proc_data['pd']
im_size = t1s.shape

# Load MRF sequence 
data = mrf_sequence()
data["TR_init"][0][-1] = 15.0
trs = torch.from_numpy(data["TR_init"][0].astype(np.float32))
rfs = torch.deg2rad(torch.from_numpy(data["FA_init"][0].astype(np.float32)))

# Data sim object
ds = data_sim(im_size=im_size, 
              rfs=rfs,
              trs=trs,
              te=3, 
              device_idx=device_idx)

# Make maps 
mps = mr.birdcage_maps((n_coil, *im_size), r=1.25).astype(np.complex64)
mps = torch.from_numpy(mps)

# Trajectory
trj_obj = trj_lib(im_size)
trj = trj_obj.gen_MRF_trj(ntr=len(rfs),
                            n_shots=16,
                            R=R)
trj = torch.from_numpy(trj)

# Estimate subspace from range of t1/t2 from xiaozhi's paper
t1_vals = np.arange(20, 3000, 20)
t1_vals = np.append(t1_vals, np.arange(3200, 5000, 200))
t2_vals = np.arange(10, 200, 2)
t2_vals = np.append(t2_vals, np.arange(220, 1000, 20))
t2_vals = np.append(t2_vals, np.arange(1050, 2000, 50))
t2_vals = np.append(t2_vals, np.arange(2100, 4000, 100))
t1_vals, t2_vals = np.meshgrid(t1_vals, t2_vals, indexing="ij")
t1_vals, t2_vals = t1_vals.flatten(), t2_vals.flatten()
phi = ds.est_subspace(t1_vals, t2_vals, sing_val_thresh=0.9)
    
# Simulate K-space data
ksp, _, _ = ds.sim_ksp(t1_map=t1s,
                       t2_map=t2s,
                       pd_map=pds,
                       mps=mps,
                       trj=trj,
                       coil_batch_size=n_coil,
                       seg_batch_size=1)

# DCF
dcf = ds.est_dcf(trj)

# Create folder with final data
try:
    os.mkdir(save_data_dir)
except FileExistsError:
    print(f"Directory {save_data_dir} already exists, ", end="")
    print("contents will be overwritten!")
except FileNotFoundError:
    raise FileNotFoundError("Invalid directory name")

# Save data
print(f'Readout time = {trj.shape[0] * 1e3 * dt:.3f}(ms)')
print(f'trj shape = {list(trj.shape)}')
print(f'dcf shape = {list(dcf.shape)}')
print(f'ksp shape = {list(ksp.shape)}')
print(f'mps shape = {list(mps.shape)}')
print(f'phi shape = {list(phi.shape)}')
np.save(save_data_dir + "trj.npy", trj.numpy())
np.save(save_data_dir + "dcf.npy", dcf.numpy())
np.save(save_data_dir + "ksp.npy", ksp.numpy())
np.save(save_data_dir + "mps.npy", mps.numpy())
np.save(save_data_dir + "phi.npy", phi.numpy())
print('saved!')
