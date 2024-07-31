import cySim_lib as sl
from utilities import get_memory

# pos, vel = sim.test_c()
# fname = '/u/mesposito/Projects/velocity-power-spectrum/Test_sims/Default-L10-N32/output/snapshot_000'
# fname = '/ptmp/anruiz/Columbus_{0}_{2}/snapshots/snapdir_00{1}/snapshot_Columbus_{0}_{2}_00{1}'.format(0, 0, 'A') 
fname = '/ptmp/mesposito/CosmoSims/Col_0_N512_L2000.0_output/0.00/snapdir_000/snapshot_000'
ngrid = 256
sim = sl.G4_Simulation(fname)
print(f"Mem before getting pos: {get_memory()}")
pos = sim.get_pos()
vel = sim.get_vel()
print(f"Mem at the end: {get_memory()}")

sim.compute_SubGrid(ngrid = ngrid, fine_grid=False)
print(f"Mem after computing SubGrid: {get_memory()}")

sim.free_SubGrid()
print(f"Mem after freeing SubGrid: {get_memory()}")

sim.sample_velocity_field(ngrid = 256, size = 'XL')
sample_pos = sim.get_sample_pos()
sample_vel = sim.get_sample_vel()
print(f"Mem after doing the Voronoi sampling: {get_memory()}")

sim.free_sample()
print(f"Mem after freeing the sample: {get_memory()}")

# sim.free_all()
# print(f"Mem at the end: {get_memory()}")

