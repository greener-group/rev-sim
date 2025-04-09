# Run water simulations across different temperatures to validate force fields
# See https://github.com/greener-group/rev-sim for setup instructions
# Licence is MIT

from openmm.app import *
from openmm import *
from openmm.unit import *
import os
import sys
from time import time

ff_name = sys.argv[1]
run_n  = int(sys.argv[2]) if len(sys.argv) > 2 else 0
n_runs = int(sys.argv[3]) if len(sys.argv) > 3 else 0

forcefield = ForceField(f"trained_force_fields/{ff_name}.xml")
out_dir = ff_name
temps = list(range(260, 365+1, 5)) # K
n_steps = 60000000 # 120 ns
n_steps_save = 25000 # 50 ps
n_steps_diff_equil = 2500000 # 5 ns
n_steps_diff = 100000 # 100 ps, shorter time step
n_steps_diff_save = 10000 # 10 ps, shorter time step
n_reps_diff = 5
dt = 0.002*picoseconds
dt_diff = 0.001*picoseconds
constraints = HBonds # None, HBonds or HAngles
rigidWater = True
pdb = PDBFile("tip3p.pdb")
platform = Platform.getPlatformByName("CUDA")

for ti, temp in enumerate(temps):
    if run_n > 0 and n_runs > 0:
        if not ((ti - run_n + 1) % n_runs == 0):
            continue

    traj_fp = f"{out_dir}/sim_{temp}K.dcd"
    if not os.path.isfile(traj_fp):
        system = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=PME,
            nonbondedCutoff=1*nanometer,
            constraints=constraints,
            rigidWater=rigidWater,
        )
        system.addForce(MonteCarloBarostat(1*bar, temp*kelvin))
        integrator = LangevinMiddleIntegrator(temp*kelvin, 1/picosecond, dt)
        simulation = Simulation(pdb.topology, system, integrator, platform)
        simulation.context.setPositions(pdb.positions)
        simulation.minimizeEnergy()
        simulation.reporters.append(DCDReporter(traj_fp, n_steps_save))
        simulation.reporters.append(StateDataReporter(f"{out_dir}/sim_{temp}K.log",
                            n_steps_save, step=True, potentialEnergy=True, temperature=True))

        t_start = time()
        simulation.step(n_steps)
        print(temp, "liquid", time() - t_start)

    for rep_n in range(1, n_reps_diff + 1):
        traj_fp_diff = f"{out_dir}/sim_diff_{rep_n}_{temp}K.dcd"
        if os.path.isfile(traj_fp_diff):
            continue

        system_equil = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=PME,
            nonbondedCutoff=1*nanometer,
            constraints=constraints,
            rigidWater=rigidWater,
        )
        system_equil.addForce(MonteCarloBarostat(1*bar, temp*kelvin))
        integrator_equil = LangevinMiddleIntegrator(temp*kelvin, 1/picosecond, dt)
        simulation_equil = Simulation(pdb.topology, system_equil, integrator_equil, platform)
        simulation_equil.context.setPositions(pdb.positions)
        simulation_equil.minimizeEnergy()

        t_start = time()
        simulation_equil.step(n_steps_diff_equil)
        state = simulation_equil.context.getState(getPositions=True, getVelocities=True)

        system = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=PME,
            nonbondedCutoff=1*nanometer,
            constraints=constraints,
            rigidWater=rigidWater,
        )
        integrator = VerletIntegrator(dt_diff)
        simulation = Simulation(pdb.topology, system, integrator, platform)
        simulation.reporters.append(DCDReporter(traj_fp_diff, n_steps_diff_save))
        simulation.reporters.append(StateDataReporter(f"{out_dir}/sim_diff_{rep_n}_{temp}K.log",
                            n_steps_diff_save, step=True, potentialEnergy=True,
                            totalEnergy=True, volume=True, temperature=True))

        bv = state.getPeriodicBoxVectors()
        simulation.context.setPeriodicBoxVectors(bv[0], bv[1], bv[2])
        simulation.context.setPositions(state.getPositions())
        simulation.context.setVelocities(state.getVelocities())
        simulation.step(n_steps_diff)
        print(temp, "diffusion", rep_n, time() - t_start)
