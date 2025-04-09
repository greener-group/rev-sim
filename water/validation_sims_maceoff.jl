# Run water simulation with MACE-OFF
# See https://github.com/greener-group/rev-sim for setup instructions
# Licence is MIT
# An appropriate conda environment should be activated
# Use CUDA_VISIBLE_DEVICES to set the GPU to run on
# Unlike the other scripts which use Molly v0.21.1, here Molly v0.22.1 is required

ENV["JULIA_CONDAPKG_BACKEND"] = "Null"

using Molly
using PythonCall

model_fp = "trained_model_maceoff.pt" # Set to nothing to use the pre-trained MACE-OFF model
out_dir = "trajs_maceoff"
traj_fp       = "$out_dir/water.dcd"
energy_fp     = "$out_dir/water.txt"
traj_fp_gas   = "$out_dir/water_gas.dcd"
energy_fp_gas = "$out_dir/water_gas.txt"
n_steps = 2_400_000 # 1.2 ns
n_steps_log = 4_000 # 2 ps
dt = 0.0005f0u"ps"
temp = 300.0f0u"K"

pymod_mc = pyimport("mace.calculators")
pymod_torch = pyimport("torch")
py_ase_calc = pymod_mc.mace_off(model="small", device="cuda", default_dtype="float32")

if !isnothing(model_fp)
    py_ase_calc.models[0].load_state_dict(pymod_torch.load(model_fp))
end

coords = []
for line in readlines("../../tip3p.xyz")[3:end]
    cols = split(line)
    push!(coords, SVector{3, Float32}(parse.(Float32, cols[2:end]))u"Å")
end
coords = [coords...]
n_atoms = length(coords)

boundary = CubicBoundary(30.0f0u"Å")
atoms = [Atom(mass=(i % 3 == 1 ? Float32(16.0 / Unitful.Na) : Float32(1.008 / Unitful.Na))u"g/mol",
              charge=0.0f0) for i in 1:n_atoms]
atoms_data = [AtomData(
                    atom_name=(i % 3 == 1 ? "O" : (i % 3 == 2 ? "H1" : "H2")),
                    res_number=cld(i, 3),
                    res_name="HOH",
                    element=(i % 3 == 1 ? "O" : "H"),
              ) for i in 1:n_atoms]

calc = ASECalculator(
    ase_calc=py_ase_calc,
    atoms=atoms,
    coords=coords,
    boundary=boundary,
    atoms_data=atoms_data,
)

function potential_energy_wrapper(sys, neighbors, step_n::Integer; n_threads::Integer,
                                  current_potential_energy=nothing, kwargs...)
    pe = potential_energy(sys, neighbors, step_n; n_threads=n_threads)
    open(energy_fp, "a") do of
        println(of, pe)
    end
    println(step_n, " - ", pe)
    return pe
end

sys = System(
    atoms=atoms,
    coords=coords,
    boundary=boundary,
    atoms_data=atoms_data,
    general_inters=(calc,),
    force_units=u"eV/Å",
    energy_units=u"eV",
    loggers=(
        pe=GeneralObservableLogger(potential_energy_wrapper, typeof(1.0f0u"eV"), n_steps_log),
        writer=TrajectoryWriter(n_steps_log, traj_fp),
    )
)

random_velocities!(sys, temp)
simulator = Langevin(
    dt=dt,
    temperature=temp,
    friction=1.0f0u"ps^-1",
    coupling=MonteCarloBarostat(1.0f0u"bar", temp, sys.boundary),
)

simulate!(sys, simulator, n_steps)

coords = []
for line in readlines("../../tip3p.xyz")[3:5] # First molecule only
    cols = split(line)
    push!(coords, SVector{3, Float32}(parse.(Float32, cols[2:end]))u"Å")
end
coords = [coords...]
n_atoms = length(coords)

boundary = CubicBoundary(30.0f0u"Å") # Anything larger than the receptive field should be fine
atoms = [Atom(mass=(i % 3 == 1 ? Float32(16.0 / Unitful.Na) : Float32(1.008 / Unitful.Na))u"g/mol",
              charge=0.0f0) for i in 1:n_atoms]
atoms_data = [AtomData(
                    atom_name=(i % 3 == 1 ? "O" : (i % 3 == 2 ? "H1" : "H2")),
                    res_number=cld(i, 3),
                    res_name="HOH",
                    element=(i % 3 == 1 ? "O" : "H"),
              ) for i in 1:n_atoms]

calc = ASECalculator(
    ase_calc=py_ase_calc,
    atoms=atoms,
    coords=coords,
    boundary=boundary,
    atoms_data=atoms_data,
)

function potential_energy_wrapper_gas(sys, neighbors, step_n::Integer; n_threads::Integer,
                                      current_potential_energy=nothing, kwargs...)
    pe = potential_energy(sys, neighbors, step_n; n_threads=n_threads)
    open(energy_fp_gas, "a") do of
        println(of, pe)
    end
    println(step_n, " - ", pe)
    return pe
end

sys = System(
    atoms=atoms,
    coords=coords,
    boundary=boundary,
    atoms_data=atoms_data,
    general_inters=(calc,),
    force_units=u"eV/Å",
    energy_units=u"eV",
    loggers=(
        pe=GeneralObservableLogger(potential_energy_wrapper_gas, typeof(1.0f0u"eV"), n_steps_log),
        writer=TrajectoryWriter(n_steps_log, traj_fp_gas),
    )
)

random_velocities!(sys, temp)
simulator = Langevin(
    dt=dt,
    temperature=temp,
    friction=1.0f0u"ps^-1", # No barostat
)

simulate!(sys, simulator, n_steps)
