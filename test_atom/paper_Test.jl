using Pkg
# Pkg.instantiate()
Pkg.activate(".")
using AcausalNets
using QuantumInformation
using LightGraphs
using LinearAlgebra

var_a = Variable(:a, 2) # the placement of the prize
var_b = Variable(:b, 2) # the initial choice of the door by the player
var_c = Variable(:c, 2) # the door opened by the host
var_d = Variable(:d, 2) # the door opened by the player
var_e = Variable(:e, 2) # whether the player has won (0 = lost, 1 = won)
var_f = Variable(:f, 2) # whether the player has won (0 = lost, 1 = won)
var_g = Variable(:g, 2) # whether the player has won (0 = lost, 1 = won)
var_h = Variable(:h, 2) # whether the player has won (0 = lost, 1 = won)
# rules of the host

roA = diagm(0 =>[
        1/2, 1/2
        ]); #

sys_a = DiscreteQuantumSystem([var_a], roA)

roBwA = diagm(0 =>[
        1/2, 1/2, # A = 0
        0.4, 0.6, # A = 1
        ]); #

sys_BwA = DiscreteQuantumSystem([var_a], [var_b], roBwA)

roCwA = diagm(0 =>[
        0.7, 0.3, # A = 0
        0.2, 0.8, # A = 1
        ]); #

sys_CwA = DiscreteQuantumSystem([var_a], [var_c], roCwA)

roDwB = diagm(0 =>[
        0.9, 0.1, # A = 0
        0.5, 0.5, # A = 1
        ]); #

sys_DwB = DiscreteQuantumSystem([var_b], [var_d], roDwB)


roEwC = diagm(0 =>[
        0.3, 0.7, # C = 0
        0.6, 0.4 # C = 1
        ]); #

sys_EwC = DiscreteQuantumSystem([var_c], [var_e], roEwC)

roFwDE = diagm(0 =>[
        0.01, 0.99, # C = 0
        0.01, 0.99, # C = 1
        0.01, 0.99, # C = 0
        0.99, 0.01 # C = 1
        ]); #

sys_FwDE = DiscreteQuantumSystem([var_d, var_e], [var_f], roFwDE)

roGwC = diagm(0 =>[
        0.8, 0.2, # C = 0
        0.1, 0.9 # C = 1
        ]); #

sys_GwC = DiscreteQuantumSystem([var_c], [var_g], roGwC)

roHwEG = diagm(0 =>[
        0.05, 0.95, # C = 0
        0.95, 0.05, # C = 1
        0.95, 0.05, # C = 0
        0.95, 0.05 # C = 1
        ]); #

sys_HwEG = DiscreteQuantumSystem([var_e, var_g], [var_h], roHwEG)
an_different = AcausalNet()

push!(an_different, sys_a)
push!(an_different, sys_BwA)
push!(an_different, sys_CwA)
push!(an_different, sys_DwB)
push!(an_different, sys_EwC)
push!(an_different, sys_FwDE)
push!(an_different, sys_GwC)
push!(an_different, sys_HwEG)
an = an_different
show(an)
a = 1
b = 1
c = 2
d = 3
e = 1

a_obs = Evidence([var_a], ketbra(a,a,3))
b_obs = Evidence([var_b], ketbra(b,b,3))
c_obs = Evidence([var_c], ketbra(c,c,3))
d_obs = Evidence([var_d], ketbra(d,d,3))
e_obs = Evidence([var_e], ketbra(e,e,2))

observations = Evidence{Matrix{Complex{Float64}}}[
#     a_obs,
#     b_obs,
#     c_obs,
#     d_obs,
#     e_obs
    ]
to_infer = [var_a]
inferred_system, debug = infer_join_tree(an, to_infer, observations)
inferred_system_naive, debug_naive = infer_naive(an, to_infer, observations)
inferred_system_belief, debug_belief = infer_belief(an, to_infer, observations)


real(distribution(inferred_system))

real(distribution(inferred_system_naive))

real(distribution(inferred_system_belief))
