using Pkg
# Pkg.instantiate()
Pkg.activate(".")
using AcausalNets
using QuantumInformation
using LightGraphs
using LinearAlgebra

var_a = Variable(:a, 3) # the placement of the prize
var_c = Variable(:b, 2) # the initial choice of the door by the player
var_e = Variable(:c, 2) # the door opened by the host

roA = diagm(0 =>[
        1/3, 1/3, 1/3
        ]); #

sys_a = DiscreteQuantumSystem([var_a], roA)

roCwA = diagm(0 =>[
        0.7, 0.3, # A = 0
        0.2, 0.8, # A = 2
        0.4, 0.6 # A = 1
        ]); #

sys_CwA = DiscreteQuantumSystem([var_a], [var_c], roCwA)

roEwC = diagm(0 =>[
        0.3, 0.7, # C = 0
        0.6, 0.4 # C = 1
        ]); #

sys_EwC = DiscreteQuantumSystem([var_c], [var_e], roEwC)







# this matrix represents the optimal strategy in classical case
# - always changing the choice
roDwBC_changing = diagm(0 => [
        1,0,0, # B=0, C=0 (impossible situation)
        0,0,1, #B=0, C=1
        0,1,0, #B=0, C=2
        0,0,1, #B=1, C=0
        0,1,0, #B=1, C=1 (impossible situation)
        1,0,0, #B=1, C=2
        0,1,0, #B=2, C=0
        1,0,0, #B=2, C=1
        0,0,1 #B=2, C=2 (impossible situation)
        ])
# this matrix represents the unoptimal strategy in classical case
# - never changing the choice
# this matrix could actually be simplified to be roDwB
# since in this case, D == B and C is irrelevant
# but we'll keep the matrices in the same shape in order to later
# test mixed strategies of the player.
roDwBC_staying = diagm(0 => [
        1,0,0, # B=0, C=0 (impossible situation)
        1,0,0, #B=0, C=1
        1,0,0, #B=0, C=2
        0,1,0, #B=1, C=0
        0,1,0, #B=1, C=1 (impossible situation)
        0,1,0, #B=1, C=2
        0,0,1, #B=2, C=0
        0,0,1, #B=2, C=1
        0,0,1 #B=2, C=2 (impossible situation)
        ])

sys_d_bc_changing = DiscreteQuantumSystem([var_b, var_c], [var_d], roDwBC_changing)
sys_d_bc_staying = DiscreteQuantumSystem([var_b, var_c], [var_d], roDwBC_staying)
sys_d_bc = sys_d_bc_changing

roEwAD = diagm(0 => [
        0,1, # A=0, D=0 (player wins)
        1,0, #A=0, D=1
        1,0, #A=0, D=2
        1,0, #A=1, D=0
        0,1, #A=1, D=1 (player wins)
        1,0, #A=1, D=2
        1,0, #A=2, D=0
        1,0, #A=2, D=1
        0,1 #A=2, D=2 (player wins)
        ])
sys_e_ad = DiscreteQuantumSystem([var_a, var_d], [var_e], roEwAD);

roAB_different = 1/6*(ket(2,9)+ket(4,9))*(bra(2,9)+bra(4,9))+1/6*(ket(3,9)+ket(7,9))*(bra(3,9)+bra(7,9))+1/6*(ket(6,9)+ket(8,9))*(bra(6,9)+bra(8,9))
roAB_test = 1/6*(ket(1,9)+ket(2,9))*(bra(1,9)+bra(2,9))+1/6*(ket(5,9)+ket(6,9))*(bra(5,9)+bra(6,9))+1/6*(ket(9,9)+ket(7,9))*(bra(9,9)+bra(7,9))

sys_ab_different = DiscreteQuantumSystem([var_a, var_b], roAB_test)
an_different = AcausalNet()

push!(an_different, sys_ab_different)
push!(an_different, sys_c_ab)
push!(an_different, sys_d_bc)
push!(an_different, sys_e_ad)

an = an_different
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
     b_obs,
     c_obs,
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
