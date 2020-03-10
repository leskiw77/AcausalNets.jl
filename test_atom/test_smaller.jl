using Pkg
# Pkg.instantiate()
Pkg.activate(".")
using AcausalNets
using QuantumInformation
using LightGraphs
using LinearAlgebra

var_a = Variable(:a, 3) # the placement of the prize
var_b = Variable(:b, 3) # the initial choice of the door by the player
var_c = Variable(:c, 3) # the door opened by the host
var_f = Variable(:f, 3) # extra for tests

roCwAB = diagm(0 =>[
        0,1/2,1/2, #A=0, B=0
        0,0,1, #A=0, B=1
        0,1,0, #A=0, B=2
        0,0,1, #A=1, B=0
        1/2,0,1/2, #A=1, B=1
        1,0,0, #A=1, B=2
        0,1,0, #A=2, B=0
        1,0,0, #A=2, B=1
        1/2,1/2,0 #A=2, B=2
        ]); #

sys_c_ab = DiscreteQuantumSystem([var_a, var_b], [var_c], roCwAB)

# dodatek
rowFwAB = diagm(0 => [
        0, 1, 0,
        1, 0, 0,
        0, 1, 0,
        1, 0, 0,
        0, 1, 0,
        1, 0, 0,
        0, 1, 0,
        1, 0, 0,
        0, 1, 0
        ])
sys_f_ab = DiscreteQuantumSystem([var_a, var_b], [var_f], rowFwAB);
rowFwB = diagm(0 => [
        1, 0, 0,
        0, 0, 1,
        ])
sys_f_b = DiscreteQuantumSystem([var_b], [var_f], rowFwB);


roAB_different = 1/6*(ket(2,9)+ket(4,9))*(bra(2,9)+bra(4,9))+1/6*(ket(3,9)+ket(7,9))*(bra(3,9)+bra(7,9))+1/6*(ket(6,9)+ket(8,9))*(bra(6,9)+bra(8,9))

sys_ab_different = DiscreteQuantumSystem([var_a, var_b], roAB_different)
an_different = AcausalNet()

push!(an_different, sys_ab_different)
push!(an_different, sys_c_ab)
push!(an_different, sys_f_ab)
push!(an_different, sys_f_b)
show(an_different)

an = an_different
show(an, true)

a = 1
b = 1

a_obs = Evidence([var_a], ketbra(a,a,3))
b_obs = Evidence([var_b], ketbra(b,b,3))

observations = Evidence{Matrix{Complex{Float64}}}[
#     a_obs,
#     b_obs,
#     c_obs,
#     d_obs,
#     e_obs
    ]
to_infer = [var_a, var_b]
inferred_system, debug = infer_join_tree(an, to_infer, observations)
inferred_system_naive, debug_naive = infer_naive(an, to_infer, observations)
inferred_system_belief, debug_belief = infer_belief(an, to_infer, observations)

(observations_jt,propagated_jt,jt,) = debug
(dbn,cluster,obs,evidence_cluster,inferred_cluster) = debug_naive
show(jt)


real(distribution(inferred_system))

real(distribution(inferred_system_naive))

real(distribution(inferred_system_belief))
