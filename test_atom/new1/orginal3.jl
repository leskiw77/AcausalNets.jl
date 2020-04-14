using Pkg
# Pkg.instantiate()
Pkg.activate(".")
using AcausalNets
using QuantumInformation
using LightGraphs
using LinearAlgebra

var_a = Variable(:a, 3)
var_b = Variable(:b, 3)
var_c = Variable(:c, 3)
var_d = Variable(:d, 3)

roA = diagm(0 =>[1/3,1/3,1/3])
sys_a = DiscreteQuantumSystem([var_a,], roA)


# rules of the host
roBwA = diagm(0 =>[
        1/3,1/3,1/3,
        1/5,0,4/5,
        0,1/2,1/2
        ]); #1
sys_b_a = DiscreteQuantumSystem([var_a,], [var_b], roBwA)

roCwB = diagm(0 =>[
        0,1/2,1/2,
        1,0,0,
        1/2,1/2,0
        ]);
sys_c_b = DiscreteQuantumSystem([var_b], [var_c], roCwB)


roDwC = diagm(0 =>[
        0,1/2,1/2,
        1,0,0,
        1/2,1/2,0
        ]); #
sys_d_c = DiscreteQuantumSystem([var_c], [var_d], roDwC)

roAC = 1/6*(ket(2,9)+ket(4,9))*(bra(2,9)+bra(4,9))+1/6*(ket(3,9)+ket(7,9))*(bra(3,9)+bra(7,9))+1/6*(ket(6,9)+ket(8,9))*(bra(6,9)+bra(8,9))
sys_ac = DiscreteQuantumSystem([var_a, var_c], roAC)

an = AcausalNet()

push!(an, sys_ac)
push!(an, sys_b_a)
push!(an, sys_d_c)


show(an)


a = 3
b = 3
c = 1
d = 3
e = 1

a_obs = Evidence([var_a], ketbra(a,a,3))
b_obs = Evidence([var_b], ketbra(b,b,3))
c_obs = Evidence([var_c], ketbra(c,c,3))
d_obs = Evidence([var_d], ketbra(d,d,3))
e_obs = Evidence([var_e], ketbra(e,e,2))

observations = Evidence{Matrix{Complex{Float64}}}[
     a_obs,
       # b_obs,
      # c_obs,
#     d_obs,
#     e_obs
    ]



to_infer = [var_c]
inferred_system, debug = infer_join_tree(an, to_infer, observations)
inferred_system_naive, debug_naive = infer_naive(an, to_infer, observations)
inferred_system_belief, debug_belief = infer_belief(an, to_infer, observations)


real(distribution(inferred_system))

real(distribution(inferred_system_naive))

real(distribution(inferred_system_belief))

show(debug[1])

show(moral_graph(an))
