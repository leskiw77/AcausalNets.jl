using Pkg
# Pkg.instantiate()
Pkg.activate(".")
using AcausalNets
using QuantumInformation
using LightGraphs
using LinearAlgebra

var_a = Variable(:a, 3)
var_b = Variable(:b, 3)
var_c = Variable(:c, 2)
var_d = Variable(:d, 2)
var_e = Variable(:e, 2)
var_f = Variable(:f, 2)
var_g = Variable(:g, 2)


roCwA = diagm(0 =>[
        1/3,1/3,
        1/5,4/5,
        1/3,2/3,
        ]); #1
sys_c_a = DiscreteQuantumSystem([var_a,], [var_c], roCwA)

roBwD = diagm(0 =>[
        1/2,1/2,
        1/2,1/2,
        1/2,1/2
        ]); #
sys_d_b = DiscreteQuantumSystem([var_b], [var_d], roBwD)


roEwC = diagm(0 =>[
        1/3,2/3,
        1/5,4/5
        ]); #1
sys_e_c = DiscreteQuantumSystem([var_c,], [var_e], roEwC)

roFwD = diagm(0 =>[
        1/2,1/2,
        1/2,1/2
        ]); #
sys_f_d = DiscreteQuantumSystem([var_d], [var_f], roFwD)


roGwEF = diagm(0 => [
        0,1,
        1,0,
        1,0,
        1,0,
        ])
sys_g_ef = DiscreteQuantumSystem([var_e, var_f], [var_g], roGwEF);

roAB_same = 1/2 * (ket(2,4)+ket(3,4))*(bra(2,4)+bra(3,4))
roAB_test = 1/6*(ket(1,9)+ket(2,9))*(bra(1,9)+bra(2,9))+1/6*(ket(5,9)+ket(6,9))*(bra(5,9)+bra(6,9))+1/6*(ket(9,9)+ket(7,9))*(bra(9,9)+bra(7,9))


sys_ab = DiscreteQuantumSystem([var_a, var_b], roAB_test)

roAB_same



an = AcausalNet()

push!(an, sys_ab)

push!(an, sys_c_a)
push!(an, sys_d_b)

push!(an, sys_e_c)
push!(an, sys_f_d)

push!(an, sys_g_ef)

show(an)



a = 2
b = 2
c = 1
d = 2
e = 1

a_obs = Evidence([var_a], ketbra(a,a,3))
b_obs = Evidence([var_b], ketbra(b,b,3))
c_obs = Evidence([var_c], ketbra(c,c,2))
d_obs = Evidence([var_d], ketbra(d,d,2))
e_obs = Evidence([var_e], ketbra(e,e,2))
f_obs = Evidence([var_f], ketbra(e,e,2))
g_obs = Evidence([var_g], ketbra(e,e,2))

observations = Evidence{Matrix{Complex{Float64}}}[
     a_obs,
       # b_obs,
      c_obs,
    # d_obs,
#     e_obs,
        f_obs
        # g_obs
    ]



to_infer = [var_b]
inferred_system, debug = infer_join_tree(an, to_infer, observations)
inferred_system_naive, debug_naive = infer_naive(an, to_infer, observations)
inferred_system_belief, debug_belief = infer_belief(an, to_infer, observations)

ptrace(real(distribution(inferred_system)), [3,3], [1])
ptrace(real(distribution(inferred_system)), [3,3], [2])
real(distribution(inferred_system))

real(distribution(inferred_system_naive))

real(distribution(inferred_system_belief))

show(debug[1])

show(moral_graph(an))
