using Pkg
# Pkg.instantiate()
Pkg.activate(".")
using AcausalNets
using QuantumInformation
using LightGraphs
using LinearAlgebra

var_a = Variable(:a, 2)
var_b = Variable(:b, 2)
var_c = Variable(:c, 2)
var_d = Variable(:d, 2)

var_e = Variable(:e, 2)
var_f = Variable(:f, 2)
var_g = Variable(:g, 2)
var_h = Variable(:h, 2)

var_i = Variable(:i, 2)
var_j = Variable(:j, 2)

var_k = Variable(:k, 2)

ro_same = 1/2 * (ket(2,4)+ket(3,4))*(bra(2,4)+bra(3,4))


sys_ab = DiscreteQuantumSystem([var_a, var_b], ro_same)
sys_cd = DiscreteQuantumSystem([var_c, var_d], ro_same)

ro2to2 = diagm(0 =>[
        1/3,1/3,
        1/5,4/5
        ]);
sys_a_e = DiscreteQuantumSystem([var_a], [var_e], ro2to2)
sys_d_f = DiscreteQuantumSystem([var_b], [var_f], ro2to2)
sys_c_g = DiscreteQuantumSystem([var_c], [var_g], ro2to2)
sys_d_h = DiscreteQuantumSystem([var_d], [var_h], ro2to2)

ro4to2 = diagm(0 => [
        0,1,
        1,0,
        1,0,
        1,0,
        ])

sys_ef_i = DiscreteQuantumSystem([var_e, var_f], [var_i], ro4to2)
sys_gh_j = DiscreteQuantumSystem([var_g, var_h], [var_j], ro4to2)
sys_ij_k = DiscreteQuantumSystem([var_i, var_j], [var_k], ro4to2)

an = AcausalNet()

push!(an, sys_ab)
push!(an, sys_cd)

push!(an, sys_a_e)
push!(an, sys_d_f)
push!(an, sys_c_g)
push!(an, sys_d_h)

push!(an, sys_ef_i)
push!(an, sys_gh_j)

push!(an, sys_ij_k)

show(an)



a = 2
b = 2
c = 1
d = 2
e = 1

a_obs = Evidence([var_a], ketbra(a,a,2))
b_obs = Evidence([var_b], ketbra(b,b,2))
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
