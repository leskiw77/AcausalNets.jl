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


roA = diagm(0 =>[1/2,1/2])
sys_a = DiscreteQuantumSystem([var_a,], roA)


# rules of the host
roBwA = diagm(0 =>[
        1/5,4/5,
        1/2,1/2
        ]); #1
sys_b_a = DiscreteQuantumSystem([var_a,], [var_b], roBwA)

roCwB = diagm(0 =>[
        1/2,1/2,
        1/2,1/2
        ]); #
sys_c_b = DiscreteQuantumSystem([var_b], [var_c], roCwB)


roDwC = diagm(0 =>[
        1/2,1/2,
        1,0,
        ]); #
sys_d_c = DiscreteQuantumSystem([var_c], [var_d], roDwC)

roAC = 1/6*(ket(2,9)+ket(4,9))*(bra(2,9)+bra(4,9))+1/6*(ket(3,9)+ket(7,9))*(bra(3,9)+bra(7,9))+1/6*(ket(6,9)+ket(8,9))*(bra(6,9)+bra(8,9))


roAC = diagm(0 =>[
        1/2,1/2,
        1,0,
        1/2,1/2,
        1,0,
        ]);
sys_ac = DiscreteQuantumSystem([var_a, var_c], [var_b], roAC)


function check_distribution(
    distribution,
    parents::Vector{Variable},
    variables::Vector{Variable}
    )
    dimensions = size(distribution)
    total_ncategories = prod([v.ncategories for v in vcat(parents, variables)])
    println(dimensions[1], dimensions[2], total_ncategories)
    dimensions[1] == dimensions[2] == total_ncategories
end

check_distribution(roAC, [var_a, var_c], [var_b])


an = AcausalNet()

push!(an, sys_ac)
push!(an, sys_b_a)
push!(an, sys_c_b)
push!(an, sys_d_c)


show(an)







roEwCD = diagm(0 => [
        0,1,
        1,0,
        1,0,
        1,0,
        0,1,
        1,0,
        1,0,
        1,0,
        0,1
        ])
sys_e_ac = DiscreteQuantumSystem([var_c, var_d], [var_e], roEwCD);

roAB_different = 1/6*(ket(2,9)+ket(4,9))*(bra(2,9)+bra(4,9))+1/6*(ket(3,9)+ket(7,9))*(bra(3,9)+bra(7,9))+1/6*(ket(6,9)+ket(8,9))*(bra(6,9)+bra(8,9))
roAB_test = 1/6*(ket(1,9)+ket(2,9))*(bra(1,9)+bra(2,9))+1/6*(ket(5,9)+ket(6,9))*(bra(5,9)+bra(6,9))+1/6*(ket(9,9)+ket(7,9))*(bra(9,9)+bra(7,9))
roAB_same = 1/3 * (ket(1,9)+ket(5,9)+ket(9,9))*(bra(1,9)+bra(5,9)+bra(9,9))
sys_ab = DiscreteQuantumSystem([var_a, var_b], roAB_same)
sys_ab = DiscreteQuantumSystem([var_a, var_b], roAB_test)


an = AcausalNet()

push!(an, sys_ab)

push!(an, sys_c_a)
push!(an, sys_d_b)

push!(an, sys_e_ac)

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
