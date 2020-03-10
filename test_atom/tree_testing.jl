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
var_d = Variable(:d, 3) # the door opened by the player
var_e = Variable(:e, 2) # whether the player has won (0 = lost, 1 = won)

# rules of the host
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
# println("sys_c_ab")
# show(Base.stdout, "text/plain", real(distribution(sys_c_ab)))

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

# println("\nsys_d_bc")
# show(Base.stdout, "text/plain", real(distribution(sys_d_bc)))

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

# println("\nsys_e_ad")
# show(Base.stdout, "text/plain", real(distribution(sys_e_ad)))

roAB_different = 1/6*(ket(2,9)+ket(4,9))*(bra(2,9)+bra(4,9))+1/6*(ket(3,9)+ket(7,9))*(bra(3,9)+bra(7,9))+1/6*(ket(6,9)+ket(8,9))*(bra(6,9)+bra(8,9))

sys_ab_different = DiscreteQuantumSystem([var_a, var_b], roAB_different)

# println("\nsys_ab_different")
# show(Base.stdout, "text/plain", real(distribution(sys_ab_different)))

an_different = AcausalNet()

push!(an_different, sys_ab_different)
# show(an_different)
push!(an_different, sys_c_ab)
# show(an_different)
push!(an_different, sys_d_bc)
# show(an_different)
push!(an_different, sys_e_ad)
# show(an_different)

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

# [show(Base.stdout, "text/plain", real(distribution(c))) for c in an.systems]
# [println(variables(c)) for c in an.systems ]

observations = Evidence{Matrix{Complex{Float64}}}[
#     a_obs,
     # b_obs,
#    c_obs,
    d_obs,
     # e_obs
    ]
to_infer = [var_a, var_b]
inferred_system, debug = infer_join_tree(an, to_infer, observations)
inferred_system_naive, debug_naive = infer_naive(an, to_infer, observations)
inferred_system_belief, debug_belief = infer_belief(an, to_infer, observations)


real(distribution(inferred_system))

real(distribution(inferred_system_naive))

real(distribution(inferred_system_belief))


observations_jt, mg, enforced_mg, tri_mg, cliques, parent_cliques, initialized_jt, propagated_jt, jt = debug
show(jt)
[show(Base.stdout, "text/plain", real(distribution(v.distribution))) for v in initialized_jt]
