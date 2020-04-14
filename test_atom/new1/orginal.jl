using Pkg
# Pkg.instantiate()
Pkg.activate(".")
using AcausalNets
using QuantumInformation
using LightGraphs
using LinearAlgebra

import AcausalNets.Systems:
    sub_system,
    multiply_star,
    multiply_kron,
    merge_systems

import AcausalNets.Algebra:
    event
var_a = Variable(:a, 3) # the placement of the prize
var_b = Variable(:b, 3) # the initial choice of the door by the player
var_c = Variable(:c, 3) # the door opened by the host
var_d = Variable(:d, 3) # the door opened by the player
var_e = Variable(:e, 2) # whether the player has won (0 = lost, 1 = won)

roC = diagm(0 =>[
        1/3, 1/3, 1/3
        ]); #

sys_c = DiscreteQuantumSystem([var_c], roC)
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
roAB_same = 1/3 * (ket(1,9)+ket(5,9)+ket(9,9))*(bra(1,9)+bra(5,9)+bra(9,9))
sys_ab_same = DiscreteQuantumSystem([var_a, var_b], roAB_same)

sys_ab_different = DiscreteQuantumSystem([var_a, var_b], roAB_test)
an_different = AcausalNet()

push!(an_different, sys_ab_different)
push!(an_different, sys_a)
push!(an_different, sys_b)
push!(an_different, sys_c_ab)
push!(an_different, sys_d_bc)
push!(an_different, sys_e_ad)
show(an_different)
an = an_different
a = 2
b = 1
c = 3
d = 1
e = 1

a_obs = Evidence([var_a], ketbra(a,a,3))
b_obs = Evidence([var_b], ketbra(b,b,3))
c_obs = Evidence([var_c], ketbra(c,c,3))
d_obs = Evidence([var_d], ketbra(d,d,3))
e_obs = Evidence([var_e], ketbra(e,e,2))


observations = Evidence{Matrix{Complex{Float64}}}[
      #a_obs,
         # b_obs,
       d_obs,
#     d_obs,
#   e_obs
    ]
to_infer = [var_a]
inferred_system, debug = infer_join_tree(an, to_infer, observations)
inferred_system_naive, debug_naive = infer_naive(an, to_infer, observations)
inferred_system_belief, debug_belief = infer_belief(an, to_infer, observations)


real(distribution(inferred_system))

real(distribution(inferred_system_naive))

real(distribution(inferred_system_belief))

show(debug[1])

show(moral_graph(an))

Pabc = multiply_star()
function smallLikeZero(x)
        if real(x) < 0.00001
        zero(zero(Complex{Float64}))
        else
        x
        end
end

a = 2
b = 1
c = 2
d = 3
e = 1

a_obs = Evidence([var_a], ketbra(a,a,3))
b_obs = Evidence([var_b], ketbra(b,b,3))
c_obs = Evidence([var_c], ketbra(c,c,3))
d_obs = Evidence([var_d], ketbra(d,d,3))
e_obs = Evidence([var_e], ketbra(e,e,2))
import AcausalNets.Algebra: eye
Pade = event(roEwAD, multiply_kron(multiply_kron(eye(3), d_obs.distribution), eye(2))) / tr(event(roEwAD, multiply_kron(multiply_kron(eye(3), d_obs.distribution), eye(2))))
Pade_d = ptrace(Pade, [3, 3, 2], [1, 3])
Pade_e = ptrace(Pade, [3, 3, 2], [1, 2])
Pade_a = ptrace(Pade, [3, 3, 2], [2, 3])

Pbcd = event(roDwBC_changing, multiply_kron(multiply_kron(eye(3), eye(3)), d_obs.distribution)) / tr(event(roDwBC_changing, multiply_kron(multiply_kron(eye(3), eye(3)), d_obs.distribution)))
Pbcd_c = ptrace(Pbcd, [3, 3, 3], [1, 3])
Pbcd_d = ptrace(Pbcd, [3, 3, 3], [1, 2])
Pbcd_b = ptrace(Pbcd, [3, 3, 3], [2, 3])
PaBC = event(PabC, multiply_kron(multiply_kron(eye(3), b_obs.distribution), eye(3))) / tr(event(PabC, multiply_kron(multiply_kron(eye(3), b_obs.distribution), eye(3))))
PaBC_b = ptrace(PaBC, [3, 3, 3], [1, 3])
PaBC_c = ptrace(PaBC, [3, 3, 3], [1, 2])
PaBC_a = ptrace(PaBC, [3, 3, 3], [2, 3])


a_test_obs = diagm(0 =>[
        1/2, 0, 1/2
        ]); #
PABC = event(PaBC, multiply_kron(multiply_kron(a_test_obs, eye(3)), eye(3))) / tr(event(PaBC, multiply_kron(multiply_kron(a_test_obs, eye(3)), eye(3))))
PABC_c = ptrace(PaBC, [3, 3, 3], [1, 3])
PABC_b = ptrace(PaBC, [3, 3, 3], [1, 2])
PABC_a = ptrace(PaBC, [3, 3, 3], [2, 3])


Pabc = event(roCwAB, multiply_kron(roAB_test, eye(3))) / tr(event(roCwAB, multiply_kron(roAB_test, eye(3))))
Pabc_b = ptrace(Pabc, [3, 3, 3], [1, 3])
Pabc_c = ptrace(Pabc, [3, 3, 3], [1, 2])
Pabc_a = ptrace(Pabc, [3, 3, 3], [2, 3])
PabC = event(Pabc, multiply_kron(roAB_test, c_obs.distribution)) / tr(event(Pabc, multiply_kron(roAB_test, c_obs.distribution)))
PabC_b = ptrace(PabC, [3, 3, 3], [1, 3])
PabC_c = ptrace(PabC, [3, 3, 3], [1, 2])
PabC_a = ptrace(PabC, [3, 3, 3], [2, 3])
PaBC = event(PabC, multiply_kron(multiply_kron(eye(3), b_obs.distribution), eye(3))) / tr(event(PabC, multiply_kron(multiply_kron(eye(3), b_obs.distribution), eye(3))))
PaBC_b = ptrace(PaBC, [3, 3, 3], [1, 3])
PaBC_c = ptrace(PaBC, [3, 3, 3], [1, 2])
PaBC_a = ptrace(PaBC, [3, 3, 3], [2, 3])


# julia> merge
# merge           merge!           merge_vertices   merge_vertices!
# julia> merge
# merge           merge!           merge_vertices   merge_vertices!
# julia> merge
#
# julia> merge_systems([sys_c_ab, sys_d_bc])
# DiscreteSystem{AbstractArray{T,2} where T}(Variable[], Variable[Variable(:c, 3), Variable(:d, 3)], [2.9999999999999996 0.0 … 0.0 0.0; 0.0 2.9999999999999996
# … 0.0 0.0; … ; 0.0 0.0 … 2.9999999999999996 0.0; 0.0 0.0 … 0.0 2.9999999999999996])
#
# julia> x = merge_systems([sys_c_ab, sys_d_bc])
# DiscreteSystem{AbstractArray{T,2} where T}(Variable[], Variable[Variable(:c, 3), Variable(:d, 3)], [2.9999999999999996 0.0 … 0.0 0.0; 0.0 2.9999999999999996 … 0.0 0.0; … ; 0.0 0.0 … 2.9999999999999996 0.0; 0.0 0.0 … 0.0 2.9999999999999996])
#
# julia> x.variables
# 2-element Array{Variable,1}:
#  Variable(:c, 3)
#  Variable(:d, 3)
#
# julia> x.distribution
# 9×9 Array{Float64,2}:
#  3.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
#  0.0  3.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
#  0.0  0.0  3.0  0.0  0.0  0.0  0.0  0.0  0.0
#  0.0  0.0  0.0  3.0  0.0  0.0  0.0  0.0  0.0
#  0.0  0.0  0.0  0.0  3.0  0.0  0.0  0.0  0.0
#  0.0  0.0  0.0  0.0  0.0  3.0  0.0  0.0  0.0
#  0.0  0.0  0.0  0.0  0.0  0.0  3.0  0.0  0.0
#  0.0  0.0  0.0  0.0  0.0  0.0  0.0  3.0  0.0
#  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  3.0
#
# julia> x = merge_systems([sys_ab_different, sys_c_ab, sys_d_bc, sys_e_ad])
# DiscreteSystem{AbstractArray{T,2} where T}(Variable[], Variable[Variable(:a, 3), Variable(:b, 3), Variable(:c, 3), Variable(:d, 3), Variable(:e, 2)], Complex{Float64}[0.0 + 0.0im 0.0 + 0.0im … 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im … 0.0 + 0.0im 0.0 + 0.0im; … ; 0.0 + 0.0im 0.0 + 0.0im … 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im … 0.0 + 0.0im 0.0 + 0.0im])
#
# julia> x = merge_systems(systems(an))
# DiscreteSystem{AbstractArray{T,2} where T}(Variable[], Variable[Variable(:a, 3), Variable(:b, 3), Variable(:c, 3), Variable(:d, 3), Variable(:e, 2)], Complex{Float64}[0.0 + 0.0im 0.0 + 0.0im … 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im … 0.0 + 0.0im 0.0 + 0.0im; … ; 0.0 + 0.0im 0.0 + 0.0im … 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im … 0.0 + 0.0im 0.0 + 0.0im])
#
# julia> x = merge_systems([d_obs])
# DiscreteSystem{Array{Complex{Float64},2}}(Variable[], Variable[Variable(:d, 3)], Complex{Float64}[1.0 + 0.0im 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im 0.0 + 0.0im])
#
# julia> ob = merge_systems([d_obs])
# DiscreteSystem{Array{Complex{Float64},2}}(Variable[], Variable[Variable(:d, 3)], Complex{Float64}[1.0 + 0.0im 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im 0.0 + 0.0im])
#
# julia> x = merge_systems([d_obs])
# DiscreteSystem{Array{Complex{Float64},2}}(Variable[], Variable[Variable(:d, 3)], Complex{Float64}[1.0 + 0.0im 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im 0.0 + 0.0im])
#
# julia> x = merge_systems([sys_ab_different, sys_c_ab, sys_d_bc, sys_e_ad])
# DiscreteSystem{AbstractArray{T,2} where T}(Variable[], Variable[Variable(:a, 3), Variable(:b, 3), Variable(:c, 3), Variable(:d, 3), Variable(:e, 2)], Complex{Float64}[0.0 + 0.0im 0.0 + 0.0im … 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im … 0.0 + 0.0im 0.0 + 0.0im; … ; 0.0 + 0.0im 0.0 + 0.0im … 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im … 0.0 + 0.0im 0.0 + 0.0im])
#
# julia> x.distribution
# 162×162 Array{Complex{Float64},2}:
#  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  …  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im
#  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im     0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im
#  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im     0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im
#  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im     0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im
#  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im     0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im
#     ⋮                                                      ⋮                  ⋱                                                 ⋮
#  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im     0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im
#  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im     0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im
#  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  …  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im
#  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im     0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im
#
# julia> sub_system(x, [var_a, var_b, var_c, var_d, var_e])
# DiscreteSystem{AbstractArray{T,2} where T}(Variable[], Variable[Variable(:a, 3), Variable(:b, 3), Variable(:c, 3), Variable(:d, 3), Variable(:e, 2)], Complex{Float64}[0.0 + 0.0im 0.0 + 0.0im … 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im … 0.0 + 0.0im 0.0 + 0.0im; … ; 0.0 + 0.0im 0.0 + 0.0im … 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im … 0.0 + 0.0im 0.0 + 0.0im])
#
# julia> sub_system(x, [var_a, var_b, var_c, var_d, var_e]).distribution
# 162×162 Array{Complex{Float64},2}:
#  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  …  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im
#  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im     0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im
#  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im     0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im
#  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im     0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im
#  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im     0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im
#     ⋮                                                      ⋮                  ⋱                                                 ⋮
#  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im     0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im
#  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im     0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im
#  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  …  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im
#  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im     0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im
#
# julia>
#
# julia>
#
# julia> x = merge_systems([sys_ab_different, sys_c_ab, sys_d_bc, sys_e_ad])
# DiscreteSystem{AbstractArray{T,2} where T}(Variable[], Variable[Variable(:a, 3), Variable(:b, 3), Variable(:c, 3), Variable(:d, 3), Variable(:e, 2)], Complex{Float64}[0.0 + 0.0im 0.0 + 0.0im … 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im … 0.0 + 0.0im 0.0 + 0.0im; … ; 0.0 + 0.0im 0.0 + 0.0im … 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im … 0.0 + 0.0im 0.0 + 0.0im])
#
# julia> ob = merge_systems([d_obs])
# DiscreteSystem{Array{Complex{Float64},2}}(Variable[], Variable[Variable(:d, 3)], Complex{Float64}[1.0 + 0.0im 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im 0.0 + 0.0im 0.0 + 0.0im])
#
# julia> event(x, sub_system(ob, [var_a, var_b, var_c, var_d, var_e]))
# ERROR: MethodError: no method matching event(::DiscreteSystem{AbstractArray{T,2} where T}, ::DiscreteSystem{Array{Complex{Float64},2}})
# Stacktrace:
#  [1] top-level scope at none:0
#
# julia> event(x.distribution, sub_system(ob, [var_a, var_b, var_c, var_d, var_e]).distribution)
# 162×162 Array{Complex{Float64},2}:
#  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  …  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im
#  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im     0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im
#  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im     0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im
#  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im     0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im
#  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im     0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im
#     ⋮                                                      ⋮                  ⋱                                                 ⋮
#  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im     0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im
#  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im     0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im
#  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  …  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im
#  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im     0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im
#
# julia> res = event(x.distribution, sub_system(ob, [var_a, var_b, var_c, var_d, var_e]).distribution)
# 162×162 Array{Complex{Float64},2}:
#  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  …  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im
#  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im     0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im
#  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im     0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im
#  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im     0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im
#  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im     0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im
#     ⋮                                                      ⋮                  ⋱                                                 ⋮
#  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im     0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im
#  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im     0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im
#  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  …  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im
#  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im     0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im
#
# julia> ptrace(res, [3, 3, 3, 3, 2], [2, 3, 4, 5, 6])
# ERROR: ArgumentError: System index out of range
# Stacktrace:
#  [1] ptrace(::Array{Complex{Float64},2}, ::Array{Int64,1}, ::Array{Int64,1}) at /home/daniel/.julia/packages/QuantumInformation/6qzGE/src/ptrace.jl:22
#  [2] top-level scope at none:0
#
# julia> ptrace(res, [3, 3, 3, 3, 2], [2, 3, 4, 5])
