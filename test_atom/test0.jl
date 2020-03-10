using Pkg
# Pkg.instantiate()
Pkg.activate(".")
using AcausalNets
using QuantumInformation
using LightGraphs
using LinearAlgebra

roA = Diagonal([1/3, 1/3, 1/3])

roB = Diagonal([1/3, 1/3, 1/3])
roCwAB = Diagonal([
        0,1/2, 1/2, #A=0, B=0
        0,0,1, #A=0, B=1
        #CHANGED to that ro(C | AB) != ro(C | BA) - Originally BE 0,1,0
        0,1/2,0, #A=0, B=2
        0,0,1, #A=1, B=0
        1/2,0,1/2, #A=1, B=1
        1,0,0, #A=1, B=2
        0,1,0, #A=2, B=0
        1,0,0, #A=2, B=1
        1/2,1/2,0 #A=2, B=2
        ]); #


var_a = Variable(:a, 3)

var_b = Variable(:b, 3)
var_c = Variable(:c, 3)

sys_a = DiscreteQuantumSystem([var_a], roA)

sys_b = DiscreteQuantumSystem([var_b], roB)

sys_c_ab = DiscreteQuantumSystem([var_a, var_b], [var_c], roCwAB)
an = AcausalNet()

push!(an, sys_a)
push!(an, sys_b)
push!(an, sys_c_ab)
show(an)

# the original matrix is ro(C | AB)
# this is ro(C | BA)
roCwBA = permute_systems(roCwAB, [3,3,3], [2, 1, 3])
print(roCwBA == roCwAB)
sys_c_ba = DiscreteQuantumSystem([var_b, var_a], [var_c], roCwBA)

# joint system (a, a2) where a and a2 are independent
# ncategories a = 3
# ncategories a2 = 2

roAA2 = Diagonal([1/6, 1/6, 1/6, 1/6, 1/6, 1/6])
var_a2 = Variable(:a2, 2)
sys_aa2 = DiscreteQuantumSystem([var_a, var_a2], roAA2)

an_2 = AcausalNet()
push!(an_2, sys_aa2)
push!(an_2, sys_b)

# we push a system with roCwBA
# during push, it's parents will be expanded to include var_a2
# (since it's in joint state with var_a)
# and the parents will be permuted to bein the same order
# as the BayesNet's order of variables
# before ppermutation their order is: a2, b, a
# after permutation their order is: a, a2, b
push!(an_2, sys_c_ba)
show(an_2)

is_parent(sys_a, sys_c_ba)

is_parent(sys_aa2, sys_c_ba)

c_system = an_2.systems[variable_to_node(var_c, an_2)]
roC_AA2B = c_system.distribution

traced_out = ptrace(roC_AA2B, [3,2,3,3], [2]) / var_a2.ncategories

traced_out == roCwAB
