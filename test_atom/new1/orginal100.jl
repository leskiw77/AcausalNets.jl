using Pkg
# Pkg.instantiate()
Pkg.activate(".")
using AcausalNets
using QuantumInformation
using LightGraphs
using LinearAlgebra

var_a = Variable(:a, 3) # the placement of the prize
var_b = Variable(:b, 3) # the initial choice of the door by the player
var_c = Variable(:c, 2) # the door opened by the host
var_d = Variable(:d, 2) # the door opened by the player
var_e = Variable(:e, 2) # whether the player has won (0 = lost, 1 = won)
var_f = Variable(:f, 2)
var_g = Variable(:g, 2)
var_h = Variable(:h, 2)

# rules of the host
roA = diagm(0 =>[
        1/3,1/3,1/3, #A=0
        ]); #

sys_a = DiscreteQuantumSystem([var_a], roA)
# rules of the host
roCwA = diagm(0 =>[
        1/2,1/2, #A=0
        1/4,3/4, #A=1
        1,0, #A=2
        ]); #

sys_c_a = DiscreteQuantumSystem([var_a], [var_c], roCwA)

# this matrix represents the optimal strategy in classical case
# - always changing the choice
roDwA = diagm(0 => [
        1/2,1/2, # A=0 (impossible situation)
        2/3,1/3, # A=1
        0,1, # A=2
        ])

sys_d_a = DiscreteQuantumSystem([var_a], [var_d], roDwA)

### NEW
roEwCD = diagm(0 => [
        0,1, # C=0, D=0 (player wins)
        1/4,3/4,# C=0, D=1
        1/2,1/2, # C=1, D=0 (player wins)
        1,0, # C=1, D=1
        ])
sys_e_cd = DiscreteQuantumSystem([var_c, var_d], [var_e], roEwCD);

roFwCD = diagm(0 => [
        0,1, # C=0, D=0 (player wins)
        1,0, #C=0, D=1
        1,0, #C=1, D=0
        1,0, #C=0, D=1
        ])
sys_f_cd = DiscreteQuantumSystem([var_c, var_d], [var_f], roFwCD);

roGwCE = diagm(0 => [
        1,0, # D=0, E=0
        1/2,1/2, # D=0, E=1
        1/3,2/3, # D=1, E=0
        2/3,1/3, # D=1, E=1
        ])
sys_g_ce = DiscreteQuantumSystem([var_c, var_e], [var_g], roGwCE);

roHwBFG = diagm(0 => [
        1,0, # B=0, F=0, G=0
        1/2,1/2, # B=0, F=0, G=1
        1/3,2/3, # B=0, F=1, G=0
        2/3,1/3, # B=0, F=1, G=1
        0,1, # B=1, F=0, G=0
        1/3,2/3, # B=1, F=0, G=1
        1/2,1/2, # B=1, F=1, G=0
        0,1, # B=1, F=1, G=1
        1,0, # B=2, F=0, G=0
        1,0, # B=2, F=0, G=1
        1/4,3/4, # B=2, F=1, G=0
        2/3,1/3, # B=2, F=1, G=1
        ])
sys_h_bfg = DiscreteQuantumSystem([var_b, var_f, var_g], [var_h], roHwBFG);


###

roAB_different = 1/6*(ket(2,9)+ket(4,9))*(bra(2,9)+bra(4,9))+1/6*(ket(3,9)+ket(7,9))*(bra(3,9)+bra(7,9))+1/6*(ket(6,9)+ket(8,9))*(bra(6,9)+bra(8,9))
roAB_test = 1/6*(ket(1,9)+ket(2,9))*(bra(1,9)+bra(2,9))+1/6*(ket(5,9)+ket(6,9))*(bra(5,9)+bra(6,9))+1/6*(ket(9,9)+ket(7,9))*(bra(9,9)+bra(7,9))
roAB_same = 1/3 * (ket(1,9)+ket(5,9)+ket(9,9))*(bra(1,9)+bra(5,9)+bra(9,9))
sys_ab_same = DiscreteQuantumSystem([var_a, var_b], roAB_test)

sys_ab = DiscreteQuantumSystem([var_a, var_b], roAB_different)
an_different = AcausalNet()

push!(an_different, sys_ab)
push!(an_different, sys_c_a)
push!(an_different, sys_d_a)
push!(an_different, sys_e_cd)
push!(an_different, sys_f_cd)
push!(an_different, sys_g_ce)
push!(an_different, sys_h_bfg)

show(an_different)
an = an_different
a = 3
b = 1
c = 1
d = 1
e = 1
f = 1
g = 1
h = 1
a_obs = Evidence([var_a], ketbra(a,a,3))
b_obs = Evidence([var_b], ketbra(b,b,3))
c_obs = Evidence([var_c], ketbra(c,c,2))
d_obs = Evidence([var_d], ketbra(d,d,2))
e_obs = Evidence([var_e], ketbra(e,e,2))
f_obs = Evidence([var_f], ketbra(f,f,2))
g_obs = Evidence([var_g], ketbra(g,g,2))
h_obs = Evidence([var_h], ketbra(h,h,2))
observations = Evidence{Matrix{Complex{Float64}}}[
     # a_obs,
     c_obs,
        # d_obs,
   # h_obs,
#     e_obs
    ]
to_infer = [var_g]
inferred_system, debug = infer_join_tree(an, to_infer, observations)
inferred_system_naive, debug_naive = infer_naive(an, to_infer, observations)
inferred_system_belief, debug_belief = infer_belief(an, to_infer, observations)


real(distribution(inferred_system))

real(distribution(inferred_system_naive))

real(distribution(inferred_system_belief))

show(debug[1])

show(moral_graph(an))


function add_children_of_quantum(dbn, parent::Variable)
        group = Set{Int}()
        for sys in systems(dbn)
                if parent in sys.parents
                        children_ind = [variable_to_node(x, dbn) for x in sys.variables]
                        for child_ind in children_ind
                            # println()
                            # println(parent)
                            # println(sys.variables)
                            push!(group, child_ind)
                            # add_edge!(mg, child_ind, parent_ind)
                        end
                end
        end
        group
end


dbn = an
mg_enforced = moral_graph(an)

for sys in systems(dbn)
    if(!isdiag(sys.distribution))
        clique = Set{Int}()
        for v in sys.variables
                parent_ind = variable_to_node(v, dbn)
                children_inds = add_children_of_quantum(dbn, v)
                push!(children_inds, parent_ind)
                clique = union(clique, children_inds)
        end

        for c1 in clique
            for c2 in clique
                if c1 > c2
                    println(c1, " ", c2)
                    add_edge!(mg_enforced, c1, c2)
                end
            end
        end

    end
end
