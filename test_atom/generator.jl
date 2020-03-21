using Pkg
Pkg.activate(".")

using AcausalNets

nodes_number = 35
edges_number = 39
an = generate_random_acasual_net(nodes_number, edges_number)
show(an)


observations = Evidence{Matrix{Complex{Float64}}}[]

an

to_infer = variables(systems(an)[1])

inferred_system, debug = infer_join_tree(an, to_infer, observations)
inferred_system_naive, debug_naive = infer_naive(an, to_infer, observations)
inferred_system_belief, debug_belief = infer_belief(an, to_infer, observations)

real(distribution(inferred_system))
real(distribution(inferred_system_naive))
real(distribution(inferred_system_belief))

(observations_jt,propagated_jt,jt,) = debug
show(jt)

jt

using LightGraphs

is_connected(an.dag)
is_connected(jt.graph)
