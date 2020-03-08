using Pkg

Pkg.activate(".")

using AcausalNets
using QuantumInformation
using LightGraphs
using LinearAlgebra


grap_rda_path = "/home/jarema/Documents/Codes/Julia/Graph/data/alarm.rda"
grap_rda_path = "/home/jarema/Documents/Codes/Julia/Graph/data/asia.rda"

nodes_set = load(grap_rda_path)

new_distribution = Diagonal([0.5, 0.5, 0.3, 0.7])

exchange_distribution(nodes_set, "tub", new_distribution)

an = AcausalNet()

pushMany(nodes_set, an)

show(an)

to_infer = variables(get_system_by_name(nodes_set, "either"))
observations = Evidence{Matrix{Complex{Float64}}}[]

inferred_system, debug = infer_join_tree(an, to_infer, observations)
real(distribution(inferred_system))

inferred_system_naive, debug_naive = infer_naive(an, to_infer, observations);
real(distribution(inferred_system_naive))

inferred_system_belief, debug_beleif = infer_belief(an, to_infer, observations);
real(distribution(inferred_system_belief))

(observations_jt,propagated_jt,jt,) = debug
show(jt)
