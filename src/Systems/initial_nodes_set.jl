using AcausalNets.Common

import AcausalNets.Systems:
    DiscreteSystem

struct InitialNodesSet{D}
	discrete_system_vec::Vector{DiscreteSystem{D}}
end

get_variable(ds::DiscreteSystem) = variables(ds)[1]
get_variable_name(ds::DiscreteSystem) = string(variables(ds)[1].name)

function get_system_by_name(nodes_set::InitialNodesSet, name::String)
	index = findfirst(ds -> get_variable_name(ds) === name, nodes_set.discrete_system_vec)
	index != nothing || error("No node with name " * name)
	nodes_set.discrete_system_vec[index]
end

function exchange_distribution(nodes_set::InitialNodesSet, name::String, distribution::D) where D
	index = findfirst(ds -> get_variable_name(ds) === name, nodes_set.discrete_system_vec)
	index != nothing || error("No node with name " * name)
	ds = nodes_set.discrete_system_vec[index]

    deleteat!(nodes_set.discrete_system_vec, index)

    new_ds = DiscreteQuantumSystem(ds.parents, ds.variables, distribution)
	push!(nodes_set.discrete_system_vec, new_ds)
end
