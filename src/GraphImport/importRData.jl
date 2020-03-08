using RData

import AcausalNets.Common:
    Variable

import AcausalNets.Systems:
        DiscreteQuantumSystem,
        InitialNodesSet

import LinearAlgebra:
    Diagonal

function load(grap_rda_path::String)::InitialNodesSet
    graph = RData.load(grap_rda_path)
    return getDiscreateSystems(graph)
end

getVariable(dictVec:: DictoVec) = Variable(Symbol(dictVec["node"]), size(dictVec["prob"])[1])

function getDiscreateSystems(graph)
    nameToVariableDict = Dict(dictVec["node"]=>getVariable(dictVec) for dictVec in graph["bn"].data)
    all_systems = [discreteQuantumSystem(d, nameToVariableDict) for d in graph["bn"].data]
    InitialNodesSet(all_systems)
end

function discreteQuantumSystem(node::DictoVec, nameToVariable::Dict{String,Variable})
      # probably a bug, if node have one parent it is read as string, otherwise its an array
    if(typeof(node["parents"]) === String)
        parents = String[node["parents"]]
    else
        parents = node["parents"]
    end
    parents_variables = [nameToVariable[p] for p in parents]
    current_node_variable = nameToVariable[node["node"]]

    distribution = Diagonal(vec(node["prob"]))

    if(isempty(parents_variables))
        return DiscreteQuantumSystem([current_node_variable], distribution)
    else
        return DiscreteQuantumSystem(parents_variables, [current_node_variable], distribution)
    end
end
