using LightGraphs, MetaGraphs
using Random

import Distributions:
    Normal

import AcausalNets.Common:
    Variable, ncategories

import AcausalNets.Systems:
        DiscreteQuantumSystem,
        InitialNodesSet

import AcausalNets.Structures:
        AcausalNet, pushMany

import LinearAlgebra:
    Diagonal, norm, diagm

function get_normal_distribution_rand(s, e)
    dif = e - s
    distribution = Normal(0, dif)
    random_value = Int(round(abs(rand(distribution))))
    s + random_value % (dif + 1)
end

function generate_random_dag(nodes_number, edges_number, start, stop)
    nodes_number <= edges_number || error("Number of edges cannot be smaller than nodes number")
    start <= stop || error("Max is smaller than min")
    g = MetaDiGraph(nodes_number)
    for i in 1:nodes_number
        set_prop!(g, i, :variable, Variable(Symbol("symbol_$(i)"), rand(start:stop)))
    end

    function add_unique_edge()
        i = 0
        max_iterations = 100
        while i <= 5
            from_node = rand(1:nodes_number-1)
            to_node = get_normal_distribution_rand(from_node+1, nodes_number)
            if !has_edge(g, from_node, to_node)
                add_edge!(g, from_node, to_node)
                return
            end
            i += 1
        end
        error("Too many edges")
    end

    # each node is connected at least to one element
    for i in 1:(nodes_number-1)
        points_to = get_normal_distribution_rand((i + 1), nodes_number)
        add_edge!(g, i, points_to)
    end

    for _ in 0:(edges_number - nodes_number)
        add_unique_edge()
    end
    g
end

function get_quantum_system(g, i)
    current_node_variable = get_prop(g, i,:variable)
    parent_variables = [get_prop(g, neighbor_index,:variable) for neighbor_index in inneighbors(g, i)]
    categories = isempty(parent_variables) ? 1 : ncategories(parent_variables)

    function get_random_probability_array(categories)
        x = rand(categories)
        x / norm(x, 1)
    end

    diag = vcat([get_random_probability_array(ncategories(current_node_variable)) for _ in (1:categories)]...)
    ro = diagm(0 => diag)

    isempty(parent_variables) ? DiscreteQuantumSystem([current_node_variable], ro) : DiscreteQuantumSystem(parent_variables, [current_node_variable], ro)
end

function generate_random_acasual_net(nodes_number, edges_number, min_vategories=2, max_categories=3)
    g = generate_random_dag(nodes_number, edges_number, min_vategories, max_categories)
    nodes_set = InitialNodesSet([get_quantum_system(g, i) for i in 1:nodes_number])

    an = AcausalNet()
    pushMany(nodes_set, an)
    an
end
