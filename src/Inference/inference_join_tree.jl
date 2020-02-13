#=
huang_inference:
- Julia version: 0.7
- Author: marcin
- Date: 2018-09-11
=#

using LightGraphs

import AcausalNets.Common:
    Variable,
    ncategories

import AcausalNets.Systems:
    DiscreteSystem,
    variables,
    distribution,
    permute_system,
    ncategories,
    reduce_distribution,
    multiply_star,
    divide_star,
    identity_distribution,
    multiply_kron,
    permute_distribution,
    sub_system

import AcausalNets.Structures:
    DiscreteBayesNet

import AcausalNets.Inference:
    JoinTree,
    normalize,
    ParentCliquesDict,
    parent_cliques_dict,
    moral_graph,
    enforce_clique,
    triangulate,
    apply_observations,
    shallowcopy,
    unpropagated_join_tree


"""
Implementation of inference as described in
http://pages.cs.wisc.edu/~dpage/ijar95.pdf

Turns out it doesn't work on quantum networks (only on classical ones), but yields identical results
to inference through belief propagation (which also doesn't work on quantum networks yet).

The difference between this and Belief Propagation is that in this implementation, *all* vertices of join tree
update their states, whereas in belief propagation, only the vertex we're inferring gathers information and updates
its state. Therefore, in Belief Propagation, 2 times less messages are sent.

Apart from building join tree, this is probably worthless and may be thrown out.
"""
function infer_join_tree(
        dbn::DiscreteBayesNet{S},
        vars_to_infer::Vector{Variable},
        observations::Vector{E} = E[]
        ) where {
            D1,
            D2 <: D1,
            S <: DiscreteSystem{D1},
            E <: Evidence{D2}
        }
    length(vars_to_infer) > 0 || error("At least one variable to infer must be specified!")
    observations_jt = unpropagated_join_tree(
                        dbn,
                        vars_to_infer,
                        observations
                    )
    propagated_jt = global_propagation(observations_jt, vars_to_infer)
    jt = normalize(propagated_jt)
    inferred_cluster = first([
            sys
            for (i, sys) in jt.edge_to_sepset
            if all([
                    v in variables(sys)
                    for v in vars_to_infer
                    ])
        ])

    inference_result = sub_system(inferred_cluster, vars_to_infer)
    intermediate_elements = (
        observations_jt,
        propagated_jt,
        jt,
    )

    inference_result, intermediate_elements
end

"""
Message passing, as described in http://pages.cs.wisc.edu/~dpage/ijar95.pdf
"""
function single_message_pass(from_ind::Int, to_ind::Int, jt::JoinTree{S}, vars_to_infer::Vector{Variable}) where S
    if (from_ind, to_ind) in edges(jt.graph)
        jt = shallowcopy(jt)
        cluster_from = jt.vertex_to_cluster[from_ind]
        cluster_to = jt.vertex_to_cluster[to_ind]
        edge_set = Set([from_ind, to_ind])
        sepset = jt.edge_to_sepset[edge_set]

        old_sepset = sub_system(sepset, variables(cluster_to))
        new_sepset = sub_system(cluster_from, variables(cluster_to))
        jt.edge_to_sepset[edge_set] = sub_system(new_sepset, variables(sepset))

        message_ordered = distribution(old_sepset)
        to_distribution = distribution(cluster_to)

        new_to_distribution = multiply_star(
            divide_star(to_distribution, distribution(old_sepset)),
            distribution(new_sepset),
        )


        new_cluster_to = S(variables(cluster_to), new_to_distribution)
        jt.vertex_to_cluster[to_ind] = new_cluster_to
    end

    jt2 = normalize(jt)
    inferred_cluster = first([
            sys
            for (i, sys) in jt2.edge_to_sepset
            if all([
                    v in variables(sys)
                    for v in vars_to_infer
                    ])
        ])

    inference_result = sub_system(inferred_cluster, vars_to_infer)

    println("print start")

    show(Base.stdout, "text/plain", real(distribution(inference_result)))
    println("\nprint end")

    return jt
end

"""
Collect-evidence stage
"""
function collect_evidence(cluster_ind::Int, cluster_marks::Vector{Bool}, jt::JoinTree, vars_to_infer::Vector{Variable})
    jt = shallowcopy(jt)
    cluster_marks[cluster_ind] = false
    for neighbor in neighbors(jt.graph, cluster_ind)
        if cluster_marks[neighbor]
            jt, cluster_marks = collect_evidence(neighbor, cluster_marks, jt, vars_to_infer)
            jt = single_message_pass(neighbor, cluster_ind, jt, vars_to_infer)
        end

    end
    jt, cluster_marks

end

"""
Distribute-evidence stage
"""
function distribute_evidence(cluster_ind::Int, cluster_marks::Vector{Bool}, jt::JoinTree, vars_to_infer::Vector{Variable})
    jt = shallowcopy(jt)
    cluster_marks[cluster_ind] = false
    for neighbor in neighbors(jt.graph, cluster_ind)
        if cluster_marks[neighbor]
            jt = single_message_pass(cluster_ind, neighbor, jt, vars_to_infer)
            # pass a message from cluster_ind to neighbor
        end
    end
    for neighbor in neighbors(jt.graph, cluster_ind)
        if cluster_marks[neighbor]
            jt, cluster_mars = distribute_evidence(neighbor, cluster_marks, jt, vars_to_infer)
        end
    end
    jt, cluster_marks
end

"""
Propagation of messages
"""
function global_propagation(jt::JoinTree, vars_to_infer::Vector{Variable}, start_ind=1)
    jt = shallowcopy(jt)
    println("start")

    cluster_marks = [true for k in keys(jt.vertex_to_cluster)]
    jt, cluster_marks = collect_evidence(start_ind, cluster_marks, jt, vars_to_infer)
    cluster_marks = [true for k in keys(jt.vertex_to_cluster)]
    jt, cluster_marks = distribute_evidence(start_ind, cluster_marks, jt, vars_to_infer)
    return jt
end
