#=
acausal_structures:
- Author: marcin
- Date: 2018-05-10
=#
module Structures
    include("bayes_net.jl")
    include("acausal_net.jl")

    export
        AcausalNet,
        systems,
        variables,
        variables_names,
        variable_to_node,
        system_to_node,
        pushMany

end #module
