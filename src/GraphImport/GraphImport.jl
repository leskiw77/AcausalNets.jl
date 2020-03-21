module GraphImport
    include("importRData.jl")
    include("randomNetGenerator.jl")

    export
        load, generate_random_acasual_net
end
