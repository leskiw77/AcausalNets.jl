using Pkg
Pkg.activate(".")

using AcausalNets

nodes_number = 30
edges_number = 35
an = generate_random_acasual_net(nodes_number, edges_number)
show(an)

Pkg.resolve()
