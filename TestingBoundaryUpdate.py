import numpy as np
from Problem import Problem
from Point import Point
from Method import Method
import BoundaryConditionsUpdate as bc

# Initialize first problem
p1 = Point(0,0)
p2 = Point(1,0)
p3 = Point(1,1)
p4 = Point(0,1)

boundary1_type = ["Dirichlet"]*3
boundary2_type = ["Dirichlet"]*3
boundary3_type = ["Dirichlet"]*3
boundary4_type = ["Dirichlet"]*3
bound_cond_types = [boundary1_type,boundary2_type,boundary3_type,boundary4_type]

boundary1_value = [15]*3
boundary2_value = [15]*3
boundary3_value = [15]*3
boundary4_value = [15]*3
bound_cond_values = [boundary1_value,boundary2_value,boundary3_value,boundary4_value]
step_size = 1/2
     
problem1 = Problem(p1,p2,p3,p4,step_size,step_size,bound_cond_types,bound_cond_values)

v = [0,1,2,3,4,5,6,7,8]
bound_info=(2,0,2)

new_types,new_values = bc.calculate_Neumann_cond(v,problem1,bound_info)
print(new_values)


p1 = Point(1,0)
p2 = Point(3,0)
p3 = Point(3,2)
p4 = Point(1,2)

boundary1_type = ["Dirichlet"]*5
boundary2_type = ["Dirichlet"]*5
boundary3_type = ["Dirichlet"]*5
boundary4_type = ["Dirichlet"]*5
bound_cond_types = [boundary1_type,boundary2_type,boundary3_type,boundary4_type]

boundary1_value = [15]*5
boundary2_value = [15]*5
boundary3_value = [15]*5
boundary4_value = [15]*5
bound_cond_values = [boundary1_value,boundary2_value,boundary3_value,boundary4_value]
step_size = 1/2

problem2 = Problem(p1,p2,p3,p4,step_size,step_size,bound_cond_types,bound_cond_values)
bound_info = (4,0,2)

print("Before:")
print(problem2.boundary_conditions_types[3])
print(problem2.boundary_conditions_values[3])

problem2.update_boundary(new_types,new_values,bound_info)

print("After:")
print(problem2.boundary_conditions_types[3])
print(problem2.boundary_conditions_values[3])

# # Initialisze second problem
# p1 = Point(2,4)
# p2 = Point(2,7)
# p3 = Point(3,7)
# p4 = Point(3,4)

# boundary1 = [("Dirichlet",15,3)]
# boundary2 = [("Dirichlet",15,3)]
# boundary3 = [("Dirichlet",15,3)]
# boundary4 = [("Dirichlet",15,3)]
# bound_cond = [boundary1,boundary2,boundary3,boundary4]
# step_size = 1/2
     
# problem2 = Problem(p1,p2,p3,p4,step_size,step_size,bound_cond)

# method1 = Method(problem1)
# method2 = Method(problem2)

# problems = [problem1,problem2]
# methods = [method1,method2]


# # bound_info[0] - problem1, bound_info[1] - problem2
# bound_info=[(2,2,6),(4,0,4)]


# v = method2.solve(problem2)






# new_values = bc.calculate_Dirichlet_cond(v,problem,bound_info)
# print(new_values)

# new_values = bc.calculate_Neumann_cond(v,problem,bound_info)
# print(new_values)



