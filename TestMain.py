import numpy as np
from Problem import Problem
from Point import Point
from Method import Method
from Neighbour import Neighbour
import BoundaryConditionsUpdate as bc
from PlotTemperature import plot_temperature, solution_vector_to_matrix

#-------------------------------------------Init method-----------------------------------------------------------------
method = Method()
omega = 0.8 # relaxation parameter
# -----------------------------------------Init problems----------------------------------------------------------------
n = 99 # number of nodes on the unit segment
step_size = 1/(n-1) # lenght of the grid size, in this case delta_x = delta_y = step_size

# Initialize first problem
# Init points
# Note: we start numbering points from bottom left corner and go conterclockwise
p1 = Point(0,0)
p2 = Point(1,0)
p3 = Point(1,1)
p4 = Point(0,1)

# Init boundary conditions types ("Dirichlet"/"Neumann")
# Note: we start numbering boundaries from the bottom one and go conterclockwise
boundary1_type = ["Dirichlet"]*n # bottom
boundary2_type = ["Dirichlet"]*n # right
boundary3_type = ["Dirichlet"]*n # top
boundary4_type = ["Dirichlet"]*n # left
bound_cond_types = [boundary1_type,boundary2_type,boundary3_type,boundary4_type]

# Init boundary conditions values
boundary1_value = [15]*n # bottom
boundary2_value = [15]*n # right
boundary3_value = [30]*n # top
boundary4_value = [25]*n # left
bound_cond_values = [boundary1_value,boundary2_value,boundary3_value,boundary4_value]

# Iniit problem 1 with id "room1"
problem1 = Problem("room1",p1,p2,p3,p4,step_size,step_size,bound_cond_types,bound_cond_values)


# Initialize second problem
# Init points
p1 = Point(1,0)
p2 = Point(3,0)
p3 = Point(3,2)
p4 = Point(1,2)

# Init boundary conditions types ("Dirichlet"/"Neumann")
boundary1_type = ["Dirichlet"]*(2*n)
boundary2_type = ["Dirichlet"]*(2*n)
boundary3_type = ["Dirichlet"]*(2*n)
boundary4_type = ["Dirichlet"]*(2*n)
bound_cond_types = [boundary1_type,boundary2_type,boundary3_type,boundary4_type]

# Init boundary conditions values
boundary1_value = [15]*(2*n)
boundary2_value = [40]*(2*n)
boundary3_value = [15]*(2*n)
boundary4_value = [15]*(2*n)
bound_cond_values = [boundary1_value,boundary2_value,boundary3_value,boundary4_value]

# Iniit problem 2 with id "room2"
problem2 = Problem("room2",p1,p2,p3,p4,step_size,step_size,bound_cond_types,bound_cond_values)

# -----------------------------------------Init neighbours----------------------------------------------------------------
# Class Neighbour stores the info about intersection (common boundary) between the two problems.

# Init neighbour for problem1 
# We pass id of the room problem1 is connected to = "room2"; main_problem - problem1; neighbouring problem - problem2; 
# "Neumann" - condition type we will pass from main_problem to neighbouring problem
neighbour1 = Neighbour("room2",problem1,problem2,"Neumann")


# Init neighbour for problem1 
neighbour2 = Neighbour("room1",problem2,problem1,"Dirichlet")
# ------------------------------------- Iterations -----------------------------------------------------------------------
iterations = 10  # Set the number of iterations (adjust as needed)

for i in range(iterations):
    # ---------------------------- Process 1-------------------------------------------
    # Solve for v in room 1 (problem1)
    # Saving previous solution for relxation step
    if(i!=0):
        v1_prev = v1
        
    v1 = method.solve(problem1)

    if(i!=0): # do relaxation step
        v1_relax = v1*omega + (1-omega)*v1_prev
        new_types, new_values = bc.calculate_new_condition(v1_relax, problem1, neighbour1)
    else: # no previous solution on first iteration
        new_types, new_values = bc.calculate_new_condition(v1, problem1, neighbour1)
    
    # send new_types, new_values to process 2
    #----------------------------------------------------------------------------------
    
    # ---------------------------- Process 2-------------------------------------------
    # Recieve new_types, new_values from process 1
    
    
    # Update the boundary of problem2 (neighbor2 defines which boundary to update)
    problem2.update_boundary(new_types, new_values, neighbour2)
    
    # Save previous value for relaxation step
    if(i!=0):
        v2_prev = v2
    # Solve for v in room 2 (problem2)
    v2 = method.solve(problem2)
    
    if(i!=0): # do relaxation step
        v2_relax = v2*omega + (1-omega)*v2_prev
        # Calculate new boundary condition for problem1 based on the solution of problem2
        new_types, new_values = bc.calculate_new_condition(v2_relax, problem2, neighbour2)
    else: # no previous value for relaxation step yet
        new_types, new_values = bc.calculate_new_condition(v2, problem2, neighbour2)
    
    # send new_types, new_values to process 1
    # -----------------------------------------------------------------------------

    # ---------------------------- Process 1 -------------------------------------------
    # Receive new_types, new_values from process 2
    
    # Update the boundary of problem1 (neighbor1 defines which boundary to update)
    problem1.update_boundary(new_types, new_values, neighbour1)
    # ---------------------------- Process 1 -------------------------------------------

v1 = method.solve(problem1)
v2 = method.solve(problem2)

# # After loop ends, print final boundary conditions
# print("Final boundary conditions for problem1:")
# print(problem1.boundary_conditions_types)
# print(problem1.boundary_conditions_values)

# print("Final boundary conditions for problem2:")
# print(problem2.boundary_conditions_types)
# print(problem2.boundary_conditions_values)

# Plot the temperature distribution
problems = [problem1, problem2]
solutions = [v1, v2]
plot_temperature(problems, solutions)
