import numpy as np
from mpi4py import MPI
from Problem import Problem
from Point import Point
from Method import Method
from Neighbour import Neighbour
import BoundaryConditionsUpdate as bc
from PlotTemperature import plot_temperature

# Starting values
ITERATIONS = 100  # Number of iterations for the simulation
n = 6 # number of nodes on the unit segment
step_size = 1/(n-1) # lenght of the grid size, in this case delta_x = delta_y = step_size

# Initialize the MPI communication
comm = MPI.Comm.Clone(MPI.COMM_WORLD)
rank = comm.Get_rank()  # Get the rank of the process

# -----------------------------------------Init problems----------------------------------------------------------------

# Initialize first problem (room1)
if rank == 0:
    # Init points
    # Note: we start numbering points from bottom left corner and go conterclockwise
    p1 = Point(0,0)
    p2 = Point(1,0)
    p3 = Point(1,1)
    p4 = Point(0,1)
    
    boundary1_type = ["Dirichlet"] * n  # Bottom
    boundary2_type = ["Neumann"] * n   # Right
    boundary3_type = ["Dirichlet"] * n  # Top
    boundary4_type = ["Dirichlet"] * n  # Left
    bound_cond_types = [boundary1_type, boundary2_type, boundary3_type, boundary4_type]

    boundary1_value = [15] * n  # Bottom
    boundary2_value = [15] * n  # Right
    boundary3_value = [15] * n  # Top
    boundary4_value = [40] * n  # Left
    bound_cond_values = [boundary1_value, boundary2_value, boundary3_value, boundary4_value]

    problem1 = Problem("room1", p1, p2, p3, p4, step_size, step_size, bound_cond_types, bound_cond_values)


# Initialize second problem (room2)
if rank == 1:
    p1 = Point(1,0)
    p2 = Point(2,0)
    p3 = Point(2,2)
    p4 = Point(1,2)

    boundary1_type = ["Dirichlet"] * (2 * n)
    boundary2_type = ["Dirichlet"] * (2 * n)
    boundary3_type = ["Dirichlet"] * (2 * n)
    boundary4_type = ["Dirichlet"] * (2 * n)
    bound_cond_types = [boundary1_type, boundary2_type, boundary3_type, boundary4_type]

    boundary1_value = [5] * (2 * n)
    boundary2_value = [15] * (2 * n)
    boundary3_value = [40] * (2 * n)
    boundary4_value = [15] * (2 * n)
    bound_cond_values = [boundary1_value, boundary2_value, boundary3_value, boundary4_value]

    problem2 = Problem("room2", p1, p2, p3, p4, step_size, step_size, bound_cond_types, bound_cond_values)



# Initialize third problem (room3)
if rank == 2:
    p1 = Point(2,1)
    p2 = Point(3,1)
    p3 = Point(3,2)
    p4 = Point(2,2)

    boundary1_type = ["Neumann"] * (1 * n)
    boundary2_type = ["Dirichlet"] * (1 * n)
    boundary3_type = ["Dirichlet"] * (1 * n)
    boundary4_type = ["Neumann"] * (1 * n)
    bound_cond_types = [boundary1_type, boundary2_type, boundary3_type, boundary4_type]

    boundary1_value = [15] * (1 * n)
    boundary2_value = [40] * (1 * n)
    boundary3_value = [15] * (1 * n)
    boundary4_value = [15] * (1 * n)
    bound_cond_values = [boundary1_value, boundary2_value, boundary3_value, boundary4_value]

    problem3 = Problem("room3", p1, p2, p3, p4, step_size, step_size, bound_cond_types, bound_cond_values)


# Initialize forth problem (room4)
if rank == 3:
    p1 = Point(2, 0.5)
    p2 = Point(2.5, 0.5)
    p3 = Point(2.5, 1)
    p4 = Point(2, 1)

    boundary1_type = ["Dirichlet"] * (0.5 * n)
    boundary2_type = ["Dirichlet"] * (0.5 * n)
    boundary3_type = ["Neumann"] * (0.5 * n)
    boundary4_type = ["Neumann"] * (0.5 * n)
    bound_cond_types = [boundary1_type, boundary2_type, boundary3_type, boundary4_type]

    boundary1_value = [40] * (0.5 * n)
    boundary2_value = [15] * (0.5 * n)
    boundary3_value = [15] * (0.5 * n)
    boundary4_value = [15] * (0.5 * n)
    bound_cond_values = [boundary1_value, boundary2_value, boundary3_value, boundary4_value]

    problem4 = Problem("room4", p1, p2, p3, p4, step_size, step_size, bound_cond_types, bound_cond_values)




# -----------------------------------------Init neighbours----------------------------------------------------------------
if rank == 0:
    neighbour1 = Neighbour("room2", problem1, None, "Neumann")  # Placeholder
elif rank == 1:
    neighbour2 = Neighbour("room1", None, problem2, "Dirichlet")  # Placeholder
elif rank == 2:
    neighbour3 = Neighbour("room4", problem3, None, "Neumann")  # Placeholder
elif rank == 3:
    neighbour4 = Neighbour("room3", None, problem4, "Dirichlet")  # Placeholder

method = Method()

# ------------------------------------- Iterations -----------------------------------------------------------------------
for i in range(ITERATIONS):
    # ---------------------------- Process 1-------------------------------------------
    if rank == 0:
        v = method.solve(problem1)
        new_types, new_values = bc.calculate_new_condition(v, problem1, neighbour1)
        comm.send((new_types, new_values), dest=1)
        
    # ---------------------------- Process 2-------------------------------------------
    elif rank == 1:
        new_types, new_values = comm.recv(source=0)
        problem2.update_boundary(new_types, new_values, neighbour2)
        v = method.solve(problem2)
        new_types, new_values = bc.calculate_new_condition(v, problem2, neighbour2)
        comm.send((new_types, new_values), dest=0)

    # ---------------------------- Process 2-------------------------------------------
    elif rank == 2:
        new_types, new_values = comm.recv(source=3)
        problem3.update_boundary(new_types, new_values, neighbour3)
        v = method.solve(problem3)
        new_types, new_values = bc.calculate_new_condition(v, problem3, neighbour3)
        comm.send((new_types, new_values), dest=3)

    # ---------------------------- Process 2-------------------------------------------
    elif rank == 3:
        new_types, new_values = comm.recv(source=2)
        problem4.update_boundary(new_types, new_values, neighbour4)
        v = method.solve(problem4)
        new_types, new_values = bc.calculate_new_condition(v, problem4, neighbour4)
        comm.send((new_types, new_values), dest=2)



# Final solve for visualization
if rank == 0:
    final_solution_problem1 = method.solve(problem1)
    comm.send(final_solution_problem1, dest=1)

elif rank == 1:
    final_solution_problem1 = comm.recv(source=0)
    final_solution_problem2 = method.solve(problem2)

    # Plot temperature distributions
    problems = [problem1, problem2]
    solutions = [final_solution_problem1, final_solution_problem2]
    plot_temperature(problems, solutions)

elif rank == 2:
    final_solution_problem1 = comm.recv(source=0)
    final_solution_problem2 = comm.recv(source=1)
    final_solution_problem3 = method.solve(problem3)
    # Plot temperature distributions
    problems = [problem1, problem2, problem3]
    solutions = [final_solution_problem1, final_solution_problem2, final_solution_problem3]
    plot_temperature(problems, solutions)

elif rank == 3:
    final_solution_problem1 = comm.recv(source=0)
    final_solution_problem2 = comm.recv(source=1)
    final_solution_problem3 = comm.recv(source=2)
    final_solution_problem4 = method.solve(problem4)

    # Plot temperature distributions
    problems = [problem1, problem2, problem3, problem4]
    solutions = [final_solution_problem1, final_solution_problem2, final_solution_problem3, final_solution_problem4]
    plot_temperature(problems, solutions)




# Print final boundary conditions if needed
if rank == 0:
    print("Final boundary conditions for problem1:")
    print(problem1.boundary_conditions_types)
    print(problem1.boundary_conditions_values)

elif rank == 1:
    print("Final boundary conditions for problem2:")
    print(problem2.boundary_conditions_types)
    print(problem2.boundary_conditions_values)

elif rank == 2:
    print("Final boundary conditions for problem3:")
    print(problem3.boundary_conditions_types)
    print(problem3.boundary_conditions_values)

elif rank == 3:
    print("Final boundary conditions for problem4:")
    print(problem4.boundary_conditions_types)
    print(problem4.boundary_conditions_values)
