import numpy as np
import matplotlib.pyplot as plt

import numpy as np

def plot_temperature(problems,solutions):
    # Assemble the apartment temperature grid
    apartment_grid = assemble_temperature_matrix(problems,solutions)

    # Create meshgrid for plotting
    x = np.arange(apartment_grid.shape[1])
    y = np.arange(apartment_grid.shape[0])
    X, Y = np.meshgrid(x, y)

    # Plot the temperature distribution
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(X, Y, apartment_grid, shading='auto', cmap='coolwarm')
    #plt.pcolormesh(X, Y, apartment_grid, shading='auto', cmap='coolwarm',vmin = 15,vmax = 40)
    #m.pcolormesh(x, y, efdata, latlon=True, vmin=-1, vmax=1)
    plt.colorbar(label="Temperature")
    plt.title("Temperature Distribution in the Apartment")
    plt.show()


def assemble_temperature_matrix(problems, solutions):
    """
    Assemble a new matrix representing the temperature distribution of an n-room apartment.
    
    Parameters:
    room_coords: List of tuples [(x1, y1, x2, y2), ...] - coordinates of room corners
    temp_matrices: List of 2D arrays [matrix1, matrix2, ...] - temperature distributions in each room
    delta_x: Grid size in the x direction
    delta_y: Grid size in the y direction
    
    Returns:
    apartment_grid: 2D array with temperature values inside rooms and NaNs outside
    """
    # Read stepsizes
    delta_x = problems[0].delta_x
    delta_y = problems[0].delta_y
    
    # Read coordinates of the rooms
    room_coords = []
    for pr in problems:
        Ax, Ay = pr.A.x, pr.A.y  # Bottom-left corner coordinates
        Cx, Cy = pr.C.x, pr.C.y  # Upper-right corner coordinates
        room_coords.append((Ax, Ay, Cx, Cy))
    
    # Convert solutions to matrices
    temp_matrices = []
    for i in range(len(solutions)):
        s = solutions[i]
        n = round( problems[i].A.distance(problems[i].B)/ problems[i].delta_x ) # (n+1) el in the row 
        m = round( problems[i].A.distance(problems[i].D)/ problems[i].delta_y ) # (m+1) el in a column
        # print("n=",n,"m=",m)
        temp_matrices.append(solution_vector_to_matrix(s,n,m))
    
    # Find the bounds of the entire apartment by taking min/max of room coordinates
    all_x = [x for coords in room_coords for x in (coords[0], coords[2])]
    all_y = [y for coords in room_coords for y in (coords[1], coords[3])]
    
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    
    # Calculate the number of grid points needed to cover the entire apartment
    grid_x_size = round((max_x - min_x) / delta_x) + 1  # +1 to include the last point
    grid_y_size = round((max_y - min_y) / delta_y) + 1  # +1 to include the last point
    
    #print("x size",grid_x_size,"y_size",grid_y_size)
    
    # Create an empty grid filled with NaN values to represent the entire apartment
    apartment_grid = np.full((grid_y_size, grid_x_size), np.nan)
    
    # Map each room's temperature matrix to the apartment grid
    for i, ((x1, y1, x2, y2), temp_matrix) in enumerate(zip(room_coords, temp_matrices)):
        # Calculate the start and end indices for this room in the apartment grid
        start_x = round((x1 - min_x) / delta_x)
        end_x = round((x2 - min_x) / delta_x)
        start_y = round((y1 - min_y) / delta_y)
        end_y = round((y2 - min_y) / delta_y)
        
        
        # Ensure the temperature matrix size matches the expected size
        grid_room_height = end_y - start_y + 1
        grid_room_width = end_x - start_x + 1
    
    
        if temp_matrix.shape != (grid_room_height, grid_room_width):
            # print(temp_matrix.shape)
            # print(grid_room_height)
            # print(grid_room_width)
            raise ValueError(f"Temperature matrix for room {i} does not match the expected size")
        
        # Place the room's temperature matrix inside the corresponding area of the apartment grid
        apartment_grid[start_y:(end_y+1), start_x:(end_x+1)] = temp_matrix
    
    return apartment_grid


def solution_vector_to_matrix(s_vector:np.array,n:int,m:int):
    """
    Converts vector of solutions into matrix
    Args:
        s_vector (np.array): solution vector
        n (int): n+1 - number of elements oon the grid in x direction
        m (int): m+1 - number of elements on the grid in y direction
    """
    s_matrix = np.zeros((m+1, n+1))
    # print(s_matrix.shape)
    for k in range(len(s_vector)):
        i = k//(n+1)
        j = k%(n+1)
        # print("i = ",i,"j = ",j,"n+1-i= ",(n+1)-i)
        s_matrix[i,j] = s_vector[k]
    
    return s_matrix


