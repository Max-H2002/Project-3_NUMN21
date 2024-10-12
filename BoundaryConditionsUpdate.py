import numpy as np
import math
from Point import Point
from Problem import Problem
from Neighbour import Neighbour

def calculate_new_condition(v: np.array, 
                         problem : Problem, 
                         neighbour: Neighbour):
    
    if(neighbour.cond_to_pass == "Dirichlet"):
        new_bound_cond_types, new_bound_cond_values = calculate_Dirichlet_cond(v,problem,neighbour)
    elif(neighbour.cond_to_pass == "Neumann"):
        new_bound_cond_types, new_bound_cond_values = calculate_Neumann_cond(v,problem,neighbour)
    else:
        raise ValueError("Condition to pass to the neighbouring class set incorrectly!")

    return new_bound_cond_types, new_bound_cond_values

def calculate_Dirichlet_cond(v: np.array, 
                             problem : Problem, 
                             neighbour: Neighbour):
     """
     Args:
         v (np.array): vector, that corresponds to a matrix of u - values on the grid
         problem (Problem): problem, that v solves
         bound_info (tuple): information about the bound, from which we have to calculate boundaary conditions 
         in the format of tuple (bound_index, i_start,i_end), where 
         bound_idx - index of the boundary, from which we need to recalculate boundary conditions (1 - bottom, 2 - right, 3 - top, 4 - left)
         i_start - index of the boundary grid that coressponds to the start of the subsegment we need to recaclculate the boundary condition from
         i_end - index of the boundary grid that coressponds to the end of the subsegment we need to recaclculate the boundary condition from

     Returns:
         list: list of tuples (string, float,float) - new Dirichlet boundary condition
     """
     # Get data from tuple
    #  bound_index = bound_info[0]
    #  i_start = bound_info[1]
    #  i_end = bound_info[2]
    
     # sizes of the grid
     n = int( problem.A.distance(problem.B)/ problem.delta_x ) # (n+1) el in the row 
     m = int( problem.A.distance(problem.D)/ problem.delta_y ) # (m+1) el in a column
     
     new_bound_cond_types = ["Dirichlet"]*(neighbour.i_end-neighbour.i_start +1)
     new_bound_cond_values = []
     
     # Bottom Boundary
     if(neighbour.bound_index == 1) :
         bottom_row = v[0:n+1]
         print(bottom_row)
         values = bottom_row[neighbour.i_start:neighbour.i_end+1]
     # Top boundary
     elif(neighbour.bound_index == 3) :
         top_row = v[-(n+1):]
         values = top_row[neighbour.i_start:neighbour.i_end+1]
     # Right boundary
     elif (neighbour.bound_index ==2):
          right_col = []
          j = n
          for i in range(0,m+1):
              right_col.append(v[j + (n+1)*i])
          values = right_col[neighbour.i_start:neighbour.i_end+1]
    # Left boundary
     elif (neighbour.bound_index == 4):
          left_col = []
          j = 0
          for i in range(0,m+1):
              left_col.append(v[j + (n+1)*i]) 
          values = left_col[neighbour.i_start:neighbour.i_end+1] 
     else:
         raise ValueError("Wrong boundary index, should be inthenger in range(0,4)")   
         
     for x in values:
        new_bound_cond_values.append(x)
        
     return new_bound_cond_types, new_bound_cond_values
 
def calculate_Neumann_cond(v: np.array, 
                             problem : Problem, 
                             neighbour: Neighbour):
     """
     Args:
         v (np.array): vector, that corresponds to a matrix of u - values on the grid
         problem (Problem): problem, that v solves
         neighbour (Neighbour): neighbouring problem info
     Returns:
         list: list of tuples (string, float,float) - new Neumann boundary condition
     """
     # Get data from tuple
    #  bound_index = bound_info[0]
    #  i_start = bound_info[1]
    #  i_end = bound_info[2]
    
     # sizes of the grid
     n = int( problem.A.distance(problem.B)/ problem.delta_x ) # (n+1) el in the row 
     m = int( problem.A.distance(problem.D)/ problem.delta_y ) # (m+1) el in a column
     
     new_bound_cond_types = ["Neumann"]*(neighbour.i_end-neighbour.i_start +1)
     new_bound_cond_values = []
     
     # Bottom Boundary
     if(neighbour.bound_index == 1) :
         bottom_curr_row = v[0:n+1] # Bottom row
         bottom_prev_row = v[n+1:2*(n+1)] # row above the bottom row
         v_bottom_curr_row  = bottom_curr_row[neighbour.i_start:neighbour.i_end+1]
         v_bottom_prev_row  = bottom_prev_row[neighbour.i_start:neighbour.i_end+1]
         
         values = (v_bottom_curr_row - v_bottom_prev_row)/problem.delta_y
     # Top boundary
     elif(neighbour.bound_index == 3) :
         top_prev_row = v[-(n+1):] # top row
         top_curr_row = v[-2*(n+1):-(n+1)] # row below the top row
         v_top_prev_row  = top_prev_row[neighbour.i_start:neighbour.i_end+1]
         v_top_curr_row  = top_curr_row[neighbour.i_start:neighbour.i_end+1]
         values = (v_top_curr_row - v_top_prev_row)/problem.delta_y
     # Right boundary
     elif (neighbour.bound_index ==2):
          curr_right_col = [] # initialize list to store right column
          prev_right_col = [] # initialize list to store the column, that is previous to the right col
          j_curr = n
          j_prev = n-1
          for i in range(0,m+1):
              curr_right_col.append(v[j_curr + (n+1)*i])
              prev_right_col.append(v[j_prev + (n+1)*i])
          v_curr_right_col  = np.array(curr_right_col[neighbour.i_start:neighbour.i_end+1])
          v_prev_right_col  = np.array(prev_right_col[neighbour.i_start:neighbour.i_end+1])
          values = (v_curr_right_col- v_prev_right_col)/problem.delta_x
    # Left boundary
     elif (neighbour.bound_index == 4):
          curr_left_col = [] # initialize list to store the second column 
          prev_left_col = [] # initialize list to store the first column (left column)
          j_curr = 1
          j_prev = 0
          for i in range(0,m+1):
              curr_left_col.append(v[j_curr + (n+1)*i])
              prev_left_col.append(v[j_prev + (n+1)*i])
          v_curr_left_col  = np.array(curr_left_col[neighbour.i_start:neighbour.i_end+1])
          v_prev_left_col  = np.array(prev_left_col[neighbour.i_start:neighbour.i_end+1])
          values = (v_curr_left_col - v_prev_left_col)/problem.delta_x 
     else:
         raise ValueError("Wrong boundary index, should be inthenger in range(0,4)")   
         
     for x in values:
        new_bound_cond_values.append(x)
        
     return new_bound_cond_types, new_bound_cond_values
    