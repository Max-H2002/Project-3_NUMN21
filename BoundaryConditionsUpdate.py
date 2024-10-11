import numpy as np
import math
from Point import Point
from Problem import Problem

def update_boundaty_cond(v,problem_from, problem_to,cond_type):
    # Find intersection points of domains of Problem1 and Problem2;
    
    # Find idices of intersection points on the grid for Problem1 and Problem2
    
    # Recalculate boundary condition
    
    # Update boundary condition
    pass

def calculate_Dirichlet_cond(v: np.array, 
                             problem : Problem, 
                             bound_info: tuple):
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
     bound_index = bound_info[0]
     i_start = bound_info[1]
     i_end = bound_info[2]
    
     # sizes of the grid
     n = int( problem.A.distance(problem.B)/ problem.delta_x ) # (n+1) el in the row 
     m = int( problem.A.distance(problem.D)/ problem.delta_y ) # (m+1) el in a column
     
     new_bound_cond_types = ["Dirichlet"]*(i_end-i_start +1)
     new_bound_cond_values = []
     
     # Bottom Boundary
     if(bound_index == 1) :
         bottom_row = v[0:n+1]
         print(bottom_row)
         values = bottom_row[i_start:i_end+1]
     # Top boundary
     elif(bound_index == 3) :
         top_row = v[-(n+1):]
         values = top_row[i_start:i_end+1]
     # Right boundary
     elif (bound_index ==2):
          right_col = []
          j = n
          for i in range(0,m+1):
              right_col.append(v[j + (n+1)*i])
          values = right_col[i_start:i_end+1]
    # Left boundary
     elif (bound_index == 4):
          left_col = []
          j = 0
          for i in range(0,m+1):
              left_col.append(v[j + (n+1)*i]) 
          values = left_col[i_start:i_end+1] 
     else:
         raise ValueError("Wrong boundary index, should be inthenger in range(0,4)")   
         
     for x in values:
        new_bound_cond_values.append(x)
        
      
    
     return new_bound_cond_types, new_bound_cond_values
 
def calculate_Neumann_cond(v: np.array, 
                             problem : Problem, 
                             bound_info: tuple):
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
         list: list of tuples (string, float,float) - new Neumann boundary condition
     """
     # Get data from tuple
     bound_index = bound_info[0]
     i_start = bound_info[1]
     i_end = bound_info[2]
    
     # sizes of the grid
     n = int( problem.A.distance(problem.B)/ problem.delta_x ) # (n+1) el in the row 
     m = int( problem.A.distance(problem.D)/ problem.delta_y ) # (m+1) el in a column
     
     new_bound_cond_types = ["Neumann"]*(i_end-i_start +1)
     new_bound_cond_values = []
     
     # Bottom Boundary
     if(bound_index == 1) :
         bottom_curr_row = v[0:n+1] # Bottom row
         bottom_prev_row = v[n+1:2*(n+1)] # row above the bottom row
         v_bottom_curr_row  = bottom_curr_row[i_start:i_end+1]
         v_bottom_prev_row  = bottom_prev_row[i_start:i_end+1]
         
         values = (v_bottom_curr_row - v_bottom_prev_row)/problem.delta_y
     # Top boundary
     elif(bound_index == 3) :
         top_prev_row = v[-(n+1):] # top row
         top_curr_row = v[-2*(n+1):-(n+1)] # row below the top row
         v_top_prev_row  = top_prev_row[i_start:i_end+1]
         v_top_curr_row  = top_curr_row[i_start:i_end+1]
         values = (v_top_curr_row - v_top_prev_row)/problem.delta_y
     # Right boundary
     elif (bound_index ==2):
          curr_right_col = [] # initialize list to store right column
          prev_right_col = [] # initialize list to store the column, that is previous to the right col
          j_curr = n
          j_prev = n-1
          for i in range(0,m+1):
              curr_right_col.append(v[j_curr + (n+1)*i])
              prev_right_col.append(v[j_prev + (n+1)*i])
          v_curr_right_col  = np.array(curr_right_col[i_start:i_end+1])
          v_prev_right_col  = np.array(prev_right_col[i_start:i_end+1])
          values = (v_curr_right_col- v_prev_right_col)/problem.delta_x
    # Left boundary
     elif (bound_index == 4):
          curr_left_col = [] # initialize list to store the second column 
          prev_left_col = [] # initialize list to store the first column (left column)
          j_curr = 1
          j_prev = 0
          for i in range(0,m+1):
              curr_left_col.append(v[j_curr + (n+1)*i])
              prev_left_col.append(v[j_prev + (n+1)*i])
          v_curr_left_col  = np.array(curr_left_col[i_start:i_end+1])
          v_prev_left_col  = np.array(prev_left_col[i_start:i_end+1])
          values = (v_curr_left_col - v_prev_left_col)/problem.delta_x 
     else:
         raise ValueError("Wrong boundary index, should be inthenger in range(0,4)")   
         
     for x in values:
        new_bound_cond_values.append(x)
        
      
    
     return new_bound_cond_types, new_bound_cond_values
    
def find_indices_in_grid(boundary_p1: Point,
                           boundary_p2: Point,
                           subboundary_p1: Point,
                           subboundary_p2: Point,
                           step_size: float):
    """
    Finds the idices of the grid of the boundary, where edges of subboundary are located.
    
    Args:
        boundary_p1 (Point): _description_
        boundary_p2 (Point): _description_
        subboundary_p1 (Point): _description_
        subboundary_p2 (Point): _description_
        step_size (float): _description_
        
    Returns:
    i1 : int, smaller index
    i2 : int, bigger index
    """
    
    # Ensure that points are stored at the ascend order
    boundary_p1,boundary_p2 = sort_points(boundary_p1,boundary_p2)
    subboundary_p1,subboundary_p2 = sort_points(subboundary_p1,subboundary_p2)
    
    # Compute the distances from smallest boundary_p to subboundary_points

    d1 = boundary_p1.distance(subboundary_p1)
    d2 = boundary_p1.distance(subboundary_p2)
    
    # Calculate indexes of the corresponding points on the grid
    i1 = d1/step_size
    i2 = d2/step_size
    
    return i1,i2

def sort_points(p1: Point,
                p2: Point):
    """
    Returns given points in the ascending order. 

    Args:
        p1 (Point): point 1
        p2 (Point): point 2

    Returns:
        p1, p2 (Point): p1 - smaller pont, p2 - bigger point
    """
    # Note: only works for lines that are parralell to axes ox/oy?
    # Compare points by coordinates
    if (p1.x, p1.y) < (p2.x, p2.y):
        return p1, p2
    else:
        return p2, p1
    
def find_overlap_points_rectangulars(problem1: Problem,problem2: Problem):
    """
    Ifv they exist, finds overlap points of two rectangles.

    Args:
        problem1 (Problem): problem from
        problem2 (Problem): problem to

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    # Here we assume that the points in the problem class are stored ordered counterclockwise with the first point A being in the left down corner.
    
    # Edges of domain of problem1
    A1 = problem1.A
    B1 = problem1.B
    C1 = problem1.C
    D1 = problem1.D
    
    # Edges of domain of problem2
    A2 = problem2.A
    B2 = problem2.B
    C2 = problem2.C
    D2 = problem2.D
    
    # Check if problem1 left bound - problem2 right bound intersect
    E1,E2 = find_overlap_points_segments(A1,D1,B2,C2)
    if (E1 != None and E2 != None): # 2 overlap points
        return E1,E2
    elif (E1 != None and E2 == None): # 1 overlap point
        return E1
    
    # Check if problem1 right bound - problem2 left bound intersect
    E1,E2 = find_overlap_points_segments(B1,C1,A2,D2)
    if (E1 != None and E2 != None): # 2 overlap points
        return E1,E2
    elif (E1 != None and E2 == None): # 1 overlap point
        return E1, None
    
    # Check if problem1 bottom bound - problem2 top bound intersect
    E1,E2 = find_overlap_points_segments(A1,B1,D2,C2)
    if (E1 != None and E2 != None): # 2 overlap points
        return E1,E2
    elif (E1 != None and E2 == None): # 1 overlap point
        return E1, None
    
    # Check if problem1 top bound - problem2 bottom bound intersect
    E1,E2 = find_overlap_points_segments(D1,C1,A2,B2)
    if (E1 != None and E2 != None): # 2 overlap points
        return E1,E2
    elif (E1 != None and E2 == None): # 1 overlap point
        return E1, None
    
    raise ValueError("Rectangles have no common boundaries!")

def find_overlap_points_segments(A1: Point,B1: Point,
                                 A2:Point,B2: Point):
    """
    If they exist, finds the overlap points of 2 parallel segment.

    Args:
        A1 (Point): edge of segment 1
        B1 (Point): edge of segment 1
        A2 (Point): edge of segment 2
        B2 (Point): edge of segment 2

    Returns:
        Point, Point: two overlap points of segments A1B1 and A2B2 (first point is left/bottom point) 
        Point, None: one overlap point
        None, None: no overlap points
    """
    # Note: segments should be parallel
    # Ensure that the segments are ordered left-to-right or top-to-bottom
    if (A1.y == B1.y and A1.x > B1.x) or (A1.x == B1.x and A1.y > B1.y):
        A1, B1 = B1, A1
    if (A2.y == B2.y and A2.x > B2.x) or (A2.x == B2.x and A2.y > B2.y):
        #print("true")
        A2, B2 = B2, A2
    # Note: segments should be parallel
    # segments should also be ordered left-to-right or bottom-to-top
    
    # Check if the segments are vertical or horizontal
    if A1.x == A2.x == B1.x == B2.x:  # Vertical segments
        #print("Vertical")
         # Ensure segments are ordered, i.e. A1 is always a furthest left point.
        if(A1.y>A2.y):
            A1, B1, A2, B2 = A2, B2, A1, B1
        # Check if the lines overlap
        if(A2.y<B1.y):
            # A1___A2___B2___B1 => A2,B2
            if(B2.y<=B1.y):
                return A2,B2
            # A1___A2___B1___B2 => A2,B1
            else:
                return A2, B1
        # Check if the lines overlap in one point
        elif(A2.y == B1.y):
            return A2, None
        # Otherwise - lines do not overlap
        else:
            return None,None

    elif A1.y == A2.y == B1.y == B2.y:  # Horizontal segments
        # Ensure segments are ordered, i.e. A1 is always a furthest left point.
        #print("Horizontal")
        if(A1.x>A2.x):
            A1, B1, A2, B2 = A2, B2, A1, B1
        # Check if the lines overlap
        if(A2.x<B1.x):
            # A1___A2___B2___B1 => A2,B2
            if(B2.x<=B1.x):
                return A2,B2
            # A1___A2___B1___B2 => A2,B1
            else:
                return A2, B1
        # Check if the lines overlap in one point
        elif(A2.x == B1.x):
            return A2, None
        # Otherwise - lines do not overlap
        else:
            return None, None
    else:
        return None,None

def test_func():
     p1 = Point(0,0)
     p2 = Point(7,0)
     
     sp1 = Point(7,0)
     sp2 = Point(3,0)
     
     #Test(Find_indices_in_grid)
    # i1,i2 = find_indices_in_grid(p1,p2,sp1,sp2,1)
    #  print(i1)
    #  print(i2)
    
    # Test find_overlap_points_segments()
    #  A,B = find_overlap_points_segments(p1,p2,sp2,sp1)
    #  if A != None:
    #     print(A.x,", ",A.y)
    #  if B != None:
    #     print(B.x,", ",B.y)
     
     #Test sort_points()
    #  sp1, sp2 = sort_points(p1,p2)
    #  print(sp1.x,",",sp1.y)
    #  print(sp2.x,",",sp2.y)
        
      # Test find_overlap_points_rectangulars  
     p1 = Point(0,0)
     p2 = Point(2,0)
     p3 = Point(2,2)
     p4 = Point(0,2)
     
     rec1 = Problem(p1,p2,p3,p4,1,1,[])
     
     p1 = Point(2,0.5)
     p2 = Point(4,0.5)
     p3 = Point(4,1)
     p4 = Point(2,1)
     
     rec2 = Problem(p1,p2,p3,p4,1,1,[])
     
     E1,E2 = find_overlap_points_rectangulars(rec1,rec2)
     
     if E1 != None:
        print(E1.x,", ",E1.y)
     if E2 != None:
        print(E2.x,", ",E2.y)
     return 0
  
#test_func() 