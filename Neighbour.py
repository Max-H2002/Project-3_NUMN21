import numpy as np
from Point import Point
from Problem import Problem

class Neighbour:
    def __init__(self,id:str,
                 main_problem: Problem,
                 neighbouring_problem: Problem,
                 cond_to_pass: str):
        """
        Initialises instance of Neighbour class with given values.
        Args:
            id (str): id of the neighbouring Problem
            
            cond_to_pass (str): condition to pass to the neighbouring problem ("Dirichlet"/"Neumann")
        """
        
        self.id_neighbour = id
        self.cond_to_pass = cond_to_pass  
        
        bound_idx,E1,E2 = self.find_overlap_points_rectangulars(main_problem,neighbouring_problem)
        if(E1 == None or E2==None):
            raise ValueError("No overlap points on boundaries!")
        else:
            self.bound_index = bound_idx
            if (self.bound_index == 1): # bottom
                p1 = main_problem.A
                p2 = main_problem.B
                step_size = main_problem.delta_x
            elif (self.bound_index == 2): # right
                p1 = main_problem.B
                p2 = main_problem.C
                step_size = main_problem.delta_y
            elif (self.bound_index == 3): # top
                p1 = main_problem.C
                p2 = main_problem.D
                step_size = main_problem.delta_x
            elif (self.bound_index == 4): # left
                p1 = main_problem.D
                p2 = main_problem.A
                step_size = main_problem.delta_y

            self.i_start, self.i_end = self.find_indices_in_grid(p1,p2,E1,E2,step_size)
        
    def find_indices_in_grid(self,boundary_p1: Point,
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
        boundary_p1,boundary_p2 = self.sort_points(boundary_p1,boundary_p2)
        subboundary_p1,subboundary_p2 = self.sort_points(subboundary_p1,subboundary_p2)
        
        # Compute the distances from smallest boundary_p to subboundary_points

        d1 = boundary_p1.distance(subboundary_p1)
        d2 = boundary_p1.distance(subboundary_p2)
        
        # Calculate indexes of the corresponding points on the grid
        i1 = int(d1/step_size)
        i2 = int(d2/step_size)
        
        return i1,i2
        
    def find_overlap_points_rectangulars(self,main_problem: Problem,neighbouring_problem: Problem):
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
        A1 = main_problem.A
        B1 = main_problem.B
        C1 = main_problem.C
        D1 = main_problem.D
        
        # Edges of domain of problem2
        A2 = neighbouring_problem.A
        B2 = neighbouring_problem.B
        C2 = neighbouring_problem.C
        D2 = neighbouring_problem.D
        
        # Check if problem1 left bound - problem2 right bound intersect
        E1,E2 = self.find_overlap_points_segments(A1,D1,B2,C2)
        if (E1 != None and E2 != None): # 2 overlap points
            return 4,E1,E2
        elif (E1 != None and E2 == None): # 1 overlap point
            return 4,E1,None
        
        # Check if problem1 right bound - problem2 left bound intersect
        E1,E2 = self.find_overlap_points_segments(B1,C1,A2,D2)
        if (E1 != None and E2 != None): # 2 overlap points
            return 2,E1,E2
        elif (E1 != None and E2 == None): # 1 overlap point
            return 2,E1, None
        
        # Check if problem1 bottom bound - problem2 top bound intersect
        E1,E2 = self.find_overlap_points_segments(A1,B1,D2,C2)
        if (E1 != None and E2 != None): # 2 overlap points
            return 1,E1,E2
        elif (E1 != None and E2 == None): # 1 overlap point
            return 1,E1, None
        
        # Check if problem1 top bound - problem2 bottom bound intersect
        E1,E2 = self.find_overlap_points_segments(D1,C1,A2,B2)
        if (E1 != None and E2 != None): # 2 overlap points
            return 3,E1,E2
        elif (E1 != None and E2 == None): # 1 overlap point
            return 3,E1, None
        
        raise ValueError("Rectangles have no common boundaries!")

    def find_overlap_points_segments(self,A1: Point,B1: Point,
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
        
    def sort_points(self,p1: Point,
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