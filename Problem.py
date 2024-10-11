'Project 3: Advanced Numerical Algorithms in Python'

from Point import Point 
        
        
class Problem:
    def __init__(self, A:Point, B: Point, C: Point, D: Point, 
                delta_x: float, delta_y:float,
                boundary_conditions_types: list,
                boundary_conditions_values: list):
        """
        Initializes class Problem with given values.
        
        Parameters:
            A: Point, coordinates of the left down corner of the rectangle
            B: Point, coordinates of the right down corner of the rectangle
            C: Point, coordinates of the right up corner of the rectangle
            D: Point, coordinates of the left up corner of the rectangle
            delta_x: float, stepsize of the grid in x direction
            delta_y: float, stepsize of the grid in y direction
            boundary_conditions: list of lenght 4. the i-th el in the list is the list of tuple pairs in the format (string, float),
            where string is the condition type ("Dirichlet"/"Neumann") and float is the value of the condition. 
            Each pair sets boundary condition one node. 
            i = 0 correspond to the lower boundary, other boundaries are numerated in the conterclockwise order respectively.

        """
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.delta_x = delta_x
        self.delta_y = delta_y
        self.boundary_conditions_types = boundary_conditions_types
        self.boundary_conditions_values = boundary_conditions_values
        
        # To do : implement checks, if the parameters are valid (delta_x/delta_y,boundary conditions)

    def update_boundary(self,new_types,new_values: list, bound_info:int):
        '''
        Updates the boundary condition with new_values for the boundary i.
        
        Parameters:
            new_values: list of tuples (string, float) - new boundary condition
            i: int, index of the boundary condition we need to change
        
        Returns:
        -
        '''
        i = bound_info[0] # index of the boundary we need to change conditions on
        i_start = bound_info[1]
        i_end = bound_info[2]
        self.boundary_conditions_types[i-1][i_start:i_end] = new_types
        self.boundary_conditions_values[i-1][i_start:i_end] = new_values
        
        # To do : implement checks, if the new boundary conditions are valid

        