import unittest
import numpy as np
from Problem import Problem
from Point import Point
from BoundaryCondtionsUpdate import calculate_Dirichlet_cond, calculate_Neumann_cond
from Neighbour import Neighbour

# derive from unittest.TestCase
class TestIdentity(unittest.TestCase):

    # implementation of a setUp method to initialize objects that will be used across multiple of the tests
    def setUp(self):
        # Initialize Points for the rectangle
        self.A = Point(0,0)
        self.B = Point(1,0)
        self.C = Point(1,1)
        self.D = Point(0,1)

        # Set the grid sizes
        self.delta_x = 1/3
        self.delta_y = 1/3

        # Define boundary condtitions for all four sides of the rectangle
        self.boundary_conditions = [
            [("Dirichlet", 0)],     # Lower boundary
            [("Neumann", 1)],       # Right boundary
            [("Dirichlet", 2)],     # Upper boundary
            [("Neumann", 3)]        # Left boundary
        ]

        # to test Dirichlet update

        self.problem = Problem(
            id=1, 
            A=Point(0, 0), 
            B=Point(1, 0), 
            C=Point(1, 1), 
            D=Point(0, 1), 
            delta_x=0.25, 
            delta_y=1/3, 
            boundary_conditions_types=["Dirichlet"],  
            boundary_conditions_values=[1]
        )

        # Define v, flattened grid array from 1 to 20
        self.v = np.arange(1, 21).flatten()

        # Set up neighbour configurations for each boundary:
        # Bottom boundary test
        self.neighbour_bottom = Neighbour(
            id="bottom_neighbour", 
            main_problem=self.problem, 
            neighbouring_problem=self.problem, 
            cond_to_pass="Dirichlet"
        )
        self.neighbour_bottom.bound_index = 1
        self.neighbour_bottom.i_start = 0
        self.neighbour_bottom.i_end = 4  # Entire bottom row
        
        # Top boundary test
        self.neighbour_top = Neighbour(
            id="top_neighbour", 
            main_problem=self.problem, 
            neighbouring_problem=self.problem, 
            cond_to_pass="Dirichlet"
        )
        self.neighbour_top.bound_index = 3
        self.neighbour_top.i_start = 0
        self.neighbour_top.i_end = 4  # Entire top row

        # Left boundary test
        self.neighbour_left = Neighbour(
            id="left_neighbour", 
            main_problem=self.problem, 
            neighbouring_problem=self.problem, 
            cond_to_pass="Dirichlet"
        )
        self.neighbour_left.bound_index = 4
        self.neighbour_left.i_start = 0
        self.neighbour_left.i_end = 3  # Entire left column

        # Right boundary test
        self.neighbour_right = Neighbour(
            id="right_neighbour", 
            main_problem=self.problem, 
            neighbouring_problem=self.problem, 
            cond_to_pass="Dirichlet"
        )
        self.neighbour_right.bound_index = 2
        self.neighbour_right.i_start = 0
        self.neighbour_right.i_end = 3  # Entire right column

        # to test Neumann update

        self.problemN = Problem(
            id=1, 
            A=Point(0, 0), 
            B=Point(1, 0), 
            C=Point(1, 1), 
            D=Point(0, 1), 
            delta_x=0.25, 
            delta_y=1/3, 
            boundary_conditions_types=["Neumann"],  
            boundary_conditions_values=[1]
        )

        # Define v, flattened grid array from 1 to 20
        self.v = np.arange(1, 21).flatten()

        # Set up neighbour configurations for each boundary:
        # Bottom boundary test
        self.neighbour_bottomN = Neighbour(
            id="bottom_neighbour", 
            main_problemN=self.problemN, 
            neighbouring_problemN=self.problemN, 
            cond_to_passN="Neumann"
        )
        self.neighbour_bottomN.bound_index = 1
        self.neighbour_bottomN.i_start = 0
        self.neighbour_bottomN.i_end = 4  # Entire bottom row
        
        # Top boundary test
        self.neighbour_topN = Neighbour(
            id="top_neighbour", 
            main_problemN=self.problemN, 
            neighbouring_problemN=self.problemN, 
            cond_to_passN="Neumann"
        )
        self.neighbour_topN.bound_index = 3
        self.neighbour_topN.i_start = 0
        self.neighbour_topN.i_end = 4  # Entire top row

        # Left boundary test
        self.neighbour_leftN = Neighbour(
            id="left_neighbour", 
            main_problemN=self.problemN, 
            neighbouring_problemN=self.problemN, 
            cond_to_passN="Neumann"
        )
        self.neighbour_leftN.bound_index = 4
        self.neighbour_leftN.i_start = 0
        self.neighbour_leftN.i_end = 3  # Entire left column

        # Right boundary test
        self.neighbour_rightN = Neighbour(
            id="right_neighbour", 
            main_problemN=self.problemN, 
            neighbouring_problemN=self.problemN, 
            cond_to_passN="Neumann"
        )
        self.neighbour_rightN.bound_index = 2
        self.neighbour_rightN.i_start = 0
        self.neighbour_rightN.i_end = 3  # Entire right column



        # Set coordinates for the Point class tests
        self.x = 3.786653456566778898
        self.y = 4.997896655352456778090
        self.point = Point(6,8)

    # all methods that start with test are executed

    """
    Tests for the Problem class.
    """
    # tests if the grid size is valid for the size of the room
    def test_grid_size(self):

        problem = Problem(id = 1, A=Point(0, 0), B=Point(1, 0), C = Point(1,1),  D=Point(0, 1), delta_x=0.25, delta_y=1/3, boundary_conditions_types=["Dirichlet"], boundary_conditions_values=[1])  # Customize based on real attributes

        # length of rectangle to check for valid delta_x
        length_rectangle = np.abs(problem.B.x - problem.A.x)
        # length of rectangle to check for valid delta_y
        width_rectangle = np.abs(problem.B.y - problem.C.y)

        # check if grid size is valid
        del_x_val = (length_rectangle/problem.delta_x) - int(length_rectangle/problem.delta_x) 
        del_y_val = (width_rectangle/problem.delta_y) - int(width_rectangle/problem.delta_y)

        expected = 0

        message_x = "The chosen grid size {problem.delta_x} is not valid!"
        message_y = "The chosen grid size {problem.delta_y} is not valid!"

        self.assertEqual(del_x_val, expected, message_x)
        self.assertEqual(del_y_val, expected, message_y)

    # tests that the grid size is non-negative
    def test_grid_size_negative(self):

        if (self.delta_x < 0 and self.delta_y < 0):
            self.fail(f"The chosen grid sizes delta_x: {self.delta_x}, delta_y: {self.delta_y} are negative. Please chose only non-negative values.")
        elif (self.delta_x < 0):
            self.fail(f"The chosen grid size delta_x: {self.delta_x} is negative. Please chose only non-negative values.")
        elif (self.delta_y < 0):
            self.fail(f"The chosen grid size delta_y: {self.delta_y} is negative. Please chose only non-negative values.")
    

        #with self.assertRaises(ValueError):
        #    Problem(self.A, self.B, self.C, self.D, -self.delta_x, self.delta_y, self.boundary_conditions)

        #with self.assertRaises(ValueError):
        #    Problem(self.A, self.B, self.C, self.D, self.delta_x, -self.delta_y, self.boundary_conditions)
        
        #with self.assertRaises(ValueError):
        #    Problem(self.A, self.B, self.C, self.D, -self.delta_x, -self.delta_y, self.boundary_conditions)

    
    # tests that grid size is smaller then the room
    def test_step_size(self):
        
        problem = Problem(id = 1, A=Point(0, 0), B=Point(1, 0), C = Point(1,1),  D=Point(0, 1), delta_x=0.25, delta_y=1/3, boundary_conditions_types=["Dirichlet"], boundary_conditions_values=[1])  # Customize based on real attributes

        # length of rectangle to check for valid delta_x
        length_rectangle = np.abs(problem.B.x - problem.A.x)
        # length of rectangle to check for valid delta_y
        width_rectangle = np.abs(problem.B.y - problem.C.y)

        if (problem.delta_x <= length_rectangle):
            size_x = 0
            return size_x
        
        if (problem.delta_y <= width_rectangle):
            size_y = 0
            return size_y
        
        message_x = "The chosen grid size {problem.delta_x} exceeds the size of the room of {length_rectangle}. Chose a grid that is smaller then the room."
        message_y = "The chosen grid size {problem.delta_y} exceeds the size of the room of {width_rectangle}. Chose a grid that is smaller then the room."

        self.assertEqual(size_x, 0, message_x)
        self.assertEqual(size_y, 0, message_y)

    
    """
    Tests for the Point class.
    """
    # tests that the distance between two points is calculated correctly
    def test_distance(self):

        point_xy = Point(self.x, self.y)

        vector = np.array([point_xy.x, point_xy.y])
        point_vec = np.array([self.point.x, self.point.y])

        dist = point_xy.distance(self.point)

        expected_dist = np.linalg.norm(vector - point_vec)

        message = "The euklidien distance hasn't been calculated correctly."

        self.assertEqual(dist, expected_dist, message)


    """
    Tests for BoundaryConditionsUpdate file.
    """
    
    def test_Dirichlet_bottom(self):
        new_types, new_values = calculate_Dirichlet_cond(self.v, self.problem, self.neighbour_bottom)
        expected_types = ["Dirichlet"] * 5
        expected_bottom_row = [1, 2, 3, 4, 5]
        self.assertEqual(new_types, expected_types)
        self.assertEqual(new_values, expected_bottom_row)

    def test_Dirichlet_top(self):
        new_types, new_values = calculate_Dirichlet_cond(self.v, self.problem, self.neighbour_top)
        expected_types = ["Dirichlet"] * 5
        expected_top_row = [16, 17, 18, 19, 20]
        self.assertEqual(new_types, expected_types)
        self.assertEqual(new_values, expected_top_row)

    def test_Dirichlet_left(self):
        new_types, new_values = calculate_Dirichlet_cond(self.v, self.problem, self.neighbour_left)
        expected_types = ["Dirichlet"] * 4
        expected_left_col = [1, 6, 11, 16]
        self.assertEqual(new_types, expected_types)
        self.assertEqual(new_values, expected_left_col)

    def test_Dirichlet_right(self):
        new_types, new_values = calculate_Dirichlet_cond(self.v, self.problem, self.neighbour_right)
        expected_types = ["Dirichlet"] * 4
        expected_right_col = [5, 10, 15, 20]
        self.assertEqual(new_types, expected_types)
        self.assertEqual(new_values, expected_right_col)

    def test_Neumann_bottom(self):
        new_types, new_values = calculate_Neumann_cond(self.v, self.problemN, self.neighbour_rightN)
        expected_types = ["Neumann"] * 5
        expected_values = [-15, -15, -15, -15, -15]  # First row in flattened `v`
        
        self.assertEqual(new_types, expected_types)
        self.assertEqual(new_values, expected_values)
    
    def test_Neumann_top(self):
        new_types, new_values = calculate_Neumann_cond(self.v, self.problemN, self.neighbour_rightN)
        expected_types = ["Neumann"] * 5
        expected_values = [-15, -15, -15, -15, -15]  # First row in flattened `v`
        
        self.assertEqual(new_types, expected_types)
        self.assertEqual(new_values, expected_values)
    
    def test_Neumann_right(self):
        new_types, new_values = calculate_Neumann_cond(self.v, self.problemN, self.neighbour_rightN)
        expected_types = ["Neumann"] * 4
        expected_values = [4, 4, 4, 4]  # Right column elements in flattened `v`
        
        self.assertEqual(new_types, expected_types)
        self.assertEqual(new_values, expected_values)
    
    def test_Dirichlet_left(self):
        new_types, new_values = calculate_Neumann_cond(self.v, self.problemN, self.neighbour_rightN)
        expected_types = ["Neumann"] * 4
        expected_values = [4, 4, 4, 4]  # Left column elements in flattened `v`
        
        self.assertEqual(new_types, expected_types)
        self.assertEqual(new_values, expected_values)

