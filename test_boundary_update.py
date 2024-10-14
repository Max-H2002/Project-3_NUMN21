import unittest
import numpy as np
from Problem import Problem
from Point import Point
from BoundaryCondtionsUpdate import calculate_Dirichlet_cond, calculate_Neumann_cond
from Neighbour import Neighbour


class TestIdentity(unittest.TestCase):

    # implementation of a setUp method to initialize objects that will be used across multiple of the tests
    def setUp(self):
        # to test Dirichlet update

        self.problem1 = Problem(
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

        self.problem2 = Problem(
            id=1, 
            A=Point(1, 0), 
            B=Point(2, 0), 
            C=Point(2, 1), 
            D=Point(1, 1), 
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
            main_problem=self.problem1, 
            neighbouring_problem=self.problem2, 
            cond_to_pass="Dirichlet"
        )
        self.neighbour_bottom.bound_index = 1
        self.neighbour_bottom.i_start = 0
        self.neighbour_bottom.i_end = 4  # Entire bottom row
        
        # Top boundary test
        self.neighbour_top = Neighbour(
            id="top_neighbour", 
            main_problem=self.problem1, 
            neighbouring_problem=self.problem2, 
            cond_to_pass="Dirichlet"
        )
        self.neighbour_top.bound_index = 3
        self.neighbour_top.i_start = 0
        self.neighbour_top.i_end = 4  # Entire top row

        # Left boundary test
        self.neighbour_left = Neighbour(
            id="left_neighbour", 
            main_problem=self.problem1, 
            neighbouring_problem=self.problem2, 
            cond_to_pass="Dirichlet"
        )
        self.neighbour_left.bound_index = 4
        self.neighbour_left.i_start = 0
        self.neighbour_left.i_end = 3  # Entire left column

        # Right boundary test
        self.neighbour_right = Neighbour(
            id="right_neighbour", 
            main_problem=self.problem1, 
            neighbouring_problem=self.problem2, 
            cond_to_pass="Dirichlet"
        )
        self.neighbour_right.bound_index = 2
        self.neighbour_right.i_start = 0
        self.neighbour_right.i_end = 3  # Entire right column

        # to test Neumann update

        self.problemN1 = Problem(
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

        self.problemN2 = Problem(
            id=1, 
            A=Point(1, 0), 
            B=Point(2, 0), 
            C=Point(2, 1), 
            D=Point(1, 1), 
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
            main_problem=self.problemN1, 
            neighbouring_problem=self.problemN2, 
            cond_to_pass="Neumann"
        )
        self.neighbour_bottomN.bound_index = 1
        self.neighbour_bottomN.i_start = 0
        self.neighbour_bottomN.i_end = 4  # Entire bottom row
        
        # Top boundary test
        self.neighbour_topN = Neighbour(
            id="top_neighbour", 
            main_problem=self.problemN1, 
            neighbouring_problem=self.problemN2, 
            cond_to_pass="Neumann"
        )
        self.neighbour_topN.bound_index = 3
        self.neighbour_topN.i_start = 0
        self.neighbour_topN.i_end = 4  # Entire top row

        # Left boundary test
        self.neighbour_leftN = Neighbour(
            id="left_neighbour", 
            main_problem=self.problemN1, 
            neighbouring_problem=self.problemN2, 
            cond_to_pass="Neumann"
        )
        self.neighbour_leftN.bound_index = 4
        self.neighbour_leftN.i_start = 0
        self.neighbour_leftN.i_end = 3  # Entire left column

        # Right boundary test
        self.neighbour_rightN = Neighbour(
            id="right_neighbour", 
            main_problem=self.problemN1, 
            neighbouring_problem=self.problemN2, 
            cond_to_pass="Neumann"
        )
        self.neighbour_rightN.bound_index = 2
        self.neighbour_rightN.i_start = 0
        self.neighbour_rightN.i_end = 3  # Entire right column



    """
    Tests for BoundaryConditionsUpdate file.
    """
    
    def test_Dirichlet_bottom(self):
        new_types, new_values = calculate_Dirichlet_cond(self.v, self.problem1, self.neighbour_bottom)
        expected_types = ["Dirichlet"] * 5
        expected_bottom_row = [1, 2, 3, 4, 5]
        self.assertEqual(new_types, expected_types)
        self.assertEqual(new_values, expected_bottom_row)

    def test_Dirichlet_top(self):
        new_types, new_values = calculate_Dirichlet_cond(self.v, self.problem1, self.neighbour_top)
        expected_types = ["Dirichlet"] * 5
        expected_top_row = [16, 17, 18, 19, 20]
        self.assertEqual(new_types, expected_types)
        self.assertEqual(new_values, expected_top_row)

    def test_Dirichlet_left(self):
        new_types, new_values = calculate_Dirichlet_cond(self.v, self.problem1, self.neighbour_left)
        expected_types = ["Dirichlet"] * 4
        expected_left_col = [1, 6, 11, 16]
        self.assertEqual(new_types, expected_types)
        self.assertEqual(new_values, expected_left_col)

    def test_Dirichlet_right(self):
        new_types, new_values = calculate_Dirichlet_cond(self.v, self.problem1, self.neighbour_right)
        expected_types = ["Dirichlet"] * 4
        expected_right_col = [5, 10, 15, 20]
        self.assertEqual(new_types, expected_types)
        self.assertEqual(new_values, expected_right_col)

    def test_Neumann_bottom(self):
        new_types, new_values = calculate_Neumann_cond(self.v, self.problemN1, self.neighbour_rightN)
        expected_types = ["Neumann"] * 5
        expected_values = [-15, -15, -15, -15, -15]  # First row in flattened `v`
        
        self.assertEqual(new_types, expected_types)
        self.assertEqual(new_values, expected_values)
    
    def test_Neumann_top(self):
        new_types, new_values = calculate_Neumann_cond(self.v, self.problemN1, self.neighbour_rightN)
        expected_types = ["Neumann"] * 5
        expected_values = [-15, -15, -15, -15, -15]  # First row in flattened `v`
        
        self.assertEqual(new_types, expected_types)
        self.assertEqual(new_values, expected_values)
    
    def test_Neumann_right(self):
        new_types, new_values = calculate_Neumann_cond(self.v, self.problemN1, self.neighbour_rightN)
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
