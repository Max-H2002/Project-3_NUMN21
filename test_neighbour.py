import unittest
import numpy as np
from Problem import Problem
from Neighbour import Neighbour
from Point import Point

class TestIdentity(unittest.TestCase):

    # implementation of a setUp method to initialize objects that will be used across multiple of the tests
    def setUp(self):

        # points to sort 
        self.p1 = Point(3,3)
        self.p2 = Point(1,1)

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

        # Set up neighbour configurations for each boundary:
        # Bottom boundary test
        self.neighbour = Neighbour(
            id="neighbour", 
            main_problem=self.problem1, 
            neighbouring_problem=self.problem2, 
            cond_to_pass="Dirichlet"
        )

    # tests the sort_point method
    def test_sort_points(self):
        # output of sort_points
        sorted_points = self.neighbour.sort_points(self.p1, self.p2)

        # Convert the sorted Point objects into a list of tuples for comparison
        sorted_points_coords = [(sorted_points[0].x, sorted_points[0].y), 
                            (sorted_points[1].x, sorted_points[1].y)]

        # expected output as tuples
        expected_coords = [(self.p1.x, self.p1.y), (self.p2.x, self.p2.y)]
        sorted_points_expected = sorted(expected_coords)

        message = "The points are not in ascending order."

        # Compare the sorted tuples instead of Point objects
        self.assertEqual(sorted_points_coords, sorted_points_expected, message)



