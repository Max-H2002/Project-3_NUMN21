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


    # tests for find_overlap_points_segments

    # tests fully overlapping segments
    def test_fully_overlapping_segments(self):
        # horizontally
        A1_h = Point(1,0)
        B1_h = Point(5,0)
        A2_h = Point(2,0)
        B2_h = Point(4,0)
        # vertically
        A1_v = Point(0,0)
        B1_v = Point(0,4)
        A2_v = Point(0,1)
        B2_v = Point(0,3)
        # horizontally
        # expected overlap is the segment of A2 to B2
        result_h = self.neighbour.find_overlap_points_segments(A1_h, B1_h, A2_h, B2_h)
        self.assertEqual(result_h, (A2_h, B2_h), f"The segments are not fully overlapping horizontally.")
        # vertically
        # expected overlap is the segment of A2 to B2
        result_v = self.neighbour.find_overlap_points_segments(A1_v, B1_v, A2_v, B2_v)
        self.assertEqual(result_v, (A2_v, B2_v), f"The segments are not fully overlapping vertically.")

    # tests partially overlapping segments
    def test_partially_overlapping_segments(self):
        # horizontally
        A1_h = Point(3,0)
        B1_h = Point(7,0)
        # overlapping segment (left)
        A2_hl = Point(0,0)
        B2_hl = Point(4,0)
        # overlapping segment (right)
        A2_hr = Point(6,0)
        B2_hr = Point(9,0)
        # vertically
        A1_v = Point(0,1)
        B1_v = Point(0,4)
        # overlapping segment (upper)
        A2_vu = Point(0,3)
        B2_vu = Point(0,5)
        # overlapping segment (lower)
        A2_vl = Point(0,0)
        B2_vl = Point(0,2)
        # horizontally
        # expected overlap (left of segment 1) of A1_h to B2_hl
        result_hl = self.neighbour.find_overlap_points_segments(A1_h, B1_h, A2_hl, B2_hl)
        self.assertEqual(result_hl, (A1_h, B2_hl), f"The segments are not overlapping on the left side of the first segment.")
        # expected overlap (right of segment 2) of A2_hr to B1_h
        result_hr = self.neighbour.find_overlap_points_segments(A1_h, B1_h, A2_hr, B2_hr)
        self.assertEqual(result_hr, (A2_hr, B1_h), f"The segments are not overlapping on the right side of the first segment.")
        # vertically
        # expected overlap (upper of segment 1) of A2_vu to B1_v
        result_vu = self.neighbour.find_overlap_points_segments(A1_v, B1_v, A2_vu, B2_vu)
        self.assertEqual(result_vu, (A2_vu, B1_v), f"The segments are not overlapping on the left side of the first segment.")
        # expected overlap (lower of segment 2) of A1_v to B2_vl
        result_vl = self.neighbour.find_overlap_points_segments(A1_v, B1_v, A2_vl, B2_vl)
        self.assertEqual(result_vl, (A1_v, B2_vl), f"The segments are not overlapping on the right side of the first segment.")

    # tests overlapping points
    def test_points_overlapping_segments(self):
        A1 = Point(3,0)
        B1 = Point(6,0)
        # intersecting points of the segments, left of the first segment
        A2_l = Point(0,0)
        B2_l = Point(3,0)
        # intersecting points of the segments, right of the first segment
        A2_r = Point(6,0)
        B2_r = Point(9,0)
        # expected intersection (left of segment 1) of B2_l and A1
        result_l = self.neighbour.find_overlap_points_segments(A1, B1, A2_l, B2_l)
        self.assertEqual(result_l, (A1, None), f"The segments are not intersecting on the left of the first segment.")
        # expected intersection (right of segment 1) of B1 and A2_r
        result_r = self.neighbour.find_overlap_points_segments(A1, B1, A2_r, B2_r)
        self.assertEqual(result_r, (A2_r, None), f"The segments are not intersecting on the right of the first segment.")

    # tests not overlapping segments
    def test_not_overlapping_segments(self):
        A1 = Point(3,0)
        B1 = Point(5,0)
        # not overlapping segment on the left side of the first segment
        A2_l = Point(0,0)
        B2_l = Point(2,0)
        # not overlapping segment on the right side of the first segment
        A2_r = Point(6,0)
        B2_r = Point(8,0)
        # expected nonoverlapping segment to the left of the first segment 
        result_l = self.neighbour.find_overlap_points_segments(A1, B1, A2_l, B2_l)
        self.assertEqual(result_l, (None, None), f"The segments are overlapping.")
        # expected nonoverlapping segment to the right of the first segment 
        result_r = self.neighbour.find_overlap_points_segments(A1, B1, A2_r, B2_r)
        self.assertEqual(result_r, (None, None), f"The segments are overlapping.")

    # tests non parallel segments
    def test_non_parallel_segments(self):
        A1 = Point(0,3)
        B1 = Point(3,3)
        A2 = Point(4,0)
        B2 = Point(6,0)
        # expected non parallel segments
        result = self.neighbour.find_overlap_points_segments(A1, B1, A2, B2)
        self.assertEqual(result, (None, None), f"The segments are overlapping.")


    # tests for find_overlap_points_rectangulars

    def tests_vertical_overlap_points_rectangulars_full_intersec(self):
        # first case, full intersection 
        A1_1 = Point(3,0)
        B1_1 = Point(6,0)
        C1_1 = Point(6,2)
        D1_1 = Point(3,2)
        A2_1 = Point(0,0)
        B2_1 = Point(3,0)
        C2_1 = Point(3,2)
        D2_1 = Point(0,2)
        problem1_1 = Problem(1, A1_1, B1_1, C1_1, D1_1, 1, 1, ["Dirichlet"], [0])
        problem2_1 = Problem(1, A2_1, B2_1, C2_1, D2_1, 1, 1, ["Dirichlet"], [0])

        result = self.neighbour.find_overlap_points_rectangulars(problem1_1, problem2_1)
        self.assertEqual(result, (4, B2_1, C2_1), "The segments overlap.")

    def tests_vertical_overlap_points_rectangulars_partial_intersec_upper(self):   
        # second case, partial intersection upper 
        A1_2 = Point(3,1)
        B1_2 = Point(6,1)
        C1_2 = Point(6,3)
        D1_2 = Point(3,3)
        A2_2 = Point(0,0)
        B2_2 = Point(3,0)
        C2_2 = Point(3,2)
        D2_2 = Point(0,2)
        problem1_2 = Problem(1, A1_2, B1_2, C1_2, D1_2, 1, 1, ["Dirichlet"], [0])
        problem2_2 = Problem(1, A2_2, B2_2, C2_2, D2_2, 1, 1, ["Dirichlet"], [0])
    
        result = self.neighbour.find_overlap_points_rectangulars(problem1_2, problem2_2)
        self.assertEqual(result, (4, A1_2, C2_2), "The segments overlap.")


    def tests_vertical_overlap_points_rectangulars_partial_intersec_lower(self):   
        # third case, partial intersection lower 
        A1_3 = Point(3,0)
        B1_3 = Point(6,0)
        C1_3 = Point(6,1)
        D1_3 = Point(3,1)
        A2_3 = Point(0,0)
        B2_3 = Point(3,0)
        C2_3 = Point(3,2)
        D2_3 = Point(0,2)
        problem1_3 = Problem(1, A1_3, B1_3, C1_3, D1_3, 1, 1, ["Dirichlet"], [0])
        problem2_3 = Problem(1, A2_3, B2_3, C2_3, D2_3, 1, 1, ["Dirichlet"], [0])

        result = self.neighbour.find_overlap_points_rectangulars(problem1_3, problem2_3)
        self.assertEqual(result, (4, A1_3, D1_3), "The segments overlap.")


    def tests_vertical_overlap_points_rectangulars_partial_intersec_lower(self):   
        # fourtht case, partial intersection middle
        A1_4 = Point(3,0.5)
        B1_4 = Point(6,0.5)
        C1_4 = Point(6,1.5)
        D1_4 = Point(3,1.5)
        A2_4 = Point(0,0)
        B2_4 = Point(3,0)
        C2_4 = Point(3,2)
        D2_4 = Point(0,2)
        problem1_4 = Problem(1, A1_4, B1_4, C1_4, D1_4, 1, 1, ["Dirichlet"], [0])
        problem2_4 = Problem(1, A2_4, B2_4, C2_4, D2_4, 1, 1, ["Dirichlet"], [0])

        result = self.neighbour.find_overlap_points_rectangulars(problem1_4, problem2_4)
        self.assertEqual(result, (4, A1_4, D1_4), "The segments overlap.")


    # tests for find_indices_in_grid

    def test_find_indices_in_grid_intersec_upper(self):
        boundary_p1 = Point(3,0)
        boundary_p2 = Point(3,2)
        subboundary_p1 = Point(3,1)
        subboundary_p2 = Point(3,3)

        i1_expected = 2
        i2_expected = 6

        result = self.neighbour.find_indices_in_grid(boundary_p1, boundary_p2, subboundary_p1, subboundary_p2, 1/2)
        self.assertEqual(result, (i1_expected, i2_expected))

    def test_find_indices_in_grid_intersec_full_lower(self):
        boundary_p1 = Point(3,1)
        boundary_p2 = Point(3,3)
        subboundary_p1 = Point(3,0)
        subboundary_p2 = Point(3,3)

        i1_expected = 2
        i2_expected = 4

        result = self.neighbour.find_indices_in_grid(boundary_p1, boundary_p2, subboundary_p1, subboundary_p2, 1/2)
        self.assertEqual(result, (i1_expected, i2_expected))

    def test_find_indices_in_grid_intersec_middle(self):
        boundary_p1 = Point(2,0)
        boundary_p2 = Point(2,2)
        subboundary_p1 = Point(2,0.5)
        subboundary_p2 = Point(2,1.5)

        i1_expected = 1
        i2_expected = 3

        result = self.neighbour.find_indices_in_grid(boundary_p1, boundary_p2, subboundary_p1, subboundary_p2, 1/2)
        self.assertEqual(result, (i1_expected, i2_expected))
