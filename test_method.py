import unittest
import numpy as np
from Problem import Problem
from Point import Point
from BoundaryCondtionsUpdate import calculate_Dirichlet_cond, calculate_Neumann_cond
from Neighbour import Neighbour
from Method import Method
from scipy.sparse import csr_matrix
import scipy as sp

# derive from unittest.TestCase
class TestIdentity(unittest.TestCase):

    # implementation of a setUp method to initialize objects that will be used across multiple of the tests
    def setUp(self):

        self.method = Method()
        
        self.A = Point(0,0)
        self.B = Point(2,0)
        self.C = Point(1,2)
        self.D = Point(0,1)

        self.delta_x = 1/4
        self.delta_y = 1/2

        self.boundary_cond_types = ['Dirichlet', 'Neumann', 'Dirichlet', 'Dirichlet']
        self.boundary_cond_values = [[15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15], [40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40]]

        self.nx = 9
        self.ny = 3

        self.problem = Problem(
            id=1,
            A=self.A,
            B=self.B,
            C=self.C,
            D=self.D,
            delta_x=self.delta_x,
            delta_y=self.delta_y,
            boundary_conditions_types=self.boundary_cond_types,
            boundary_conditions_values=self.boundary_cond_values
        )


    def test_compute_b(self):

        data_b_expected = [40, 15, 15, 15, 15, 15, 15, 15, 15, 40, 0, 15, 15, 15, 15, 15, 15, 15, 15, 15]
        rows_b_expected = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
        cols_b_expected = np.zeros((len(rows_b_expected)))

        b_sparse_expected = csr_matrix((data_b_expected, (rows_b_expected, cols_b_expected)), shape=(27,1))

        b_sparse = self.method.compute_b(self.problem)

        np.testing.assert_array_equal(b_sparse.toarray(), b_sparse_expected.toarray())