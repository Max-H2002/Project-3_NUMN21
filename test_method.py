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

        #self.delta_x = 1/4
        self.delta_x = 1/3
        self.delta_y = 1/2

        self.boundary_cond_types = [['Dirichlet', 'Dirichlet', 'Dirichlet', 'Dirichlet', 'Dirichlet', 'Dirichlet', 'Dirichlet', 'Dirichlet'], ['Neumann', 'Neumann', 'Neumann', 'Neumann', 'Neumann', 'Neumann', 'Neumann', 'Neumann', 'Neumann', 'Neumann'], ['Dirichlet', 'Dirichlet', 'Dirichlet', 'Dirichlet', 'Dirichlet', 'Dirichlet', 'Dirichlet', 'Dirichlet'], ['Dirichlet', 'Dirichlet', 'Dirichlet', 'Dirichlet', 'Dirichlet', 'Dirichlet', 'Dirichlet', 'Dirichlet']]
        self.boundary_cond_values = [[15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15], [40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40]]

        #self.nx = 9
        self.nx = 7
        self.ny = 3

        self.n_total = self.nx * self.ny

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

        #data_b_expected = [40, 15, 15, 15, 15, 15, 15, 15, 15, 40, 0, 15, 15, 15, 15, 15, 15, 15, 15, 15]
        #rows_b_expected = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
        #cols_b_expected = np.zeros((len(rows_b_expected)))

        #b_sparse_expected = csr_matrix((data_b_expected, (rows_b_expected, cols_b_expected)), shape=(27,1))

        data_b_expected = [40, 15, 15, 15, 15, 15, 15, 40, 0, 15, 15, 15, 15, 15, 15, 15]
        rows_b_expected = [0, 1, 2, 3, 4, 5, 6, 7, 13, 14, 15, 16, 17, 18, 19, 20]
        cols_b_expected = np.zeros((len(rows_b_expected)))

        b_sparse_expected = csr_matrix((data_b_expected, (rows_b_expected, cols_b_expected)), shape=(self.n_total,1))

        b_sparse = self.method.compute_b(self.problem)

        np.testing.assert_array_equal(b_sparse.toarray(), b_sparse_expected.toarray())


    def test_compute_A(self):

        data_A_expected = [1, 1, 1, 1, 1, 1, 1, 1, -26, 9, 9, 4, 4, -26, 9, 9, 4, 4, -26, 9, 9, 4, 4, -26, 9, 9, 4, 4, -26, 9, 9, 4, 4, 17, -9, -4, -4, 1, 1, 1, 1, 1, 1, 1]
        rows_A_expected = [0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 15, 16, 17, 18, 19, 20]
        cols_A_expected = [0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 9, 1, 15, 9, 8, 10, 2, 16, 10, 9, 11, 3, 17, 11, 10, 12, 4, 18, 12, 11, 13, 5, 19, 13, 12, 20, 6, 14, 15, 16, 17, 18, 19, 20]

        A_sparse_expected = csr_matrix((data_A_expected, (rows_A_expected, cols_A_expected)), shape=(self.n_total,self.n_total))

        A_sparse = self.method.compute_A(self.problem)

        np.testing.assert_array_equal(A_sparse.toarray(), A_sparse_expected.toarray())
