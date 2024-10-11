import numpy as np
from scipy.sparse import csr_matrix
import scipy as sp

class Method:
    def __init__(self, problem):
        self.nx = int((problem.B.x - problem.A.x) / problem.delta_x) + 1  # plus one to account for the one missing element in this
        self.ny = int((problem.D.y - problem.A.y) / problem.delta_y) + 1

        self.h_x = problem.delta_x
        self.h_y = problem.delta_y

        self.n_total = self.nx * self.ny

        self.data_b = []
        self.rows_b = []
        self.cols_b = []

        self.rows_A = []
        self.cols_A = []
        self.data_A = []
        
        self.bound_con_lower = []
        self.bound_con_right = []
        self.bound_con_upper = []
        self.bound_con_left = []

        self.bound_value_lower = []
        self.bound_value_right = []
        self.bound_value_upper = []
        self.bound_value_left = []
        
        self.boundary_lower = set(i for i in range(self.nx))
        self.boundary_right = set((i + 1) * self.nx - 1 for i in range(self.ny))
        self.boundary_upper = set((self.ny - 1) * self.nx + i for i in range(self.nx))
        self.boundary_left = set(i * self.nx for i in range(self.ny))  # Correct boundary_left here

        # filling the boundary condition lists
        self.boundary_length_horizontal = problem.B.x - problem.A.x 
        for condition_type, value, length in problem.boundary_conditions[0]:
            # Convert length to number of grid points
            num_grid_points = int((length / self.boundary_length_horizontal) * self.nx)
            for i in range(num_grid_points):
                self.bound_con_lower.append(condition_type)
                self.bound_value_lower.append(value)

        
        self.boundary_length_vertical = problem.C.y - problem.B.y  
        for condition_type, value, length in problem.boundary_conditions[1]:
            # Convert length to number of grid points
            num_grid_points = int((length / self.boundary_length_vertical) * self.ny)
            for i in range(num_grid_points):
                self.bound_con_right.append(condition_type)
                self.bound_value_right.append(value)

        
        for condition_type, value, length in problem.boundary_conditions[2]:
            # Convert length to number of grid points
            num_grid_points = int((length / self.boundary_length_horizontal) * self.nx)
            for i in range(num_grid_points):
                self.bound_con_upper.append(condition_type)
                self.bound_value_upper.append(value)

        
        
        for condition_type, value, length in problem.boundary_conditions[3]:
            # Convert length to number of grid points
            num_grid_points = int((length / self.boundary_length_vertical) * self.ny)
            for i in range(num_grid_points):
                self.bound_con_left.append(condition_type)
                self.bound_value_left.append(value)


    def solve(self, problem):
        A = self.compute_A(problem)
        b = self.compute_b(problem)
        
        A_dense = A.toarray()
        b_dense = b.toarray()
        v = sp.sparse.linalg.spsolve(A,b)
        v_dense = v
        
        return v_dense

    def compute_b(self, problem):
        """
        Creates a sparse vector b representing the RHS of the discretized equation.
        
        For inside points, RHS = 0.
        For boundary values, RHS is determined by boundary conditions:
            - Gamma_H/Gamma_WF/Gamma_W for known boundaries
            - Gamma_i for the unknown boundary wall
            
        Parameters:
            problem (Problem): takes the parameters from the problem class

        Returns:
            b: A scipy sparse vector
        """
       
        # inner_boundary_lower = set(i + self.nx for i in range(1, self.nx - 1))
        # inner_boundary_right = set((i + 1) * self.nx - 2 for i in range(1, self.ny - 1))
        # inner_boundary_upper = set((self.ny - 2) * self.nx + i for i in range(1, self.nx - 1))
        # inner_boundary_left = set(i * self.nx + 1 for i in range(self.ny))  # Correct naming here

        # Computing all the different b's for the different boundaries + conditions
        for i in range(self.n_total):

            # Corner cases
            if i in self.boundary_lower and i in self.boundary_right:
                if self.bound_con_lower[-1] == 'Neumann' and self.bound_con_right[0]== 'Neumann':
                    self.data_b.append(self.bound_value_lower[-1]/self.h_y)
                
                elif self.bound_con_lower[-1] == 'Neumann' and self.bound_con_right[0]== 'Dirichlet':
                    self.data_b.append(self.bound_value_right[0])
                else:
                    self.data_b.append(self.bound_value_lower[-1])
                    
                self.rows_b.append(i)
                self.cols_b.append(0)
                continue

            elif i in self.boundary_right and i in self.boundary_upper:
                if self.bound_con_right[-1] == 'Neumann' and self.bound_con_upper[-1]== 'Neumann':
                    self.data_b.append(self.bound_value_right[-1]/self.h_x)
                elif self.bound_con_right[-1] == 'Neumann' and self.bound_con_upper[-1]== 'Dirichlet':
                    self.data_b.append(self.bound_value_upper[-1])
                else:
                     self.data_b.append(self.bound_value_right[-1])
                self.rows_b.append(i)
                self.cols_b.append(0)
                continue

            elif i in self.boundary_upper and i in self.boundary_left:
                if self.bound_con_upper[0] == 'Neumann' and self.bound_con_left[-1]== 'Neumann':
                    self.data_b.append(self.bound_value_upper[0]/self.h_y)
                elif self.bound_con_upper[-1] == 'Neumann' and self.bound_con_left[-1]== 'Dirichlet':
                    self.data_b.append(self.bound_value_left[-1])
                else:
                    self.data_b.append(self.bound_value_upper[0])
                self.rows_b.append(i)
                self.cols_b.append(0)
                continue

            elif i in self.boundary_left and i in self.boundary_lower:
                if self.bound_con_left[0] == 'Neumann' and self.bound_con_lower[0]== 'Neumann':
                    self.data_b.append(self.bound_value_left[0]/self.h_x)
                elif self.bound_con_left[0] == 'Neumann' and self.bound_con_lower[0]== 'Dirichlet':
                    self.data_b.append(self.bound_value_lower[0])
                else:
                    self.data_b.append(self.bound_value_left[0])
                self.rows_b.append(i)
                self.cols_b.append(0)
                continue

            # Bundaries
            elif i in self.boundary_lower:
                if self.bound_con_lower[i] == 'Neumann':
                    self.data_b.append(self.bound_value_lower[i]/self.h_y)
                else:
                    self.data_b.append(self.bound_value_lower[i])
                self.rows_b.append(i)
                self.cols_b.append(0)
                continue

            elif i in self.boundary_right:
                k = int(i / (self.nx))
                if self.bound_con_right[k] == 'Neumann':
                    self.data_b.append(self.bound_value_right[k]/self.h_x)
                else:
                    self.data_b.append(self.bound_value_right[k])
                self.rows_b.append(i)
                self.cols_b.append(0)
                continue
            
            elif i in self.boundary_upper:
                k = np.remainder(i, self.nx)
                if self.bound_con_upper[k] == 'Neumann':
                    self.data_b.append(self.bound_value_upper[k]/self.h_y)
                else:
                    self.data_b.append(self.bound_value_upper[k])
                self.rows_b.append(i)
                self.cols_b.append(0)
                continue

            elif i in self.boundary_left:
                k = int(i/self.nx)
                if self.bound_con_left[k] == 'Neumann':
                    self.data_b.append(self.bound_value_left[k]/self.h_x)
                else:
                    self.data_b.append(self.bound_value_left[k])
                self.rows_b.append(i)
                self.cols_b.append(0)
                continue
                
            # elif i in inner_boundary_lower:
            #     print(f"i={i} is in iner_boundary_lower")
            # elif i in inner_boundary_right:
            #     print(f"i={i} is in iner_boundary_right")
            # elif i in inner_boundary_upper:
            #     print(f"i={i} is in iner_boundary_upper")
            # elif i in inner_boundary_left:
            #     print(f"i={i} is in iner_boundary_left")


         # Create the sparse RHS vector
        
        b_sparse = csr_matrix((self.data_b, (self.rows_b, self.cols_b)), shape=(self.n_total, 1))
        return b_sparse
    

    def compute_A(self, problem):
        '''
        Should create a A matrix with the LHS of the discretized equation

        Parameters:
            boundary_condition: Tuple, should specify which bs correspond to which condition (Neumann, Dirichlet)

        Should use self.boundary_values to set the LHS for the unknown points on the wall Gamma_i depending on the condition
        Inner points should follow the approximation equation from the lecture
        For known boundary points set the point in the matrix A = 1 (no equation to compute needed as we already have the point given)
        '''
        for i in range(self.n_total):
            if i in self.boundary_lower and i in self.boundary_right:
                condition_type, value, length = problem.boundary_conditions[0][0]
                if condition_type == 'Neumann':
                    self.data_A.append(1)
                    self.rows_A.append(i)
                    self.cols_A.append(i)

                else:
                    self.data_A.append(1)
                    self.rows_A.append(i)
                    self.cols_A.append(i)
                continue

            elif i in self.boundary_right and i in self.boundary_upper:
                condition_type, value, length = problem.boundary_conditions[1][0]
                if condition_type == 'Neumann':
                    self.data_A.append(1)
                    self.rows_A.append(i)
                    self.cols_A.append(i)
                else:
                    self.data_A.append(1)
                    self.rows_A.append(i)
                    self.cols_A.append(i)
                continue

            elif i in self.boundary_upper and i in self.boundary_left:
                condition_type, value, length = problem.boundary_conditions[2][-1]
                if condition_type == 'Neumann':
                    self.data_A.append(1)
                    self.rows_A.append(i)
                    self.cols_A.append(i)
                else:
                    self.data_A.append(1)
                    self.rows_A.append(i)
                    self.cols_A.append(i)
                continue

            elif i in self.boundary_left and i in self.boundary_lower:
                condition_type, value, length = problem.boundary_conditions[3][-1]
                if condition_type == 'Neumann':
                    self.data_A.append(1)
                    self.rows_A.append(i)
                    self.cols_A.append(i)
                else:
                    self.data_A.append(1)
                    self.rows_A.append(i)
                    self.cols_A.append(i)
                continue

            # Bundaries
            elif i in self.boundary_lower:
                if self.bound_con_lower[i] == 'Neumann':
                    self.rows_A.append(i)
                    self.cols_A.append(i)  
                    self.data_A.append(- 1 / self.h_y**2 - 2 / self.h_x**2)  

                    # Left neighbor (v_{i-1,j})
                    self.rows_A.append(i)
                    self.cols_A.append(i - 1)  
                    self.data_A.append(1 / self.h_x**2)  

                    # Right neighbor (v_{i+1,j})
                    self.rows_A.append(i)
                    self.cols_A.append(i + 1)  
                    self.data_A.append(1 / self.h_x**2) 

                    # Top neighbor (v_{i,j+1})
                    self.rows_A.append(i)
                    self.cols_A.append(i + self.nx )  
                    self.data_A.append(1 / self.h_y**2)

                else:
                    self.data_A.append(1)
                    self.rows_A.append(i)
                    self.cols_A.append(i)
                continue

            elif i in self.boundary_right:
                k = int(i / (self.nx))
                if self.bound_con_right[k] == 'Neumann':
                    self.rows_A.append(i)
                    self.cols_A.append(i)  
                    self.data_A.append(2 / self.h_y**2 + 1 / self.h_x**2)  

                    # Left neighbor (v_{i-1,j})
                    self.rows_A.append(i)
                    self.cols_A.append(i - 1)  
                    self.data_A.append(-1 / self.h_x**2)  

                    # Upper neighbor (v_{i,j+1})
                    self.rows_A.append(i)
                    self.cols_A.append(i + self.nx)  
                    self.data_A.append(-1 / self.h_y**2)

                    # Bottom neighbor (v_{i,j-1})
                    self.rows_A.append(i)
                    self.cols_A.append(i - self.nx )  
                    self.data_A.append(-1 / self.h_y**2)

                else:
                    self.data_A.append(1)
                    self.rows_A.append(i)
                    self.cols_A.append(i)
                continue
            
            elif i in self.boundary_upper:
                k = np.remainder(i, self.nx)
                if self.bound_con_upper[k] == 'Neumann':
                    self.rows_A.append(i)
                    self.cols_A.append(i)  
                    self.data_A.append(1 / self.h_y**2 + 2 / self.h_x**2)  

                    # Left neighbor (v_{i-1,j})
                    self.rows_A.append(i)
                    self.cols_A.append(i - 1)  
                    self.data_A.append(-1 / self.h_x**2)  

                    # Right neighbor (v_{i+1,j})
                    self.rows_A.append(i)
                    self.cols_A.append(i + 1)  
                    self.data_A.append(-1 / self.h_x**2) 

                    # Bottom neighbor (v_{i,j-1})
                    self.rows_A.append(i)
                    self.cols_A.append(i - self.nx )  
                    self.data_A.append(-1 / self.h_y**2)


                else:
                    self.data_A.append(1)
                    self.rows_A.append(i)
                    self.cols_A.append(i)
                continue

            elif i in self.boundary_left:
                k = int(i/self.nx)
                if self.bound_con_left[k] == 'Neumann':
                    self.rows_A.append(i)
                    self.cols_A.append(i)  
                    self.data_A.append(-2 / self.h_y**2 - 1 / self.h_x**2)  

                    # Right neighbor (v_{i+1,j})
                    self.rows_A.append(i)
                    self.cols_A.append(i + 1)  
                    self.data_A.append(1 / self.h_x**2)    

                    # Upper neighbor (v_{i,j+1})
                    self.rows_A.append(i)
                    self.cols_A.append(i + self.nx)  
                    self.data_A.append(1 / self.h_y**2)

                    # Bottom neighbor (v_{i,j-1})
                    self.rows_A.append(i)
                    self.cols_A.append(i - self.nx )  
                    self.data_A.append(1 / self.h_y**2)
                else:
                    self.data_A.append(1)
                    self.rows_A.append(i)
                    self.cols_A.append(i)
                continue
            else:
                self.rows_A.append(i)
                self.cols_A.append(i)  
                self.data_A.append(-2 / self.h_x**2 - 2 / self.h_y**2)  

                # Left neighbor (v_{i-1,j})
                self.rows_A.append(i)
                self.cols_A.append(i - 1)  
                self.data_A.append(1 / self.h_x**2)  

                # Right neighbor (v_{i+1,j})
                self.rows_A.append(i)
                self.cols_A.append(i + 1)  
                self.data_A.append(1 / self.h_x**2) 

                # Bottom neighbor (v_{i,j-1})
                self.rows_A.append(i)
                self.cols_A.append(i - self.nx) 
                self.data_A.append(1 / self.h_y**2)  

                # Top neighbor (v_{i,j+1})
                self.rows_A.append(i)
                self.cols_A.append(i + self.nx )  
                self.data_A.append(1 / self.h_y**2)
        A = csr_matrix((self.data_A, (self.rows_A, self.cols_A)), shape=(self.n_total, self.n_total))
        return A