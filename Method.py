import numpy as np
from scipy.sparse import csr_matrix
import scipy as sp

class Method:

    def solve(self, problem):
        A = self.compute_A(problem)
        b = self.compute_b(problem)
        v = sp.sparse.linalg.spsolve(A,b)
        v_dense = v
        
        # a = A.todense()
        # print(a)
        # #B = b.todense()
        # print(b)
        
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
        bound_con_lower = problem.boundary_conditions_types[0]
        bound_con_right = problem.boundary_conditions_types[1]
        bound_con_upper = problem.boundary_conditions_types[2]
        bound_con_left = problem.boundary_conditions_types[3]

        bound_value_lower = problem.boundary_conditions_values[0]
        bound_value_right = problem.boundary_conditions_values[1]
        bound_value_upper = problem.boundary_conditions_values[2]
        bound_value_left = problem.boundary_conditions_values[3]
        
        nx = round((problem.B.x - problem.A.x) / problem.delta_x) + 1  # plus one to account for the one missing element in this
        ny = round((problem.D.y - problem.A.y) / problem.delta_y) + 1

        h_x = problem.delta_x
        h_y = problem.delta_y

        n_total = nx * ny

        data_b = []
        rows_b = []
        cols_b = []

        
        boundary_lower = set(i for i in range(nx))
        boundary_right = set((i + 1) * nx - 1 for i in range(ny))
        boundary_upper = set((ny - 1) * nx + i for i in range(nx))
        boundary_left = set(i * nx for i in range(ny))  # Correct boundary_left here

        # inner_boundary_lower = set(i + nx for i in range(1, nx - 1))
        # inner_boundary_right = set((i + 1) * nx - 2 for i in range(1, ny - 1))
        # inner_boundary_upper = set((ny - 2) * nx + i for i in range(1, nx - 1))
        # inner_boundary_left = set(i * nx + 1 for i in range(ny))  # Correct naming here

        # Computing all the different b's for the different boundaries + conditions
        for i in range(n_total):

            # Corner cases
            if i in boundary_lower and i in boundary_right:
                if bound_con_lower[-1] == 'Neumann' and bound_con_right[0]== 'Neumann':
                    data_b.append(bound_value_lower[-1]/h_y)
                
                elif bound_con_lower[-1] == 'Neumann' and bound_con_right[0]== 'Dirichlet':
                    data_b.append(bound_value_right[0])
                else:
                    data_b.append(bound_value_lower[-1])
                    
                rows_b.append(i)
                cols_b.append(0)
                continue

            elif i in boundary_right and i in boundary_upper:
                if bound_con_right[-1] == 'Neumann' and bound_con_upper[-1]== 'Neumann':
                    data_b.append(bound_value_right[-1]/h_x)
                elif bound_con_right[-1] == 'Neumann' and bound_con_upper[-1]== 'Dirichlet':
                    data_b.append(bound_value_upper[-1])
                else:
                     data_b.append(bound_value_right[-1])
                rows_b.append(i)
                cols_b.append(0)
                continue

            elif i in boundary_upper and i in boundary_left:
                if bound_con_upper[0] == 'Neumann' and bound_con_left[-1]== 'Neumann':
                    data_b.append(bound_value_upper[0]/h_y)
                elif bound_con_upper[-1] == 'Neumann' and bound_con_left[-1]== 'Dirichlet':
                    data_b.append(bound_value_left[-1])
                else:
                    data_b.append(bound_value_upper[0])
                rows_b.append(i)
                cols_b.append(0)
                continue

            elif i in boundary_left and i in boundary_lower:
                if bound_con_left[0] == 'Neumann' and bound_con_lower[0]== 'Neumann':
                    data_b.append(bound_value_left[0]/h_x)
                elif bound_con_left[0] == 'Neumann' and bound_con_lower[0]== 'Dirichlet':
                    data_b.append(bound_value_lower[0])
                else:
                    data_b.append(bound_value_left[0])
                rows_b.append(i)
                cols_b.append(0)
                continue

            # Bundaries
            elif i in boundary_lower:
                if bound_con_lower[i] == 'Neumann':
                    data_b.append(bound_value_lower[i]/h_y)
                else:
                    data_b.append(bound_value_lower[i])
                rows_b.append(i)
                cols_b.append(0)
                continue

            elif i in boundary_right:
                k = int(i / (nx))
                if bound_con_right[k] == 'Neumann':
                    data_b.append(bound_value_right[k]/h_x)
                else:
                    data_b.append(bound_value_right[k])
                rows_b.append(i)
                cols_b.append(0)
                continue
            
            elif i in boundary_upper:
                k = np.remainder(i, nx)
                if bound_con_upper[k] == 'Neumann':
                    data_b.append(bound_value_upper[k]/h_y)
                else:
                    data_b.append(bound_value_upper[k])
                rows_b.append(i)
                cols_b.append(0)
                continue

            elif i in boundary_left:
                k = int(i/nx)
                if bound_con_left[k] == 'Neumann':
                    data_b.append(bound_value_left[k]/h_x)
                else:
                    data_b.append(bound_value_left[k])
                rows_b.append(i)
                cols_b.append(0)
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
        
        b_sparse = csr_matrix((data_b, (rows_b, cols_b)), shape=(n_total, 1))
        return b_sparse
    
    def compute_A(self, problem):
        '''
        Should create a A matrix with the LHS of the discretized equation

        Parameters:
            boundary_condition: Tuple, should specify which bs correspond to which condition (Neumann, Dirichlet)

        Should use boundary_values to set the LHS for the unknown points on the wall Gamma_i depending on the condition
        Inner points should follow the approximation equation from the lecture
        For known boundary points set the point in the matrix A = 1 (no equation to compute needed as we already have the point given)
        '''
        nx = round((problem.B.x - problem.A.x) / problem.delta_x) + 1  # plus one to account for the one missing element in this
        ny = round((problem.D.y - problem.A.y) / problem.delta_y) + 1

        h_x = problem.delta_x
        h_y = problem.delta_y

        n_total = nx * ny

        rows_A = []
        cols_A = []
        data_A = []

        bound_con_lower = problem.boundary_conditions_types[0]
        bound_con_right = problem.boundary_conditions_types[1]
        bound_con_upper = problem.boundary_conditions_types[2]
        bound_con_left = problem.boundary_conditions_types[3]

        boundary_lower = set(i for i in range(nx))
        boundary_right = set((i + 1) * nx - 1 for i in range(ny))
        boundary_upper = set((ny - 1) * nx + i for i in range(nx))
        boundary_left = set(i * nx for i in range(ny))  # Correct boundary_left here

        for i in range(n_total):
            if i in boundary_lower and i in boundary_right:
                if bound_con_lower[0] == 'Neumann':
                    data_A.append(1)
                    rows_A.append(i)
                    cols_A.append(i)

                else:
                    data_A.append(1)
                    rows_A.append(i)
                    cols_A.append(i)
                continue

            elif i in boundary_right and i in boundary_upper:
                if bound_con_right[0] == 'Neumann':
                    data_A.append(1)
                    rows_A.append(i)
                    cols_A.append(i)
                else:
                    data_A.append(1)
                    rows_A.append(i)
                    cols_A.append(i)
                continue

            elif i in boundary_upper and i in boundary_left:
                if bound_con_upper[-1] == 'Neumann':
                    data_A.append(1)
                    rows_A.append(i)
                    cols_A.append(i)
                else:
                    data_A.append(1)
                    rows_A.append(i)
                    cols_A.append(i)
                continue

            elif i in boundary_left and i in boundary_lower:
                if bound_con_left[-1] == 'Neumann':
                    data_A.append(1)
                    rows_A.append(i)
                    cols_A.append(i)
                else:
                    data_A.append(1)
                    rows_A.append(i)
                    cols_A.append(i)
                continue

            # Bundaries
            elif i in boundary_lower:
                if bound_con_lower[i] == 'Neumann':
                    rows_A.append(i)
                    cols_A.append(i)  
                    data_A.append(- 1 / h_y**2 - 2 / h_x**2)  

                    # Left neighbor (v_{i-1,j})
                    rows_A.append(i)
                    cols_A.append(i - 1)  
                    data_A.append(1 / h_x**2)  

                    # Right neighbor (v_{i+1,j})
                    rows_A.append(i)
                    cols_A.append(i + 1)  
                    data_A.append(1 / h_x**2) 

                    # Top neighbor (v_{i,j+1})
                    rows_A.append(i)
                    cols_A.append(i + nx )  
                    data_A.append(1 / h_y**2)

                else:
                    data_A.append(1)
                    rows_A.append(i)
                    cols_A.append(i)
                continue

            elif i in boundary_right:
                k = round(i / (nx))
                if bound_con_right[k] == 'Neumann':
                    rows_A.append(i)
                    cols_A.append(i)  
                    data_A.append(2 / h_y**2 + 1 / h_x**2)  

                    # Left neighbor (v_{i-1,j})
                    rows_A.append(i)
                    cols_A.append(i - 1)  
                    data_A.append(-1 / h_x**2)  

                    # Upper neighbor (v_{i,j+1})
                    rows_A.append(i)
                    cols_A.append(i + nx)  
                    data_A.append(-1 / h_y**2)

                    # Bottom neighbor (v_{i,j-1})
                    rows_A.append(i)
                    cols_A.append(i - nx )  
                    data_A.append(-1 / h_y**2)

                else:
                    data_A.append(1)
                    rows_A.append(i)
                    cols_A.append(i)
                continue
            
            elif i in boundary_upper:
                k = np.remainder(i, nx)
                if bound_con_upper[k] == 'Neumann':
                    rows_A.append(i)
                    cols_A.append(i)  
                    data_A.append(1 / h_y**2 + 2 / h_x**2)  

                    # Left neighbor (v_{i-1,j})
                    rows_A.append(i)
                    cols_A.append(i - 1)  
                    data_A.append(-1 / h_x**2)  

                    # Right neighbor (v_{i+1,j})
                    rows_A.append(i)
                    cols_A.append(i + 1)  
                    data_A.append(-1 / h_x**2) 

                    # Bottom neighbor (v_{i,j-1})
                    rows_A.append(i)
                    cols_A.append(i - nx )  
                    data_A.append(-1 / h_y**2)


                else:
                    data_A.append(1)
                    rows_A.append(i)
                    cols_A.append(i)
                continue

            elif i in boundary_left:
                k = round(i/nx)
                if bound_con_left[k] == 'Neumann':
                    rows_A.append(i)
                    cols_A.append(i)  
                    data_A.append(-2 / h_y**2 - 1 / h_x**2)  

                    # Right neighbor (v_{i+1,j})
                    rows_A.append(i)
                    cols_A.append(i + 1)  
                    data_A.append(1 / h_x**2)    

                    # Upper neighbor (v_{i,j+1})
                    rows_A.append(i)
                    cols_A.append(i + nx)  
                    data_A.append(1 / h_y**2)

                    # Bottom neighbor (v_{i,j-1})
                    rows_A.append(i)
                    cols_A.append(i - nx )  
                    data_A.append(1 / h_y**2)
                else:
                    data_A.append(1)
                    rows_A.append(i)
                    cols_A.append(i)
                continue
            else:
                rows_A.append(i)
                cols_A.append(i)  
                data_A.append(-2 / h_x**2 - 2 / h_y**2)  

                # Left neighbor (v_{i-1,j})
                rows_A.append(i)
                cols_A.append(i - 1)  
                data_A.append(1 / h_x**2)  

                # Right neighbor (v_{i+1,j})
                rows_A.append(i)
                cols_A.append(i + 1)  
                data_A.append(1 / h_x**2) 

                # Bottom neighbor (v_{i,j-1})
                rows_A.append(i)
                cols_A.append(i - nx) 
                data_A.append(1 / h_y**2)  

                # Top neighbor (v_{i,j+1})
                rows_A.append(i)
                cols_A.append(i + nx )  
                data_A.append(1 / h_y**2)
        A = csr_matrix((data_A, (rows_A, cols_A)), shape=(n_total, n_total))
        return A