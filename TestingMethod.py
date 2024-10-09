import numpy as np
from Problem import Problem
from Point import Point
from Method import Method

p1 = Point(0,0)
p2 = Point(1,0)
p3 = Point(1,1)
p4 = Point(0,1)

boundary1 = [("Dirichlet",10,3)]
boundary2 = [("Dirichlet",15,3)]
boundary3 = [("Dirichlet",13,3)]
boundary4 = [("Dirichlet",20,3)]
bound_cond = [boundary1,boundary2,boundary3,boundary4]
step_size = 1/2
     
problem = Problem(p1,p2,p3,p4,step_size,step_size,bound_cond)

method = Method()

method.solve(problem)
     