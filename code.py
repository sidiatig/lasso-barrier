import numpy as np


class QuadraticProblem:
    
    def __init__(self, Q, p, A, b):
        self.Q = Q
        self.p = p
        self.A = A
        self.b = b
        
    def is_feasible(self, v):
        return np.all((self.A.dot(v) - self.b) <= 0)
    
    def compute_obj(self, v):
        return v.dot(self.Q.dot(v)) + self.p.dot(v)
    
    def compute_grad(self, v):
        return 2 * self.Q.dot(v) + self.p
    
    def compute_hessian(self, v):
        return 2 * self.Q
    

class CenteringProblem:
    
    def __init__(self, problem, t):
        self.problem = problem
        self.A = problem.A
        self.b = problem.b
        self.t = t
        
    def is_feasible(self, v):
        return np.all((self.A.dot(v) - self.b) <= 0)
        
    def compute_obj(self, v):
        obj = self.problem.compute_obj(v)
        barrier = - np.sum(np.log(self.b - self.A.dot(v)))
        return self.t * obj + barrier
    
    def compute_grad(self, v):
        grad = self.problem.compute_grad(v)
        
        aux = 1 / (self.b - self.A.dot(v))
        barrier_grad = self.A.T.dot(aux)
        
        return self.t * grad + barrier_grad
    
    def compute_hessian(self, v):
        hessian = self.problem.compute_hessian(v)
        
        aux = 1 / (self.b - self.A.dot(v))
        barrier_hessian = self.A.T.dot(np.diag(aux) ** 2).dot(self.A)
        
        return self.t * hessian + barrier_hessian
    

def backtracking_line_search(problem, start_point, step, alpha, beta):
        step_size = 1.0
        start_obj = problem.compute_obj(start_point)
        start_grad = problem.compute_grad(start_point)
        search_point = start_point + step_size * step
        
        while not(problem.is_feasible(search_point)) or \
              problem.compute_obj(search_point) > start_obj + alpha*step_size * start_grad.dot(step):
            
            step_size *= beta
            search_point = start_point + step_size * step
            
        return step_size
    
    
def newton_method(problem, v0, eps):
    
    v_tab = [v0]
    convergence = False
    
    while not(convergence):
        v = v_tab[-1]

        grad = problem.compute_grad(v)
        hessian = problem.compute_hessian(v)

        nt_step = - np.linalg.inv(hessian).dot(grad)
        nt_decrement = np.dot(grad, np.linalg.inv(hessian).dot(grad))
        convergence = nt_decrement**2 / 2 <= eps

        if not(convergence):
            step_size = backtracking_line_search(problem, v, nt_step, 0.01, 0.5)
            v_tab.append(v + step_size * nt_step)

    return v_tab


def centering_step(problem, t, v0, eps):
    
    centering_problem = CenteringProblem(problem, t)
    v_tab = newton_method(centering_problem, v0, eps)
    
    return v_tab


def barr_method(problem, t0, v0, mu, eps, eps_centering):
    
    m = problem.A.shape[0]
    t = t0
    
    v_tab = []
    t_tab = [] # allows to keep track of the duality gap
    inner_steps_tab = [] # allows to keep track of the number of Newton steps
    
    # initial centering step
    centering_step_iterates = centering_step(problem, t, v0, eps_centering)
    
    v_tab.append(centering_step_iterates[-1])
    t_tab.append(t)
    inner_steps_tab.append(len(centering_step_iterates) - 1)
    
    while m/t >= eps:
        t *= mu
            
        centering_step_iterates = centering_step(problem, t, v_tab[-1], eps_centering)
        
        v_tab.append(centering_step_iterates[-1])
        t_tab.append(t)
        inner_steps_tab.append(inner_steps_tab[-1] + len(centering_step_iterates) - 1)
    
    return v_tab, t_tab, inner_steps_tab