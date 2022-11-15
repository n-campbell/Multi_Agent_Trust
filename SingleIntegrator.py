import numpy as np 

class SingleIntegrator:
	def __init__(self, x0, r, G):
		self.X0 = x0
		self.X = x0
		self.G = G
		self.R = r
		# self.U = np.array([0,0]).reshape(-1,1)
	
	# def f(self):
	# 	return np.array([0,0]).reshape(-1,1)

	# def g(self):
	# 	return np.array([[1,0], [0,1]])

	# def step(self,U):
	# 	self.U = U.reshape(-1,1)
	# 	self.X = self.X + (self.f() + self.g() @self.U)*self.dt
	# 	return self.X

	# def render_plot(self):
	# 	self.body.set_offsets(self.X[0,0],self.X[1,0])

	def lyapunov(self,G):
		V = np.linalg.norm(self.X[-3:] -  G)**2
		dV_dx = 2 * (self.X[-3:] - G).T
		return V, dV_dx

	def circle_barrier(self, r, XC):
		# r: obstacle radius
		# XC: center of obstacle
		h = np.linalg.norm(self.X[-3:] - XC)**2 - (self.R + r)**2
		dh_dx = 2 * (self.X[-3:]- XC).T
		return h, dh_dx

	def agent_barrier(self, r, X):
		# r: radius of other agent
		# X: coordinates of other agent
		h = np.linalg.norm(self.X[-3:] - X[-3:])**2 - (self.R + r)**2
		dh = 2 * (self.X[-3:] - X[-3:]).T
		return h, dh





