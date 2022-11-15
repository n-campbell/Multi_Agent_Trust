import numpy as np 

class SingleIntegrator:
	def __init__(self, x0, r, G):
		self.X0 = x0
		self.X = x0
		self.G = G
		self.R = r

	def lyapunov(self,G):
		V = np.linalg.norm(self.X[-3:] -  self.G)**2
		dV_dx = 2 * (self.X[-3:] - self.G).T
		return V, dV_dx

	def sphere_barrier(self, r, XC):
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





