from colorsys import hsv_to_rgb
import numpy as np 
from numpy import NAN, linalg as LA

class SingleIntegrator:
	def __init__(self, x0, r, G, alpha = 1, d_thresh = 0.7, u = np.array([[0],[0],[0]])):
		self.X0 = x0
		self.X = x0
		self.G = G
		self.R = r
		self.alpha = alpha
		self.d_thresh = d_thresh
		self.u = u

	def lyapunov(self):
		V = np.linalg.norm(self.X[-3:] -  self.G)**2
		self.dV_dx = 2 * (self.X[-3:] - self.G).T
		return V, self.dV_dx

	def agent_barrier(self, r, X):
		# r: radius of other agent
		# X: coordinates of other agent
		self.h = np.linalg.norm(self.X[-3:] - X[-3:])**2 - (self.R + r)**2
		self.dhi = 2 * ( self.X[-3:] - X[-3:] ).T # wrt to ego agent
		self.dhj = - 2 * ( self.X[-3:] - X[-3:] ).T # wrt to other agent

		return self.h, self.dhi, self.dhj

	def agent_alpha(self, uj, ui_best, dvj, n, print_stuff= False):
		# uj: velocity input of other agent
		# ui_best: velocity input for ego agent in best case
		# dvj: derivative of lyapunov fucntion of agent j 
		# n: nth robot for alpha
		
		rho_d = np.tanh( self.dhj @ uj - ( - self.alpha[n] * self.h - self.dhi @ ui_best) ) - np.tanh( self.d_thresh )
		nj = (- dvj / LA.norm(dvj)).reshape(-1,1)
		if LA.norm(uj) <= 0:
			rho_theta = [[0.0]]
		else:
			inner_term = (self.dhj @ uj) / (LA.norm(self.dhj) * LA.norm(uj))
			if inner_term < -1.0:
				inner_term = [[-1.0]]
			elif inner_term > 1.0:
				inner_term = [[1.0]]
			theta1 = np.arccos( inner_term ) #  angle between surface normal and actual agent trajectory

			if theta1 < 1E-20:
				rho_theta = [[0]]
			else:
				inner_term = (self.dhj @ nj) / (LA.norm(self.dhj) * LA.norm(nj)) 
				if inner_term < -1.0:
					inner_term = [[-1.0]]
				elif inner_term > 1.0:
					inner_term = [[1.0]]
				theta2 = np.arccos( inner_term ) #  angle between surface normal and nominal agent trajectory
				rho_theta = np.tanh( theta2 / theta1 )

		# if print_stuff:
		# 	print(f'agent n: {n}')
		# 	print(f'alpha_dot: {(rho_d[0][0] *  rho_theta[0][0])}')
		# 	print(f'rho_theta: {rho_theta}')
		# 	print(f'rho_d: {rho_d}')
		# # 	print(f'rho_theta: {rho_theta}')
		# # # 	print(f'theta2: {theta2} ')
		# # # 	print(f'inner_term: {inner_term}')


		return (rho_d[0][0] *  rho_theta[0][0])
		







