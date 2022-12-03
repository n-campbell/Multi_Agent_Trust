from colorsys import hsv_to_rgb
import numpy as np 

class SingleIntegrator:
	def __init__(self, x0, r, G, alpha = 10, d_thresh = 1, u = np.array([[0],[0],[0]])):
		self.X0 = x0
		self.X = x0
		self.G = G
		self.R = r
		self.alpha = alpha
		self.d_thresh = d_thresh
		self.u = u

	def lyapunov(self):
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
		self.h = np.linalg.norm(self.X[-3:] - X[-3:])**2 - (self.R + r)**2
		self.dhi = 2 * ( self.X[-3:] - X[-3:] ).T # wrt to ego agent
		self.dhj = -2 * ( self.X[-3:] - X[-3:] ).T # wrt to other agent

		return self.h, self.dhi, self.dhj

	def agent_alpha(self, uj, ui_best): #, nomj):
		# uj: velocity input of other agent
		# dhj: dh_dx for other agent
		# dhi: dh_dx for ego agent
		# hi: h for ego agent
		# nomj: nominal direction of other agent
		# rho_d = np.tanh( self.dhj @ uj - ( -self.alpha * self.h - self.dhi @ ui_best) ) - np.tanh( self.d_thresh )
		# theta1 = np.acos( (dhi * uj) / (np.norm(dhi) * np.norm(uj)) ) #  angle between surface normal and actual agent trajectory
		# theta2 = np.acos( (dhi * nomj) / (np.norm(dhi) * np.norm(nomj)) ) #  angle between surface normal and nominal agent trajectory
		# rho_theta = np.tanh( theta2 / theta1 )
		# self.alpha = rho_d #* rho_theta
		# return rho_d[0][0]
		return 2





