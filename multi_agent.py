
""" Code for 3D control problem with spherical obstacle and n agents"""
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from SingleIntegrator import SingleIntegrator as SI
from random import randint
from matplotlib.animation import FuncAnimation

# time info
T = 10
dt = 0.001
t_steps = int(T/dt)

# define cooperative robots
alpha_init = 1
c_robots = []
# c_robots.append(SI(np.array([[0],[0],[0]]), 1, np.array([[10],[10],[11]]),alpha_init))
# c_robots.append( SI(np.array([[10],[0],[0]]), 1, np.array([[0],[10],[11]]),alpha_init))
# c_robots.append( SI(np.array([[10],[3],[9]]), 1, np.array([[0],[3],[9]]),alpha_init))
# c_robots.append( SI(np.array([[10],[8],[10]]), 1, np.array([[0],[8],[10]]),alpha_init))
c_robots.append( SI(np.array([[0],[7],[5]]), 1, np.array([[10],[7],[5]]),alpha_init))
num_c_robots = len(c_robots)


# define uncooperative robots
uc_robots = []
# uc_robots.append( SI(np.array([[10],[10],[0]]), 1, np.array([[1],[0],[12]]),alpha_init))
uc_robots.append( SI(np.array([[6],[10],[5]]), 1, np.array([[6],[0],[5]]),alpha_init))
uc_robots.append( SI(np.array([[8],[10],[5]]), 1, np.array([[8],[0],[5]]),alpha_init))
# uc_robots.append( SI(np.array([[10],[5],[8]]), 1, np.array([[0],[5],[8]]),alpha_init))
# uc_robots.append( SI(np.array([[4],[0],[2]]), 1, np.array([[4],[10],[2]]),alpha_init))
num_uc_robots = len(uc_robots)

# define adveserial robots
a_robots = []
# a_robots.append( SI(np.array([[0],[0],[0]]), 1, uc_robots[0].X, alpha_init))
a_robots.append( SI(np.array([[0],[10],[5]]), 1, c_robots[0].X, alpha_init))
num_a_robots = len(a_robots)

# set up control for uncooperative agents
u1 = cp.Variable((3,1))
delta1 = cp.Variable(1)
V1 = cp.Parameter()
dV1_dx = cp.Parameter((1,3))
P = np.identity(3)

alpha1 = 15
k1 = 1

V1.value = uc_robots[0].lyapunov()[0] 
dV1_dx.value = uc_robots[0].lyapunov()[1]

objective1 = cp.Minimize(10 * cp.quad_form(u1,P) + 10 * cp.square(delta1))
constraint1 = [ ]
constraint1 += [ dV1_dx @ u1 <= - k1 * V1 + delta1 ] # CLF
prob1 = cp.Problem(objective1, constraint1)

# set up control for cooperative agents
u2 = cp.Variable((3,1))
delta2 = cp.Variable(1)
V2 = cp.Parameter()
dV2_dx = cp.Parameter((1,3))

h2_a = []
dhi2_dx_a = []
dhj2_dx_a = []
alpha2 = []
xj_dot = []

for i in range( num_uc_robots + num_c_robots + num_a_robots - 1 ):
	h2_a.append(cp.Parameter())
	dhi2_dx_a.append(cp.Parameter((1,3)))
	dhj2_dx_a.append(cp.Parameter((1,3)))
	alpha2.append(15) # alpha2.append(cp.Parameter())
	xj_dot.append(cp.Parameter((3,1)))

k2 = 3 

V2.value = c_robots[0].lyapunov()[0] 
dV2_dx.value = c_robots[0].lyapunov()[1] 


objective2 = cp.Minimize(10 * cp.quad_form(u2,P) + 10 * cp.square(delta2))
constraint2 = [ ]
constraint2 += [ dV2_dx @ u2 <= - k2 * V2 + delta2 ] # CLF

for i in range( num_uc_robots + num_c_robots - 1 ):
	constraint2 += [ dhi2_dx_a[i] @ u2 + dhj2_dx_a[i] @ xj_dot[i] >= - alpha2[i] * h2_a[i] ] # CBF for other agents

prob2 = cp.Problem(objective2, constraint2) 

# set up control for best case cooperative agent
# u_b = cp.Variable((3,1))
# Q = cp.Parameter((1,3))

# h_b = []
# dhi_dx_b = []
# dhj_dx_b = []
# alpha_b = []
# xj_dot_b = []

# for i in range( num_uc_robots + num_c_robots + num_a_robots - 1 ):
# 	h_b.append(cp.Parameter())
# 	dhi_dx_b.append(cp.Parameter((1,3)))
# 	dhj_dx_b.append(cp.Parameter((1,3)))
# 	alpha_b.append(cp.Parameter())
# 	xj_dot_b.append(cp.Parameter((3,1)))

# objective_b = cp.Maximize( Q @ u_b )
# constraint_b = [ ]

# for i in range( num_uc_robots + num_c_robots - 1 ):
# 	constraint_b += [ dhi_dx_b[i] @ u_b + dhj_dx_b[i] @ xj_dot_b[i] >= - alpha_b[i] * h_b[i] ] # CBF for other agents

# prob_b = cp.Problem(objective_b, constraint_b) 

# plot setup
plt.ion()
fig = plt.figure()
ax = plt.axes(projection='3d')

# update dynamics
for t in range( t_steps ):

	for i in range( num_uc_robots ):
		
		V1.value = uc_robots[i].lyapunov()[0] 
		dV1_dx.value = uc_robots[i].lyapunov()[1]

		prob1.solve() 

		X_next = uc_robots[i].X[-3:] + u1.value * dt
		uc_robots[i].X = np.concatenate((uc_robots[i].X, X_next), axis = 0)
		uc_robots[i].u = u1.value

	for i in range( num_a_robots ):

		a_robots[i].G = c_robots[0].X[-3:]
		
		V1.value = a_robots[i].lyapunov()[0] 
		dV1_dx.value = a_robots[i].lyapunov()[1]

		prob1.solve()

		X_next = a_robots[i].X[-3:] + u1.value * dt
		a_robots[i].X = np.concatenate((a_robots[i].X, X_next), axis = 0)
		a_robots[i].u = u1.value

	for j in range( num_c_robots ):

		V2.value = c_robots[j].lyapunov()[0] 
		dV2_dx.value = c_robots[j].lyapunov()[1] 

		n = 0 # n-th robot

		for k in range( num_uc_robots ):  # uj, dhj, dhi, hi
			h2_a[n].value = c_robots[j].agent_barrier(uc_robots[k].R, uc_robots[k].X)[0] 
			dhi2_dx_a[n].value = c_robots[j].agent_barrier(uc_robots[k].R, uc_robots[k].X)[1] 
			dhj2_dx_a[n].value = c_robots[j].agent_barrier(uc_robots[k].R, uc_robots[k].X)[2] 
			xj_dot[n].value = uc_robots[k].u

			# h_b = c_robots[j].agent_barrier(uc_robots[k].R, uc_robots[k].X)[0] 
			# dhi_dx_b[n].value = c_robots[j].agent_barrier(uc_robots[k].R, uc_robots[k].X)[1] 
			# dhj_dx_b[n].value = c_robots[j].agent_barrier(uc_robots[k].R, uc_robots[k].X)[2] 
			# xj_dot_b[n].value = uc_robots[k].u
			# Q.value = dhi_dx_b[n].value
			# prob_b.solve()

			# alpha2[n].value = c_robots[j].agent_alpha(uc_robots[k].u,u_b.value) 

			n = n + 1 

		for k in range( num_c_robots ):

			if k == j:
				continue

			h2_a[n].value = c_robots[j].agent_barrier(c_robots[k].R, c_robots[k].X)[0]  
			dhi2_dx_a[n].value = c_robots[j].agent_barrier(c_robots[k].R, c_robots[k].X)[1]
			dhj2_dx_a[n].value = c_robots[j].agent_barrier(c_robots[k].R, c_robots[k].X)[2]
			xj_dot[n].value = c_robots[k].u

			# h_b = c_robots[j].agent_barrier(c_robots[k].R, c_robots[k].X)[0]  
			# dhi_dx_b[n].value = c_robots[j].agent_barrier(c_robots[k].R, c_robots[k].X)[1]
			# dhj_dx_b[n].value = c_robots[j].agent_barrier(c_robots[k].R, c_robots[k].X)[2]
			# xj_dot_b[n].value = c_robots[k].u
			# Q.value = dhi_dx_b[n].value
			# prob_b.solve()

			# alpha2[n].value = c_robots[j].agent_alpha(c_robots[k].u, u_b.value) # alpha_dot
			
			n = n + 1

		for k in range( num_a_robots ):

			if k == j:
				continue

			h2_a[n].value = c_robots[j].agent_barrier(a_robots[k].R, a_robots[k].X)[0]  
			dhi2_dx_a[n].value = c_robots[j].agent_barrier(a_robots[k].R, a_robots[k].X)[1]
			dhj2_dx_a[n].value = c_robots[j].agent_barrier(a_robots[k].R, a_robots[k].X)[2]
			xj_dot[n].value = a_robots[k].u

			# h_b = c_robots[j].agent_barrier(a_robots[k].R, a_robots[k].X)[0]  
			# dhi_dx_b[n].value = c_robots[j].agent_barrier(a_robots[k].R, a_robots[k].X)[1]
			# dhj_dx_b[n].value = c_robots[j].agent_barrier(a_robots[k].R, a_robots[k].X)[2]
			# xj_dot_b[n].value = a_robots[k].u
			# Q.value = dhi_dx_b[n].value
			# prob_b.solve()
			# alpha2[n].value = c_robots[j].agent_alpha(a_robots[k].u,u_b.value) 

			n = n + 1

		prob2.solve()

		X_next = c_robots[j].X[-3:] + u2.value * dt
		c_robots[j].X = np.concatenate((c_robots[j].X, X_next), axis = 0)
		c_robots[j].u = u2.value



	for j in range( num_c_robots ):
		ax.scatter3D(c_robots[j].X[::3], c_robots[j].X[1::3] , c_robots[j].X[2::3], color = "green")

	for i in range( num_uc_robots ):
		ax.scatter3D(uc_robots[i].X[::3], uc_robots[i].X[1::3] , uc_robots[i].X[2::3], color = "blue")

	for i in range( num_a_robots ):
		ax.scatter3D(a_robots[i].X[::3], a_robots[i].X[1::3] , a_robots[i].X[2::3], color = "red")

	plt.xlabel("x")
	plt.ylabel("y")
	plt.draw()
	plt.pause(0.000001)
	ax.cla()
	# plt.show()

plt.ioff
plt.close('all')
plt.close(fig)
