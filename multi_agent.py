
""" Code for 3D control problem with spherical obstacle and n agents"""
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from SingleIntegrator import SingleIntegrator as SI
from random import randint
from matplotlib.animation import FuncAnimation
from numpy import linalg as LA


# time info
T = 10
dt = 0.01
t_steps = int(T/dt)

# define cooperative robots
c_robots = []
c_robots.append(SI(np.array([[0],[0],[0]]), 1, np.array([[10],[10],[11]])))
# c_robots.append( SI(np.array([[10],[0],[0]]), 1, np.array([[0],[10],[11]]),alpha_init))
# c_robots.append( SI(np.array([[10],[3],[9]]), 1, np.array([[0],[3],[9]]),alpha_init))
# c_robots.append( SI(np.array([[10],[8],[10]]), 1, np.array([[0],[8],[10]])))
c_robots.append( SI(np.array([[0],[7],[5]]), 1, np.array([[10],[7],[5]])))
num_c_robots = len(c_robots)
print(f'num_c: {num_c_robots}')


# define uncooperative robots
uc_robots = []
# uc_robots.append( SI(np.array([[10],[10],[0]]), 1, np.array([[1],[0],[12]]),alpha_init))
uc_robots.append( SI(np.array([[6],[10],[5]]), 1, np.array([[6],[0],[5]])))
uc_robots.append( SI(np.array([[8],[10],[5]]), 1, np.array([[8],[0],[5]])))
# uc_robots.append( SI(np.array([[10],[5],[8]]), 1, np.array([[0],[5],[8]]),alpha_init))
# uc_robots.append( SI(np.array([[4],[0],[2]]), 1, np.array([[4],[10],[2]]),alpha_init))
num_uc_robots = len(uc_robots)

# define adveserial robots
a_robots = []
# a_robots.append( SI(np.array([[0],[0],[0]]), 1, uc_robots[0].X, alpha_init))
a_robots.append( SI(np.array([[0],[10],[5]]), 1, c_robots[0].X))
num_a_robots = len(a_robots)

# initialize alphas for ego agents
num_robots = num_c_robots + num_uc_robots + num_a_robots
for i in range( num_c_robots ):
	c_robots[i].alpha = np.ones( num_robots - 1 )  * 2

# set up control for uncooperative agents
u1 = cp.Variable((3,1))
delta1 = cp.Variable(1)
V1 = cp.Parameter()
dV1_dx = cp.Parameter((1,3))
P = np.identity(3)

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
P = np.identity(3)

h2_a = []
dhi2_dx_a = []
dhj2_dx_a = []
alpha2 = []
xj_dot = []

for i in range( num_robots - 1 ):
	h2_a.append(cp.Parameter())
	dhi2_dx_a.append(cp.Parameter((1,3)))
	dhj2_dx_a.append(cp.Parameter((1,3)))
	alpha2.append(cp.Parameter())
	xj_dot.append(cp.Parameter((3,1)))

k2 = 3 

V2.value = c_robots[0].lyapunov()[0] 
dV2_dx.value = c_robots[0].lyapunov()[1] 

objective2 = cp.Minimize(10 * cp.quad_form(u2,P) + 10 * cp.square(delta2))
constraint2 = []
constraint2 += [ dV2_dx @ u2 <= - k2 * V2 + delta2 ] # CLF

input_bound = 10
for i in range(3):
	constraint2 += [ cp.abs(u2[i,0]) <= input_bound ]

for i in range( num_robots - 1 ):
	constraint2 += [ dhi2_dx_a[i] @ u2 + dhj2_dx_a[i] @ xj_dot[i] >= - alpha2[i] * h2_a[i] ] # CBF for other agents

prob2 = cp.Problem(objective2, constraint2) 

# set up control for best case cooperative agent
u_b = cp.Variable((3,1))
Q = cp.Parameter((1,3))

h_b = []
dhi_dx_b = []
dhj_dx_b = []
alpha_b = []
xj_dot_b = []

for i in range( num_robots - 1 ):
	h_b.append(cp.Parameter())
	dhi_dx_b.append(cp.Parameter((1,3)))
	dhj_dx_b.append(cp.Parameter((1,3)))
	alpha_b.append(cp.Parameter())
	xj_dot_b.append(cp.Parameter((3,1)))

objective_b = cp.Maximize( Q @ u_b )
constraint_b = [ ]

for i in range(3):
	constraint_b += [ cp.abs(u_b[i,0]) <= input_bound ]

for i in range( num_robots - 1 ):
 	constraint_b += [ dhi_dx_b[i] @ u_b + dhj_dx_b[i] @ xj_dot_b[i] >= - alpha_b[i] * h_b[i] ] # CBF for other agents

prob_b = cp.Problem(objective_b, constraint_b) 

# Trajectory plot setup
# plt.ion()
fig1 = plt.figure()
ax1 = plt.axes(projection='3d')

# initialize inter-agent distances
c_dists = np.zeros(( num_c_robots, t_steps, num_c_robots  - 1 ))
uc_dists = np.zeros(( num_c_robots, t_steps, num_uc_robots ))
a_dists = np.zeros(( num_c_robots, t_steps, num_a_robots ))

print(f'c_dists shape: {c_dists.shape}')

# initialize trust metrics
c_trust = np.zeros(( num_c_robots, t_steps, num_c_robots - 1 ))
uc_trust= np.zeros(( num_c_robots, t_steps, num_uc_robots ))
a_trust = np.zeros(( num_c_robots, t_steps, num_a_robots ))

# initialize trust metrics
c_alpha = np.zeros(( num_c_robots, t_steps, num_c_robots -1))
uc_alpha = np.zeros(( num_c_robots, t_steps, num_uc_robots ))
a_alpha = np.zeros(( num_c_robots, t_steps, num_a_robots ))

# update dynamics
for t in range( t_steps ):

	for i in range( num_uc_robots ):
		
		V1.value = uc_robots[i].lyapunov()[0] 
		dV1_dx.value = uc_robots[i].lyapunov()[1]

		prob1.solve(solver = cp.GUROBI) 

		X_next = uc_robots[i].X[-3:] + u1.value * dt
		uc_robots[i].X = np.concatenate((uc_robots[i].X, X_next), axis = 0)
		uc_robots[i].u = u1.value

	for i in range( num_a_robots ):

		a_robots[i].G = c_robots[0].X[-3:]
		
		V1.value = a_robots[i].lyapunov()[0] 
		dV1_dx.value = a_robots[i].lyapunov()[1]

		prob1.solve(solver = cp.GUROBI)

		X_next = a_robots[i].X[-3:] + u1.value * dt
		a_robots[i].X = np.concatenate((a_robots[i].X, X_next), axis = 0)
		a_robots[i].u = u1.value

	# loop through ego (cooperative) agents
	for j in range( num_c_robots ):
		c_robots[j].dV_dx = c_robots[j].lyapunov()[1]

	for j in range( num_c_robots ):

		V2.value, dV2_dx.value = c_robots[j].lyapunov()

		n = 0 # n-th robot

		# loop through uncooperative agents
		for k in range( num_uc_robots ): 

			# get constraints for cooperative controller
			h2_a[n].value, dhi2_dx_a[n].value, dhj2_dx_a[n].value = c_robots[j].agent_barrier(uc_robots[k].R, uc_robots[k].X)
			xj_dot[n].value = uc_robots[k].u
			alpha2[n].value = c_robots[j].alpha[n]

			# get constraints for best case controller
			h_b[n].value, dhi_dx_b[n].value,dhj_dx_b[n].value   = c_robots[j].agent_barrier(uc_robots[k].R, uc_robots[k].X)
			xj_dot_b[n].value = uc_robots[k].u 
			alpha_b[n].value = c_robots[j].alpha[n]

			# update inter-agent distance
			uc_dists[j][t][k] = LA.norm( c_robots[j].X[-3:]  - uc_robots[k].X[-3:]  )

			n = n + 1 

		# loop through (non-ego) cooperative agents
		for k in range( num_c_robots ):

			if k == j:
				continue

			# get constraints for cooperative controller
			h2_a[n].value, dhi2_dx_a[n].value, dhj2_dx_a[n].value  = c_robots[j].agent_barrier(c_robots[k].R, c_robots[k].X) 
			xj_dot[n].value = c_robots[k].u
			alpha2[n].value = c_robots[j].alpha[n]

			# get constraints for best case controller
			h_b[n].value, dhi_dx_b[n].value, dhj_dx_b[n].value = c_robots[j].agent_barrier(c_robots[k].R, c_robots[k].X)
			xj_dot_b[n].value = c_robots[k].u
			alpha_b[n].value = c_robots[j].alpha[n]

			# update inter-agent distance
			c_dists[j][t][k-1] = LA.norm( c_robots[j].X[-3:] - c_robots[k].X[-3:] )
			
			n = n + 1

		# loop through adveserial agents
		for k in range( num_a_robots ):

			# get constraints for cooperative controller
			h2_a[n].value, dhi2_dx_a[n].value, dhj2_dx_a[n].value  = c_robots[j].agent_barrier(a_robots[k].R, a_robots[k].X) 
			xj_dot[n].value = a_robots[k].u
			alpha2[n].value = c_robots[j].alpha[n]

			# get constraints for best case controller
			h_b[n].value, dhi_dx_b[n].value, dhj_dx_b[n].value  = c_robots[j].agent_barrier(a_robots[k].R, a_robots[k].X)
			xj_dot_b[n].value = a_robots[k].u
			alpha_b[n].value = c_robots[j].alpha[n]

			# update inter-agent distance
			a_dists[j][t][k-1] = LA.norm( c_robots[j].X[-3:]  - a_robots[k].X[-3:]  )

			n = n + 1

		# loop through all agents again and solve for trust metric
		n = 0
		for k in range( num_uc_robots ):
			Q.value = dhi_dx_b[n].value
			prob_b.solve(solver=cp.GUROBI, reoptimize=True)
			alpha_dot = c_robots[j].agent_alpha(uc_robots[k].u,u_b.value, uc_robots[k].dV_dx,n) # added
			alpha = c_robots[j].alpha[n] + alpha_dot * dt
			alpha2[n].value  = alpha
			c_robots[j].alpha[n] = alpha

			# save trust metric
			uc_trust[j][t][k] = alpha_dot
			uc_alpha[j][t][k] = alpha
			# print(f'alpha_dot for uc {k}: {alpha_dot}')

			n = n + 1

		for k in range( num_c_robots ):
			if k == j:
				continue
			Q.value = dhi_dx_b[n].value
			prob_b.solve(solver=cp.GUROBI,reoptimize=True )
			alpha_dot = c_robots[j].agent_alpha(c_robots[k].u, u_b.value, c_robots[k].dV_dx,n) # added
			alpha = c_robots[j].alpha[n] + alpha_dot * dt
			alpha2[n].value = alpha
			c_robots[j].alpha[n] = alpha

			# save trust metric
			c_trust[j][t][k-1] = alpha_dot
			c_alpha[j][t][k-1] = alpha

			n = n + 1

		for k in range( num_a_robots ):
			Q.value = dhi_dx_b[n].value
			prob_b.solve(solver=cp.GUROBI, reoptimize=True)
			alpha_dot = c_robots[j].agent_alpha(a_robots[k].u,u_b.value, a_robots[k].dV_dx, n) # added
			alpha = c_robots[j].alpha[n] + alpha_dot * dt
			alpha2[n].value = alpha
			c_robots[j].alpha[n] = alpha

			# save trust metric
			a_trust[j][t][k] = alpha_dot
			a_alpha[j][t][k] = alpha

			n = n + 1


		prob2.solve(solver = cp.GUROBI, reoptimize=True)
		if prob2.status!='optimal':
			print("Error: QP controller infeasible")
			exit()
		X_next = c_robots[j].X[-3:] + u2.value * dt 
		c_robots[j].X = np.concatenate((c_robots[j].X, X_next), axis = 0)
		c_robots[j].u = u2.value


	cc = np.linspace(100,1101, num = t_steps+1)

# plot trajectories for agents
for j in range( num_c_robots ):
	ax1.scatter3D(c_robots[j].X[::3], c_robots[j].X[1::3] , c_robots[j].X[2::3], color = 'green')#, c = cc, cmap = 'summer')

for i in range( num_uc_robots ):
	ax1.scatter3D(uc_robots[i].X[::3], uc_robots[i].X[1::3] , uc_robots[i].X[2::3],color = 'blue') #, c = cc, cmap = 'spring')

for i in range( num_a_robots ):
	ax1.scatter3D(a_robots[i].X[::3], a_robots[i].X[1::3] , a_robots[i].X[2::3], color = 'red') #, c = cc, cmap = 'autumn')

	# fig1.canvas.draw()
	# fig1.canvas.flush_events()
	# plt.pause(0.000001)
	# ax1.cla()

ax1.set_xlabel('X')
ax1.set_ylabel('Y')



# cbar = plt.colorbar()
# cbar.set_label('Color Intensity')
# plt.ioff


# plot inter-agent distances for robot 1 
fig2, ax2 = plt.subplots( 3 )
ax2[0].set_title(f"Distances Between Ego Agent 1 and Cooperative Agents")
legend = []
for k in range( num_c_robots ):
	ax2[0].plot(c_dists[0,:,k-1])
	legend.append(f'Cooperative Agent {k}')
ax2[0].legend(legend)
ax2[1].set_title(f"Distances Between Ego Agent 1 and Uncooperative Agents")
legdend = []
for k in range( num_uc_robots ):
	ax2[1].plot(uc_dists[0,:,k])
	legend.append(f'Unooperative Agent {k}')
ax2[1].legend(legend)
legend = []
ax2[2].set_title(f"Distances Between Ego Agent 1 and Adveserial Agents")
for k in range( num_a_robots ):
	ax2[2].plot(a_dists[0,:,k], label = f'Adveserial Agent {k}')
	legend.append(f'Adveserial Agent {k}')
ax2[2].legend(legend)

plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)

fig2.supxlabel('Time (s)')
fig2.supylabel('Inter-Agent Distance (meters)')
fig2.show()

# plot alpha dot for robot 1
fig3, ax3 = plt.subplots()
legend = []
for k in range( num_c_robots ):
	ax3.plot(c_trust[0,:,k-1])
	legend.append(f'Cooperative Agent {k}')
for k in range( num_uc_robots ):
	ax3.plot(uc_trust[0,:,k])
	legend.append(f'Uncooperative Agent {k}')
for k in range( num_a_robots ):
	ax3.plot(a_trust[0,:,k])
	legend.append(f'Adveserial Agent {k}')
ax3.legend(legend)
fig3.supxlabel('Time (s)')
fig3.supylabel('Alpha Dot')
fig3.show()

# plot alpha for robot 1
fig4, ax4 = plt.subplots()
legend = []
for k in range( num_c_robots ):
	ax4.plot(c_alpha[0,:,k-1])
	legend.append(f'Cooperative Agent {k}')
for k in range( num_uc_robots ):
	ax4.plot(uc_alpha[0,:,k])
	legend.append(f'Uncooperative Agent {k}')
for k in range( num_a_robots ):
	ax4.plot(a_alpha[0,:,k])
	legend.append(f'Adveserial Agent {k}')
ax4.legend(legend)
fig4.supxlabel('Time (s)')
fig4.supylabel('Alpha ')
fig4.show()

plt.show()