
""" Code for 3D control problem with spherical obstacle and n agents"""
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from SingleIntegrator import SingleIntegrator as SI
from random import randint
from matplotlib.animation import FuncAnimation
from numpy import True_, linalg as LA
import matplotlib as mpl


# time info
T = 10
dt = 0.01
t_steps = int(T/dt)


# define uncooperative robots
uc_robots = []
uc_robots.append( SI(np.array([[3],[10],[3]]), 1, np.array([[3],[0],[3]])))
uc_robots.append( SI(np.array([[6],[10],[3]]), 1, np.array([[6],[0],[3]])))
uc_robots.append( SI(np.array([[9],[10],[3]]), 1, np.array([[9],[0],[3]])))
num_uc_robots = len(uc_robots)

# define cooperative robots
c_robots = []
c_robots.append( SI(np.array([[0],[2],[0]]), 1, np.array([[10],[2],[5]])))
c_robots.append(SI(np.array([[0],[7],[0]]), 1, np.array([[10],[7],[5]])))
c_robots.append(SI(np.array([[0],[5],[2]]), 1, np.array([[10],[5],[5]])))
num_c_robots = len(c_robots)

# define adveserial robots
a_robots = []
a_robots.append( SI(np.array([[0],[10],[0]]), 1, c_robots[0].X))
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

input_bound = 0.5
for i in range(3):
	constraint1 += [ cp.abs(u1[i,0]) <= input_bound ]

prob1 = cp.Problem(objective1, constraint1)

# set up control for cooperative agents
u2 = cp.Variable((3,1))
delta2 = cp.Variable(1)
V2 = cp.Parameter()
dV2_dx = cp.Parameter((1,3))
P = np.identity(3)

learning_rate = 1

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
R = 0.3
u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
x = R * np.cos(u)*np.sin(v)
y = R * np.sin(u)*np.sin(v)
z = R * np.cos(v)

# plot start and goal markers for ego agents
for i in range(num_c_robots):
	ax1.plot_surface(x + c_robots[i].X[0], y + c_robots[i].X[1], z + c_robots[i].X[2], color = 'yellow')
	ax1.plot_surface(x + c_robots[i].G[0], y + c_robots[i].G[1], z + c_robots[i].G[2], color = 'green')

# initialize inter-agent distances
c_dists = np.zeros(( num_c_robots, t_steps, num_c_robots ))
uc_dists = np.zeros(( num_c_robots, t_steps, num_uc_robots ))
a_dists = np.zeros(( num_c_robots, t_steps, num_a_robots ))

print(f'c_dists shape: {c_dists.shape}')

# initialize trust metrics
c_trust = np.zeros(( num_c_robots, t_steps, num_c_robots ))
uc_trust= np.zeros(( num_c_robots, t_steps, num_uc_robots ))
a_trust = np.zeros(( num_c_robots, t_steps, num_a_robots ))

# initialize trust metrics
c_alpha = np.zeros(( num_c_robots, t_steps, num_c_robots ))
uc_alpha = np.zeros(( num_c_robots, t_steps, num_uc_robots ))
a_alpha = np.zeros(( num_c_robots, t_steps, num_a_robots ))

# initialize trust metrics
c_h = np.zeros(( num_c_robots, t_steps, num_c_robots ))
uc_h = np.zeros(( num_c_robots, t_steps, num_uc_robots ))
a_h= np.zeros(( num_c_robots, t_steps, num_a_robots ))

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

			# save inter-agent distance
			uc_dists[j][t][k] = LA.norm( c_robots[j].X[-3:]  - uc_robots[k].X[-3:]  )

			# save cbf
			uc_h[j][t][k] = h2_a[n].value

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

			# save inter-agent distance
			c_dists[j][t][k] = LA.norm( c_robots[j].X[-3:] - c_robots[k].X[-3:] )

			# save cbf
			c_h[j][t][k] = h2_a[n].value
			
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
			a_dists[j][t][k] = LA.norm( c_robots[j].X[-3:]  - a_robots[k].X[-3:]  )

			# save cbf
			a_h[j][t][k] = h2_a[n].value

			n = n + 1

		# loop through all agents again and solve for trust metric
		n = 0
		for k in range( num_uc_robots ):
			Q.value = dhi_dx_b[n].value
			prob_b.solve(solver=cp.GUROBI, reoptimize=True)
			if prob_b.status!='optimal':
				print("Error: LP controller infeasible")
				exit()
			alpha_dot = c_robots[j].agent_alpha(uc_robots[k].u,u_b.value, uc_robots[k].dV_dx,n, print_stuff= False) 
			if alpha_dot == 0.0:
				print(f'alpha dot for uc_robots:{alpha_dot}')
			alpha = c_robots[j].alpha[n] + learning_rate * alpha_dot * dt
			alpha2[n].value  = alpha
			c_robots[j].alpha[n] = alpha

			# save trust metric
			uc_trust[j][t][k] = alpha_dot
			uc_alpha[j][t][k] = alpha

			n = n + 1

		for k in range( num_c_robots ):
			if k == j:
				continue
			Q.value = dhi_dx_b[n].value
			prob_b.solve(solver=cp.GUROBI,reoptimize=True )
			if prob_b.status!='optimal':
				print("Error: LP controller infeasible")
				exit()
			alpha_dot = c_robots[j].agent_alpha(c_robots[k].u, u_b.value, c_robots[k].dV_dx,n, print_stuff= True)
			if alpha_dot == 0.0:
				print(f'alpha dot for c_robots:{alpha_dot}')
			alpha = c_robots[j].alpha[n] + learning_rate * alpha_dot * dt
			alpha2[n].value = alpha
			c_robots[j].alpha[n] = alpha

			# save trust metric
			c_trust[j][t][k] = alpha_dot
			c_alpha[j][t][k] = alpha

			n = n + 1

		for k in range( num_a_robots ):
			Q.value = dhi_dx_b[n].value
			prob_b.solve(solver=cp.GUROBI, reoptimize=True)
			if prob_b.status!='optimal':
				print("Error: LP controller infeasible")
				exit()
			alpha_dot = c_robots[j].agent_alpha(a_robots[k].u,u_b.value, a_robots[k].dV_dx, n) 
			alpha = c_robots[j].alpha[n] + learning_rate * alpha_dot * dt
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

		# # plot trajectories for agents LIVE
		# for j in range( num_c_robots ):
		# 	ax1.scatter3D(c_robots[j].XX[-3:] , c_robots[j].X[1::3] , c_robots[j].X[2::3], color = 'green')

		# for i in range( num_uc_robots ):
		# 	ax1.scatter3D(uc_robots[i].X[::3], uc_robots[i].X[1::3] , uc_robots[i].X[2::3],color = 'blue') 

		# for i in range( num_a_robots ):
		# 	ax1.scatter3D(a_robots[i].X[::3], a_robots[i].X[1::3] , a_robots[i].X[2::3], color = 'red')

		# fig1.canvas.draw()
		# fig1.canvas.flush_events()
		# plt.pause(0.000001)
		# ax1.cla()
	
	print(f'At Timestep: {t}/{t_steps}')

# plot trajectories for agents with colormap
cc = np.linspace(0,1001, num = t_steps+1)
ccmap1 = plt.get_cmap('winter_r', t_steps)
ccmap2 = plt.get_cmap('cool_r', t_steps)
ccmap3 = plt.get_cmap('autumn_r', t_steps)

for i in range( num_uc_robots ):
	ax1.scatter3D(uc_robots[i].X[::3], uc_robots[i].X[1::3] , uc_robots[i].X[2::3], c = cc, cmap = 'cool_r')

for j in range( num_c_robots ):
	ax1.scatter3D(c_robots[j].X[::3], c_robots[j].X[1::3] , c_robots[j].X[2::3], c = cc, cmap = 'winter_r')

for i in range( num_a_robots ):
	ax1.scatter3D(a_robots[i].X[::3], a_robots[i].X[1::3] , a_robots[i].X[2::3], c = cc, cmap = 'autumn_r')

ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

# Normalizer
norm = mpl.colors.Normalize(vmin=0, vmax=t_steps)
  
# creating ScalarMappabler
sm1 = plt.cm.ScalarMappable(cmap=ccmap1 ,norm=norm)
sm1.set_array([])
sm2 = plt.cm.ScalarMappable(cmap=ccmap2 ,norm=norm)
sm2.set_array([])
sm3 = plt.cm.ScalarMappable(cmap=ccmap3 ,norm=norm)
sm3.set_array([])
  
plt.colorbar(sm1, label = '(Ego) Cooperative')
plt.colorbar(sm2, label = 'Uncooperative')
plt.colorbar(sm3, label = 'Adveserial')

# plt.ioff

# plot inter-agent distances for ego agent
def plot_dists(fig, ax, agent_num):
	legend = []
	for k in range( num_c_robots ):
		if k == agent_num:
			continue
		ax.plot(c_dists[agent_num,:,k])
		legend.append(f'Cooperative Agent {k}')
	for k in range( num_uc_robots ):
		ax.plot(uc_dists[agent_num,:,k])
		legend.append(f'Uncooperative Agent {k}')
	for k in range( num_a_robots ):
		ax.plot(a_dists[agent_num,:,k])
		legend.append(f'Adveserial Agent {k}')
	ax.legend(legend)
	ax.set_title(f"Inter-Agent Distances for Ego Agent {agent_num}")
	fig.supxlabel('Timestep')
	fig.supylabel('Distance (m)')
	return ax


# plot inter-agent distances for ego agent 0
fig2, ax2 = plt.subplots()
plot_dists(fig2, ax2, 0)

# # plot inter-agent distances for ego agent 1
# fig3, ax3 = plt.subplots()
# plot_dists(fig3, ax3, 1)

# # plot inter-agent distances for ego agent 2
# fig4, ax4 = plt.subplots()
# plot_dists(fig4, ax4, 2)

# plot h for ego agents
def plot_h(fig, ax, agent_num):
	legend = []
	for k in range( num_uc_robots ):
		ax.plot(uc_h[agent_num,:,k])
		legend.append(f'Uncooperative Agent {k}')
	for k in range( num_c_robots ):
		if k == agent_num:
			continue
		ax.plot(c_h[agent_num,:,k])
		legend.append(f'Cooperative Agent {k}')
	for k in range( num_a_robots ):
		ax.plot(a_h[agent_num,:,k])
		legend.append(f'Adveserial Agent {k}')
	ax.legend(legend)
	ax.set_title(f"CBFs for Ego Agent {agent_num}")
	fig.supxlabel('Timestep')
	fig.supylabel('CBF')
	return ax


# plot cbf for ego agent 0
fig5, ax5 = plt.subplots()
plot_h(fig5, ax5, 0)

# # plot cbf for ego agent 1
# fig6, ax6 = plt.subplots()
# plot_h(fig6, ax6, 1)

# # plot cbf for ego agent 2
# fig7, ax7 = plt.subplots()
# plot_h(fig7, ax7, 2)

# plot alpha for ego agents
def plot_alpha(fig, ax, agent_num):
	legend = []
	for k in range( num_uc_robots ):
		ax.plot(uc_alpha[agent_num,:,k])
		legend.append(f'Uncooperative Agent {k}')
	for k in range( num_c_robots ):
		if k == agent_num:
			continue
		ax.plot(c_alpha[agent_num,:,k])
		legend.append(f'Cooperative Agent {k}')
	for k in range( num_a_robots ):
		ax.plot(a_alpha[agent_num,:,k])
		legend.append(f'Adveserial Agent {k}')
	ax.legend(legend)
	ax.set_title(f"Alphas for Ego Agent {agent_num}")
	fig.supxlabel('Timestep')
	fig.supylabel('Alpha ')
	return ax

# plot alpha for ego agent 0
fig8, ax8 = plt.subplots()
plot_alpha(fig8, ax8, 0)

# # plot alpha for ego agent 1
# fig9, ax9 = plt.subplots()
# plot_h(fig9, ax9, 1)

# # plot alpha for ego agent 2
# fig10, ax10 = plt.subplots()
# plot_dists(fig10, ax10, 2)

# plot alpha for ego agents
def plot_alpha_dot(fig, ax, agent_num):
	legend = []
	for k in range( num_uc_robots ):
		ax.plot(uc_trust[agent_num,:,k])
		legend.append(f'Uncooperative Agent {k}')
	for k in range( num_c_robots ):
		if k == agent_num:
			continue
		ax.plot(c_trust[agent_num,:,k])
		legend.append(f'Cooperative Agent {k}')
	for k in range( num_a_robots ):
		ax.plot(a_trust[agent_num,:,k])
		legend.append(f'Adveserial Agent {k}')
	ax.legend(legend)
	ax.set_title(f"Alpha_dot for Ego Agent {agent_num}")
	fig.supxlabel('Timestep')
	fig.supylabel('Alpha_dot ')
	return ax

# plot alpha dot for ego agent 0
fig11, ax11 = plt.subplots()
plot_alpha_dot(fig11, ax11, 0)

plt.show()