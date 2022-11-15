
""" Code for 3D control problem with spherical obstacle and n agents"""
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from SingleIntegrator import SingleIntegrator as SI

# time info
T = 10
dt = 0.01
t_steps = int(T/dt)

# environment info 
XC = np.array([[5],[5],[5]]) # obstacle center
R = 1  # obstacle radius

# define cooperative robots
c_robots = []
num_c_robots = 5
c_robots.append( SI(np.array([[0],[0],[0]]), 1, np.array([[10],[10],[11]])) )
c_robots.append( SI(np.array([[10],[0],[0]]), 1, np.array([[0],[10],[11]]))  )
c_robots.append( SI(np.array([[10],[3],[9]]), 1, np.array([[0],[3],[9]]))  )
c_robots.append( SI(np.array([[10],[8],[10]]), 1, np.array([[0],[8],[10]]))  )
c_robots.append( SI(np.array([[0],[4],[2]]), 1, np.array([[10],[4],[2]]) )  )

# define uncooperative robots
uc_robots = []
num_uc_robots = 5
uc_robots.append( SI(np.array([[10],[10],[0]]), 1, np.array([[1],[0],[12]]) ) )
uc_robots.append( SI(np.array([[2],[10],[5]]), 1, np.array([[2],[0],[5]]) )  )
uc_robots.append( SI(np.array([[8],[10],[5]]), 1, np.array([[8],[0],[5]]) )  )
uc_robots.append( SI(np.array([[10],[5],[8]]), 1, np.array([[0],[5],[8]]) )  )
uc_robots.append( SI(np.array([[4],[0],[2]]), 1, np.array([[4],[10],[2]]) )  )

# set up control for uncooperative agents
u1 = cp.Variable((3,1))
delta1 = cp.Variable(1)
V1 = cp.Parameter()
dV1_dx = cp.Parameter((1,3))
h1 = cp.Parameter()
dh1_dx = cp.Parameter((1,3))
P = np.identity(3)

alpha1 = 10
k1 = 1

V1.value = uc_robots[0].lyapunov()[0] 
dV1_dx.value = uc_robots[0].lyapunov()[1]

h1.value = uc_robots[0].sphere_obstacle(R, XC)[0]
dh1_dx.value = uc_robots[0].sphere_obstacle(R, XC)[1]

objective1 = cp.Minimize(10 * cp.quad_form(u1,P) + 10 * cp.square(delta1))
constraint1 = [ ]
constraint1 += [ dV1_dx @ u1 <= - k1 * V1 + delta1 ] # CLF
constraint1 += [ dh1_dx @ u1 >= - alpha1 * h1 ] # CBF for obstacle
prob1 = cp.Problem(objective1, constraint1)

# set up control for cooperative agents
u2 = cp.Variable((3,1))
delta2 = cp.Variable(1)
V2 = cp.Parameter()
dV2_dx = cp.Parameter((1,3))
h2 = cp.Parameter()
dh2_dx = cp.Parameter((1,3))
h2_a = []
dh2_dx_a = []

for i in range( num_uc_robots + num_c_robots - 1 ):
	h2_a.append(cp.Parameter())
	dh2_dx_a.append(cp.Parameter((1,3)))

alpha2 = 10
k2 = 3

V2.value = c_robots[0].lyapunov()[0] 
dV2_dx.value = c_robots[0].lyapunov()[0][1] 

h2.value = c_robots[0].sphere_barrier(R, XC)[0] 
dh2_dx.value = c_robots[0].sphere_barrier(R, XC)[1]

objective2 = cp.Minimize(10 * cp.quad_form(u2,P) + 10 * cp.square(delta2))
constraint2 = [ ]
constraint2 += [ dV2_dx @ u2 <= - k2 * V2 + delta2 ] # CLF
constraint2 += [ dh2_dx @ u2 >= - alpha2 * h2 ] # CBF for obstacle

for i in range( num_uc_robots + num_c_robots - 1 ):
	constraint2 += [ dh2_dx_a[i] @ u2 >= - alpha2 * h2_a[i] ] # CBF for other agents

prob2 = cp.Problem(objective2, constraint2)

# plot setup
fig = plt.figure()
ax = plt.axes(projection='3d')

u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
x = R * np.cos(u)*np.sin(v)
y = R * np.sin(u)*np.sin(v)
z = R * np.cos(v)
ax.plot_surface(x + XC[0], y + XC[1], z + XC[2], color = 'orange')

# update dynamics
for t in range( t_steps ):

	for i in range( num_uc_robots ):
		
		V1.value = uc_robots[i].lyapunov()[0] 
		dV1_dx.value = uc_robots[i].lyapunov()[1]

		h1.value = uc_robots[i].sphere_barrier(R, XC)[0] 
		dh1_dx.value = uc_robots[i].sphere_barrier(R, XC)[1] 

		prob1.solve()
		print("status of prob 1: ", prob1.status)

		X_next = uc_robots[i].X[-3:] + u1.value * dt
		uc_robots[i].X = np.concatenate((uc_robots[i].X, X_next), axis = 0)

	for j in range( num_c_robots ):

		V2.value = c_robots[j].lyapunov()[0] 
		dV2_dx.value = c_robots[j].lyapunov()[1] 

		h2.value = c_robots[j].sphere_barrier(R, XC)[0]
		dh2_dx.value = c_robots[j].sphere_barrier(R, XC)[1]

		n = 0 # n-th robot

		for k in range( num_uc_robots ):
			h2_a[n].value = c_robots[j].agent_barrier(uc_robots[k].R, uc_robots[k].X)[0] 
			dh2_dx_a[n].value = c_robots[j].agent_barrier(uc_robots[k].R, uc_robots[k].X)[1] 
			n = n + 1

		for k in range( num_c_robots ):
			if k == j:
				continue
			h2_a[n].value = c_robots[j].agent_barrier(uc_robots[k].R, c_robots[k].X)[0]  
			dh2_dx_a[n].value = c_robots[j].agent_barrier(uc_robots[k].R, c_robots[k].X)[1]
			n = n + 1

		prob2.solve()
		print("status of prob 2: ", prob2.status)

		X_next = c_robots[j].X[-3:] + u2.value * dt
		c_robots[j].X = np.concatenate((c_robots[j].X, X_next), axis = 0)

for j in range( num_c_robots ):
	ax.scatter3D(c_robots[j].X[::3], c_robots[j].X[1::3] , c_robots[j].X[2::3], color = "blue")

for i in range( num_uc_robots ):
	ax.scatter3D(uc_robots[i].X[::3], uc_robots[i].X[1::3] , uc_robots[i].X[2::3], color = "red")

plt.xlabel("x")
plt.ylabel("y")

plt.show()
