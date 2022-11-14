
""" Code for 3D control problem with spherical obstacle and n agents"""
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

# time info
T = 10
dt = 0.01
t_steps = int(T/dt)

# environment info 
XC = np.array([[5],[5],[5]]) # obstacle center
R = 1  # obstacle radius

# agent info
num_agents = 2
# X0 = np.array([[[0],[0],[0]],[[0],[0],[0]]])
# X = np.copy(X0)
# r = np.array([0.1, 0.1])
# alpha = np.array([1, 1])
# k = np.array([1, 1])

X01 = np.array([[0],[0],[0]])
X1 = np.copy(X01)
G1 = np.array([[10],[10],[11]])
r1 = 1 # agent radius
alpha1 = 10
k1 = 1

X02 = np.array([[10],[0],[0]])
G2 = np.array([[0],[10],[11]])
X2 = np.copy(X02)
r2 = 2 # agent radius
alpha2 = 10
k2 = 1

# cvxpy setup
u1 = cp.Variable((3,1))
delta1 = cp.Variable(1)
V1 = cp.Parameter()
dV1_dx = cp.Parameter((1,3))
h1 = cp.Parameter()
dh1_dx = cp.Parameter((1,3))
P = np.identity(3)

V1.value = np.linalg.norm(X1[-3:] -  G1)**2
dV1_dx.value = 2 * (X1[-3:] - G1).T

h1.value = np.linalg.norm(X1[-3:] - XC)**2 - (R + r1)**2
dh1_dx.value = 2 * (X1[-3:]- XC).T

objective1 = cp.Minimize(10 * cp.quad_form(u1,P) + 10 * cp.square(delta1))
constraint1 = [ ]
constraint1 += [ dV1_dx @ u1 <= - k1 * V1 + delta1 ] # CLF
constraint1 += [ dh1_dx @ u1 >= - alpha1 * h1 ] # CBF for obstacle
prob1 = cp.Problem(objective1, constraint1)

u2 = cp.Variable((3,1))
delta2 = cp.Variable(1)
V2 = cp.Parameter()
dV2_dx = cp.Parameter((1,3))
h2 = cp.Parameter()
dh2_dx = cp.Parameter((1,3))
h2_a = cp.Parameter()
dh2_dx_a = cp.Parameter((1,3))

V2.value = np.linalg.norm(X2[-3:] -  G2)**2
dV2_dx.value = 2 * (X2[-3:] - G2).T

h2.value = np.linalg.norm(X2[-3:] - XC)**2 - (R + r2)**2
dh2_dx.value = 2 * (X2[-3:]- XC).T

objective2 = cp.Minimize(10 * cp.quad_form(u2,P) + 10 * cp.square(delta2))
constraint2 = [ ]
constraint2 += [ dV2_dx @ u2 <= - k2 * V2 + delta2 ] # CLF
constraint2 += [ dh2_dx @ u2 >= - alpha2 * h2 ] # CBF for obstacle
constraint2 += [ dh2_dx_a @ u2 >= - alpha2 * h2_a ] # CBF for other agents
prob2 = cp.Problem(objective2, constraint2)

# plot setup
fig = plt.figure()
ax = plt.axes(projection='3d')

u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
x = R * np.cos(u)*np.sin(v)
y = R * np.sin(u)*np.sin(v)
z = R * np.cos(v)
ax.plot_surface(x + XC[0], y + XC[1], z + XC[2], color = 'r')

# update dynamics
for t in range( t_steps ):

	for n in range( num_agents ):

		V1.value = np.linalg.norm(X1[-3:] -  G1)**2
		dV1_dx.value = 2 * (X1[-3:] - G1).T

		h1.value = np.linalg.norm(X1[-3:] - XC)**2 - (R + r1)**2
		dh1_dx.value = 2 * (X1[-3:]- XC).T

		prob1.solve()
		print("status of prob 1: ", prob1.status)

		X1_next = X1[-3:]  + u1.value * dt
		X1 = np.concatenate((X1, X1_next), axis = 0)

		# robots[0].X
		# robots[2].X

		# V1.value = np.linalg.norm(X3[-3:] -  G3)**2
		# dV1_dx.value = 2 * (X3[-3:] - G3).T

		# h1.value = np.linalg.norm(X3[-3:] - XC)**2 - (R + r1)**2
		# dh1_dx.value = 2 * (X3[-3:]- XC).T
		
		# prob1.solve()
		# print("status of prob 1: ", prob1.status)

		# X3_next = X3[-3:]  + u1.value * dt
		# X3 = np.concatenate((X3 X3_next), axis = 0)
		

		V2.value = np.linalg.norm(X2[-3:] -  G2)**2
		dV2_dx.value = 2 * (X2[-3:] - G2).T

		h2.value = np.linalg.norm(X2[-3:] - XC)**2 - (R + r2)**2
		dh2_dx.value = 2 * (X2[-3:]- XC).T

		h2_a.value = np.linalg.norm(X2[-3:] - X1[-3:])**2 - (r2 + r1)**2
		dh2_dx_a.value = 2 * (X2[-3:] - X1[-3:]).T

		prob2.solve()
		print("status of prob 2: ", prob2.status)

		X2_next = X2[-3:] + u2.value * dt
		X2 = np.concatenate((X2, X2_next), axis = 0)

		
ax.scatter3D(X1[::3], X1[1::3] , X1[2::3])
ax.scatter3D(X2[::3], X2[1::3] , X2[2::3])

plt.show()