import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


file = "drone_data.csv"
df = pd.read_csv(file)

# NED
ned_x = df['aircraft_x']
ned_y = df['aircraft_y']
ned_z = df['aircraft_z']
ned_roll = np.rad2deg(df['aircraft_roll'])
ned_pitch = np.rad2deg(df['aircraft_pitch'])
ned_yaw = np.rad2deg(df['aircraft_yaw'])

# ENU
model_x = df['model_x']
model_y = df['model_y']
model_z = df['model_z']
model_roll = np.rad2deg(df['model_roll'])
model_pitch = np.rad2deg(df['model_pitch'])
model_yaw = np.rad2deg(df['model_yaw'])

# Commands
u_phi = np.rad2deg(df['u_phi'])
u_theta = np.rad2deg(df['u_theta'])
u_psi = np.rad2deg(df['u_psi'])
u_vel = df['u_velocity']

# plot the lateral positions
fig, ax = plt.subplots()
ax.plot(ned_x, ned_y, label='NED')
ax.plot(model_x, model_y, label='Model (What I should see on screen)')
ax.scatter(ned_x[0], ned_y[0], label='Start', color='green')
ax.scatter(model_x[0], model_y[0], label='Start', color='green')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.legend()
ax.set_title("Lateral Position")

# plot the attitudes
fig, ax = plt.subplots(nrows=3, ncols=1)
ax[0].plot(ned_roll, label='NED Roll')
ax[0].plot(model_roll, label='Model')
ax[0].set_ylabel("Roll")

ax[1].plot(ned_pitch, label='NED Pitch')
ax[1].plot(model_pitch, label='Model')
ax[1].set_ylabel("Pitch")

ax[2].plot(ned_yaw, label='NED Yaw')
ax[2].plot(model_yaw, label='Model')

for a in ax:
    a.legend()

plt.show()
