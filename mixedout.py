import pandas as pd
import numpy as np

csv_file = r'C:\Users\brian\Downloads\SliceData.csv'

df = pd.read_csv(csv_file)
Vx = df['Velocity_0'].values
Vy = df['Velocity_1'].values
Vz = df['Velocity_1'].values

nx = df['Normal_0'].values
ny = df['Normal_1'].values
nz = df['Normal_2'].values

static_p_upstream  = df['StaticPressure'].values

Ax = df['Area_0'].values
Ay = df['Area_1'].values
Az = df['Area_2'].values

rho = df['Density'].values[0]

u_norm = Un = Vx*nx + Vy*ny + Vz*nz
A_mag = np.sqrt(Ax**2 + Ay**2 + Az**2)

m_dot_local = rho*u_norm*A_mag
mom_local = (rho*u_norm**2+static_p_upstream) * A_mag

m_dot_total = np.sum(m_dot_local)
mom_total = np.sum(mom_local)
area_total = np.sum(A_mag)

u_mixed = m_dot_total / (rho*area_total)
static_p_mixed = mom_total / area_total - rho * u_mixed**2
totalP_mixed = static_p_mixed + 0.5 * rho * u_mixed**2

print("\n===== MIXED-OUT RESULTS =====")
print(f"Mass flow rate (kg/s): {m_dot_total:.6f}")
print(f"Mixed-out velocity (m/s): {u_mixed:.6f}")
print(f"Mixed-out static pressure (Pa): {static_p_mixed:.6f}")
print(f"Mixed-out total pressure (Pa): {totalP_mixed:.6f}")
print("=============================\n")