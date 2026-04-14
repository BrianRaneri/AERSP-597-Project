import pandas as pd
import numpy as np
import os

class plane:
    def __init__(self,path):
        self.path=path

        self.mdot = 0
        self.x_velo = 0
        self.static_p = 0
        self.total_p = 0
        self.rho = 0

        self.mix_out() 

    def mix_out(self):
        df = pd.read_csv(self.path)
        Vx = df['Velocity_0'].values
        Vy = df['Velocity_1'].values
        Vz = df['Velocity_1'].values

        nx = df['Normal_0'].values
        ny = df['Normal_1'].values
        nz = df['Normal_2'].values

        static_p = df['StaticPressure'].values

        Ax = df['Area_0'].values
        Ay = df['Area_1'].values
        Az = df['Area_2'].values

        rho = df['Density'].values[0]
        self.rho=rho

        u_norm = Vx*nx + Vy*ny + Vz*nz
        A_mag = np.sqrt(Ax**2 + Ay**2 + Az**2)

        m_dot_local = rho*u_norm*A_mag
        mom_local = (rho*u_norm**2+static_p) * A_mag

        m_dot_total = np.sum(m_dot_local)
        mom_total = np.sum(mom_local)
        area_total = np.sum(A_mag)

        u_mixed = m_dot_total / (rho*area_total)
        static_p_mixed = mom_total / area_total - rho * u_mixed**2
        totalP_mixed = static_p_mixed + 0.5 * rho * u_mixed**2

        Pt_local = static_p + 0.5 * rho * (Vx**2 + Vy**2 + Vz**2)
        Pt_area_avg = np.sum(Pt_local * A_mag) / area_total

        self.mdot = m_dot_total
        self.x_velo = u_mixed
        self.static_p = static_p_mixed
        self.total_p = totalP_mixed

class cfdrun:
    def __init__(self,folder_path):
        self.name = os.path.basename(folder_path)
        self.folder_path = folder_path
        self.mdot_in = 0
        self.mdot_out = 0
        self.static_p_in = 0
        self.static_p_mixed = 0
        self.total_p_in = 0
        self.total_p_mixed = 0
        self.inlet_velo = 0
        self.outlet_velo_mixed = 0
        self.rho = 0
        self.mdot_error = 0
        self.total_p_ratio = 0
        self.total_p_loss_coeff = 0

        self.inlet_plane = plane(folder_path+r'\inlet_data.csv')
        self.outlet_plane = plane(folder_path+r'\outlet_data.csv')

        self.calculate_vals()

    def calculate_vals(self):
        self.mdot_error = abs((abs(self.inlet_plane.mdot) - self.outlet_plane.mdot) / abs(self.inlet_plane.mdot))
        self.total_p_ratio = self.outlet_plane.total_p/self.inlet_plane.total_p
        self.total_p_loss_coeff = (self.inlet_plane.total_p-self.outlet_plane.total_p)/(0.5*self.outlet_plane.rho*self.outlet_plane.x_velo**2)


    def print_results(self):
        print(f'\n----------{self.name}----------')
        print(f'Mass Flow Error: {self.mdot_error}')
        print(f'Pt2/Pt1: {self.total_p_ratio }')
        print(f'dPt/q: {self.total_p_loss_coeff}')

        pass

base_dir = r"D:\Documents\PSU\2025-2026\AERSP597\Project\Results"

cfd_runs = []
for name in os.listdir(base_dir):
    full_path = os.path.join(base_dir, name)
    
    if os.path.isdir(full_path):
        cfd_runs.append(cfdrun(full_path))

for cfd_run in cfd_runs:
    cfd_run.print_results()