import pandas as pd
import numpy as np
import os
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
class plane:
    def __init__(self,path,method='star'):
        self.path=path

        self.mdot = None
        self.x_velo = None
        self.static_p = None
        self.total_p = None
        self.rho = None

        self.df = None


        if method == 'star':
            self.mix_out_star() 
        else:
            self.mix_out()

    def mix_out(self):
        df = pd.read_csv(self.path)
        Vx = df['Velocity_0'].values
        Vy = df['Velocity_1'].values
        Vz = df['Velocity_2'].values

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
        self.df = df


    def mix_out_star(self):
        df = pd.read_csv(self.path)
        Vx = df['Velocity[i] (m/s)'].values
        Vy = df['Velocity[j] (m/s)'].values
        Vz = df['Velocity[k] (m/s)'].values

        nx = df['Normal[i]'].values
        ny = df['Normal[j]'].values
        nz = df['Normal[k]'].values

        static_p = df['Static Pressure (Pa)'].values

        Ax = df['Area[i] (m^2)'].values
        Ay = df['Area[j] (m^2)'].values
        Az = df['Area[k] (m^2)'].values

        rho = df['Density (kg/m^3)'].values[0]
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
        self.df = df

    def contour_plot(self,name = None,plot_type = 'filled',levels = 20,y_lim=None,value_col = 'Vorticity[i] (/s)',x_col = 'Centroid[Y] (m)',y_col = 'Centroid[Z] (m)',deadband=0,vmin = 0, vmax=50000):


        x = self.df[x_col].values
        y = self.df[y_col].values
        z = self.df[value_col].values

        # Trim NAN values
        initial_n = len(x)
        valid_mask = (~np.isnan(x)) & (~np.isnan(y)) & (~np.isnan(z))
        x = x[valid_mask]
        y = y[valid_mask]
        z = z[valid_mask]

        final_n = len(x)

        if (initial_n - final_n) != 0:
            print(f"\n[Plot Cleaning] Removed {initial_n - final_n} rows "
            f"({100*(initial_n-final_n)/initial_n:.2f}%) due to NaNs")
        
        plt.figure()
        #levels_array = np.linspace(c_min, c_max, levels)   

        if deadband == 0:
            levels = np.linspace(vmin,vmax,200)
        else:
            levels = np.concatenate([np.linspace(-vmax,-deadband,100),np.linspace(vmax,deadband,100)])  
        
        levels = np.sort(levels)

        # FILLED CONTOUR
        if plot_type in ["filled", "both"]:
            cf = plt.tricontourf(x, y, z, levels=levels,cmap='bwr',vmin=vmin,vmax=vmax)
            plt.colorbar(cf)

        # LINE CONTOUR
        if plot_type in ["lines", "both"]:
            
            cs = plt.tricontour(
                x, y, z,
                colors='black',
                linewidths=0.8)
            plt.clabel(cs, inline=True, fontsize=8, fmt="%.2f")

        x_axis_label = 'Y'
        y_axis_label = 'Z'

        plt.xlabel(x_axis_label)
        plt.ylabel(y_axis_label)
        plt.xlim([-2,2])
        plt.title(name)
        plt.tight_layout()    


        if y_lim is not None:
            plt.ylim(0, y_lim)

class cfdrun:
    def __init__(self,folder_path,type='star'):
        self.name = os.path.basename(folder_path)
        self.folder_path = folder_path
        self.mdot_in = None
        self.mdot_out = None
        self.static_p_in = None
        self.static_p_mixed = None
        self.total_p_in = None
        self.total_p_mixed = None
        self.inlet_velo = None
        self.outlet_velo_mixed = None
        self.rho = None
        self.mdot_error = None
        self.total_p_ratio = None
        self.total_p_loss_coeff = None
        self.entropy_rise = None
        self.BL_param = float(self.name.split('_h')[1].split('_')[0])
        self.Mach = float(self.name.split('_Mach')[1].split('_')[0])

        self.inlet_plane = plane(folder_path+r'\inlet_data.csv',method=type)
        self.outlet_plane = plane(folder_path+r'\outlet_data.csv',method=type)

        self.calculate_vals()

    def calculate_vals(self):
        self.mdot_error = abs((abs(self.inlet_plane.mdot) - self.outlet_plane.mdot) / abs(self.inlet_plane.mdot))
        self.total_p_ratio = self.outlet_plane.total_p/self.inlet_plane.total_p
        self.total_p_loss_coeff = (self.inlet_plane.total_p-self.outlet_plane.total_p)/(0.5*self.outlet_plane.rho*self.outlet_plane.x_velo**2)
        self.entropy_rise = np.log(self.inlet_plane.total_p / self.outlet_plane.total_p)

    def print_results(self):
        print(f'\n----------{self.name}----------')
        print(f'Mass Flow Error: {self.mdot_error}')
        print(f'Pt2/Pt1: {self.total_p_ratio }')
        print(f'dPt/q: {self.total_p_loss_coeff}')
        print(f'Entropy Rise: {self.entropy_rise}')

def update_loss_vs_h_plot(cfd_run):

    if not hasattr(update_loss_vs_h_plot, "fig"):
        update_loss_vs_h_plot.fig, update_loss_vs_h_plot.ax = plt.subplots()
        update_loss_vs_h_plot.data = {}  # key = Mach, value = (h list, loss list)

        ax = update_loss_vs_h_plot.ax
        ax.set_xlabel('h (Boundary Layer Parameter)')
        ax.set_ylabel('Total Pressure Loss Coefficient (ΔPt/q)')
        ax.set_title('Total Pressure Loss vs h')
        ax.grid()

    Mach = cfd_run.Mach

    if Mach not in update_loss_vs_h_plot.data:
        update_loss_vs_h_plot.data[Mach] = {"h": [], "loss": []}


    update_loss_vs_h_plot.data[Mach]["h"].append(cfd_run.BL_param)
    update_loss_vs_h_plot.data[Mach]["loss"].append(cfd_run.total_p_loss_coeff)

  
    ax = update_loss_vs_h_plot.ax
    ax.cla()


    for Mach, vals in sorted(update_loss_vs_h_plot.data.items()):
        h_vals = vals["h"]
        loss_vals = vals["loss"]


        data_sorted = sorted(zip(h_vals, loss_vals))
        h_sorted, loss_sorted = zip(*data_sorted)

        ax.plot(h_sorted, loss_sorted, marker='o', label=f'Mach {Mach}')

    # Reapply labels
    ax.set_xlabel('h (Boundary Layer Parameter)')
    ax.set_ylabel('Total Pressure Loss Coefficient (ΔPt/q)')
    ax.set_title('Total Pressure Loss vs h')
    ax.grid()
    ax.legend()

base_dir = r"D:\Documents\PSU\2025-2026\AERSP597\Project\Results"

cfd_runs = []
for name in os.listdir(base_dir):
    full_path = os.path.join(base_dir, name)
    
    if os.path.isdir(full_path):
        cfd_runs.append(cfdrun(full_path,type='star'))

for cfd_run in cfd_runs:
    cfd_run.print_results()

    '''

    cfd_run.outlet_plane.contour_plot(name = (cfd_run.name + ' Outlet - Velocity i'), 
                                      value_col='Velocity[i] (m/s)',
                                      plot_type = 'filled',
                                      deadband = 0,
                                      levels = 500, 
                                      y_lim = 0.01,
                                      vmin = 0,
                                      vmax = 80)
    
    '''
    
    cfd_run.outlet_plane.contour_plot(name = (cfd_run.name + ' Outlet - Vorticity i'), 
                                    value_col='Vorticity[i] (/s)',
                                    plot_type = 'filled',
                                    deadband = 0,
                                    levels = 500, 
                                    y_lim = 0.01,
                                    vmin = -65000,
                                    vmax = 65000)



    
    update_loss_vs_h_plot(cfd_run)

    #cfd_run.inlet_plane.contour_plot(name = (cfd_run.name + ' Inlet - Vorticity j'), value_col = 'Vorticityj', plot_type = 'filled',levels = 500, y_lim = 0.025,grid_res = 5000)

plt.show()