import pandas as pd
import numpy as np
import os
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "black",
    "grid.color": "0.85",
    "grid.linestyle": "-",
    "grid.linewidth": 0.8,
    "font.size": 11,
    "legend.frameon": True
})

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
        self.deltaPt = None

        self.inlet_plane = plane(folder_path+r'\inlet_data.csv',method=type)
        self.outlet_plane = plane(folder_path+r'\outlet_data.csv',method=type)

        self.calculate_vals()

    def calculate_vals(self):
        self.mdot_error = abs((abs(self.inlet_plane.mdot) - self.outlet_plane.mdot) / abs(self.inlet_plane.mdot))
        self.deltaPt = self.outlet_plane.total_p-self.inlet_plane.total_p
        self.total_p_ratio = self.outlet_plane.total_p/self.inlet_plane.total_p
        self.total_p_loss_coeff = (self.inlet_plane.total_p-self.outlet_plane.total_p)/(0.5*self.outlet_plane.rho*self.outlet_plane.x_velo**2)
        self.entropy_rise = np.log(self.inlet_plane.total_p / self.outlet_plane.total_p)

    def print_results(self):
        print(f'\n----------{self.name}----------')
        print(f'Mass Flow Error: {self.mdot_error}')
        print(f'Pt2/Pt1: {self.total_p_ratio }')
        print(f'dPt/q: {self.total_p_loss_coeff}')
        print(f'dPt: {self.deltaPt}')
        print(f'Entropy Rise: {self.entropy_rise}')

class RunPlot:
    def __init__(
        self,
        name,
        xdata,
        ydata,
        group_by,
        sort_by=None,
        xlabel='',
        ylabel='',
        title='',
        
    ):
        self.name = name
        self.fig, self.ax = plt.subplots()
        self.data = {}

        # Data config
        self.xdata = xdata
        self.ydata = ydata
        self.group_by = group_by
        self.sort_by = sort_by if sort_by else xdata

        self.markers = ['o', 's', '^']
        #self.linestyles = ['-', '--', ':']
        self.linestyles = ['-', '--', '-.']


        # Labels
        self.xlabel = xlabel if xlabel else xdata
        self.ylabel = ylabel if ylabel else ydata
        self.title = title if title else f'{ydata} vs {xdata}'

        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        self.ax.set_title(self.title)
        self.ax.grid()

    def add(self, cfd_run):
        # Extract values
        group_val = getattr(cfd_run, self.group_by)

        x_val = getattr(cfd_run, self.xdata)
        y_val = getattr(cfd_run, self.ydata)
        sort_val = getattr(cfd_run, self.sort_by)

        # Initialize group
        if group_val not in self.data:
            self.data[group_val] = {"x": [], "y": [], "sort": []}

        self.data[group_val]["x"].append(x_val)
        self.data[group_val]["y"].append(y_val)
        self.data[group_val]["sort"].append(sort_val)

    def update_plot(self):
        self.ax.cla()

        if not self.data:
            return

        for i, (group, vals) in enumerate(sorted(self.data.items())):
            combined = list(zip(vals["sort"], vals["x"], vals["y"]))
            combined.sort()

            _, x_sorted, y_sorted = zip(*combined)

            marker = self.markers[i %len(self.markers)]
            linestyle = self.linestyles[i %len(self.linestyles)]

            self.ax.plot(
                x_sorted,
                y_sorted,
                marker=marker,
                linestyle = linestyle,
                linewidth =  2,
                markersize = 7,
                label=f'{self.group_by} {group}',
                color = 'black'
            )

        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        self.ax.set_title(self.title)
        self.ax.grid(True, linewidth = 0.8,alpha =0.4, linestyle='-')
        self.ax.legend()

    def save(self, folder="plots"):
        os.makedirs(folder, exist_ok=True)
        filepath = os.path.join(folder, f"{self.name}.png")
        self.fig.savefig(filepath, dpi=300)

    def reset(self):
        self.data = {}
        self.ax.cla()
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        self.ax.set_title(self.title)
        self.ax.grid()

base_dir = r"D:\Documents\PSU\2025-2026\AERSP597\Project\Results"

cfd_runs = []
for name in os.listdir(base_dir):
    full_path = os.path.join(base_dir, name)
    
    if os.path.isdir(full_path):
        cfd_runs.append(cfdrun(full_path,type='star'))

plot_configs = [
    {
        "name": "loss_vs_bl",
        "xdata": "BL_param",
        "ydata": "total_p_loss_coeff",
        "group_by": "Mach",
        "sort_by": "Mach",
        "xlabel": "h (BL Parameter)",
        "ylabel": "Total Pressure Loss Coefficient",
        "title": "Loss vs BL Parameter"
    },
    {
        "name": "dpt_vs_bl",
        "xdata": "BL_param",
        "ydata": "deltaPt",
        "group_by": "Mach",
        "sort_by": "Mach",
        "xlabel": "h (BL Parameter)",
        "ylabel": "Total Pressure Loss",
        "title": "ΔPt vs BL Parameter"
    },
    {
        "name": "losscoeff_vs_mach",
        "xdata": "Mach",
        "ydata": "total_p_loss_coeff",
        "group_by": "BL_param",
        "sort_by": "BL_param",
        "xlabel": "Mach",
        "ylabel": "Total Pressure Loss Coefficient",
        "title": "Loss Coefficient vs Mach"
    },
    {
        "name": "dpt_vs_mach",
        "xdata": "Mach",
        "ydata": "deltaPt",
        "group_by": "BL_param",
        "sort_by": "BL_param",
        "xlabel": "Mach",
        "ylabel": "Total Pressure Loss",
        "title": "ΔPt vs Mach"
    },
        {
        "name": "Entropy_Rise_vs_mach",
        "xdata": "Mach",
        "ydata": "entropy_rise",
        "group_by": "BL_param",
        "sort_by": "BL_param",
        "xlabel": "Mach",
        "ylabel": "Entropy Rise",
        "title": "dS vs Mach"
    }
]
plots = {cfg["name"]: RunPlot(**cfg) for cfg in plot_configs}

for cfd_run in cfd_runs:
    cfd_run.print_results()

    for plot in plots.values():
        plot.add(cfd_run)
    '''

    cfd_run.outlet_plane.contour_plot(name = (cfd_run.name + ' Outlet - Velocity i'), 
                                      value_col='Velocity[i] (m/s)',
                                      plot_type = 'filled',
                                      deadband = 0,
                                      levels = 500, 
                                      y_lim = 0.01,
                                      vmin = 0,
                                      vmax = 80)
    
    
    cfd_run.outlet_plane.contour_plot(name = (cfd_run.name + ' Outlet - Vorticity i'), 
                                    value_col='Vorticity[i] (/s)',
                                    plot_type = 'filled',
                                    deadband = 0,
                                    levels = 500, 
                                    y_lim = 0.01,
                                    vmin = -65000,
                                    vmax = 65000)

    '''

    #cfd_run.inlet_plane.contour_plot(name = (cfd_run.name + ' Inlet - Vorticity j'), value_col = 'Vorticityj', plot_type = 'filled',levels = 500, y_lim = 0.025,grid_res = 5000)

for plot in plots.values():
    plot.update_plot()
plt.show()