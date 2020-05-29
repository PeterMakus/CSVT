'''
Tool to plot coulomb stress changes.

Author: Peter Makus (peter.makus@student.uib.no)
Created: Sat May 23 2020 15:31:20
Last Modified: Friday, 29th May 2020 03:34:02 pm
'''

import os

from geographiclib.geodesic import Geodesic
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import numpy as np


DEG2KM = 111.195

def test_read():
    folder = '.'
    files = ['coseis.dat', 'years_3.dat']
    t = [0, 3]  # time in years
    cs = read_pscmp(folder, files, t, 15)
    
    lats = [71.08, 71.226, 71.093, 70.99, 71.44, 71.703, 71]
    lons = [-7.78, -8.398, -7.472, -6.65, -9.84, -11.628, -7.7]
    strikes = [112, 110, 109, 119, 111, 112, 115]
    dips = [80, 75, 87, 79, 83, 81, 90]
    lengths = [18.81, 16.19, 24.67, 27.21, 42.38, 45.47, 500]
    depths = [15, 15, 10.5, 17, 24, 10, 15]
    
    for lat, lon, s, d, l, z in zip(lats, lons, strikes, dips, lengths, depths):
        cs.add_fault(Fault(lat, lon, z,  s, d, l))
    cs.plot()
    return cs

def read_pscmp(folder, files, times, depth):
    out = []
    for f in files:
        out.append(np.loadtxt(os.path.join(folder, f), skiprows=1).T)
    out = np.array(out)

    # Assign values
    # Coords
    lat = np.unique(out[0, 0, :])
    lon = np.unique(out[0, 1, :])
    
    # create variable dict
    vars = {}

    # Create grids
    # latgrid, longrid = np.meshgrid(lat, lon)
    gs = (len(times), len(lat), len(lon))  # shape
    
    # Displacement
    # North
    vars['disp_n'] = np.reshape(out[:, 2, :], gs)
    
    # East
    vars['disp_e'] = np.reshape(out[:, 3, :], gs)

    # down
    vars['disp_z'] = np.reshape(out[:, 4, :], gs)
    
    # Stresses
    vars['stress_nn'] = np.reshape(out[:, 5, :], gs)
    vars['stress_ee'] = np.reshape(out[:, 6, :], gs)
    vars['stress_zz'] = np.reshape(out[:, 7, :], gs)
    vars['stress_ne'] = np.reshape(out[:, 8, :], gs)
    vars['stress_ez'] = np.reshape(out[:, 9, :], gs)
    vars['stress_zn'] = np.reshape(out[:, 10, :], gs)
    
    # Tilt
    vars['tilt_n'] = np.reshape(out[:, 11, :], gs)
    vars['tilt_e'] = np.reshape(out[:, 12, :], gs)
    
    # Rotation
    vars['rot'] = np.reshape(out[:, 13, :], gs)
    
    vars['geoid'] = np.reshape(out[:, 14, :], gs)
    
    vars['gravity'] = np.reshape(out[:, 15, :], gs)
    
    # Coulomb failure stress
    vars['cfs_max'] = np.reshape(out[:, 16, :], gs)
    vars['cfs_mas'] = np.reshape(out[:, 17, :], gs)
    vars['cfs_mas_opt'] = np.reshape(out[:, 18, :], gs)
    vars['cfs_opt'] = np.reshape(out[:, 21, :], gs)
    
    # Principal stress axes
    vars['sigma_mas'] = np.reshape(out[:, 19, :], gs)
    vars['sigma_opt_1'] = np.reshape(out[:, 22, :], gs)
    vars['sigma_opt_2'] = np.reshape(out[:, 26, :], gs)
    
    # Rake
    vars['rake_mas_opt'] = np.reshape(out[:, 20, :], gs)
    vars['rake_opt_1'] = np.reshape(out[:, 25, :], gs)
    vars['rake_opt_2'] = np.reshape(out[:, 29, :], gs)
    
    vars['strike_opt_1'] = np.reshape(out[:, 23, :], gs)
    vars['strike_opt_2'] = np.reshape(out[:, 27, :], gs)
    
    vars['dip_opt_1'] = np.reshape(out[:, 24, :], gs)
    vars['dip_opt_2'] = np.reshape(out[:, 28, :], gs)
    
    # clear memory
    del out
    
    # Create variable list
    # vars = [
    #     disp_n, disp_e, disp_z, stress_nn, stress_ee, stress_zz, stress_ne,
    #     stress_ez, stress_zn, tilt_n, tilt_e, rot, geoid, gravity, cfs_max,
    #     cfs_mas, cfs_mas_opt, sigma_mas, rake_mas_opt, cfs_opt, sigma_opt_1,
    #     strike_opt_1, dip_opt_1, rake_opt_1, sigma_opt_2, strike_opt_2,
    #     dip_opt_2, rake_opt_2        
    # ]
    
    # Return Object
    return CoulombStress(lat, lon, times, depth, vars)    
    

class CoulombStress(object):
    """
    Object for Coulomb stress plotting.
    """
    def __init__(self, lat, lon, times, depth, vars):
        """Object for visualisation of Coulomb stress changes computed.
        by the PSCMP2019 code (Wang et. al., 2006).

        Arguments:
            lat {np.ndarray 1D} -- Array containing latitudes.
            lon {np.ndarray 1D} -- Array containing longitudes.
            times {np.ndarray 1D} -- Array containg time for each of the
                pages of the provided matrices in vars.
            depth {float} -- Observation depth in km.
            vars {dict} -- Dictionary containing the output of PSCMP2019.
        """
        self.depth = depth
        self.lat = lat
        self.lon = lon
        self.t = np.array(times)
        self.faults = []
        self.vars = vars
    
    def add_fault(self, fault):
        """Add a fault object to the CoulombStress object.
        Will be plotted, when calling plot with plot_fault_traces=True.

        Arguments:
            fault {Fault} -- [Fault object.]
        """
        self.faults.append(fault)
    
    def plot(self, plot_fault_traces=True):
        fig, ax = plt.subplots()
        # im = ax.imshow(
        #     self.vars['cfs_opt'][0,:,:], interpolation='nearest',
        #     extent=(min(self.lon), max(self.lon), min(self.lat), max(self.lat)))
        im = ax.pcolormesh(self.lon,self.lat,self.vars['cfs_opt'][0,:,:].T)
        cb = fig.colorbar(im)
        cb.set_label('Coulomb Stress change [Pa]')
        ax.set_xlabel('longitude [deg]')
        ax.set_ylabel('latitude [deg]')
        # ax.set_aspect((max(self.lon)-min(self.lon)/(max(self.lat)-min(self.lat))))
        
        # time slider
        axcolor = 'lightgoldenrodyellow'
        axtime = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
        
        stime = Slider(
            axtime, 'years', min(self.t), max(self.t), valinit=min(self.t),
            valstep=min(np.diff(self.t)))
        
        def update(val):
            i = np.abs(self.t - stime.val).argmin()
            im.set_array(
                self.vars['cfs_opt'][i,:-1,:-1].T.reshape(im.get_array().shape))
            fig.canvas.draw_idle()
        
        if plot_fault_traces:
            for fault in self.faults:
                # Compute midpoint coordinates
                flat, flon = fault.coords_at_depth(self.depth)
                d = fault.length/DEG2KM
                
                # Endpoints
                coords = Geodesic.WGS84.ArcDirect(
                    flat, flon, fault.strike, d/2)
                alat, alon = coords['lat2'], coords['lon2']
                coords = Geodesic.WGS84.ArcDirect(
                    flat, flon, fault.strike+180, d/2)
                blat, blon = coords['lat2'], coords['lon2']
                ax.plot([alon, blon], [alat,blat], 'k')
        
        # limits
        ax.set_xlim(left=min(self.lon), right=max(self.lon))
        ax.set_ylim(min(self.lat), max(self.lat))
            
        stime.on_changed(update)
        
        plt.show()


class Fault(object):
    def __init__(self, lat, lon, depth, strike, dip, length):
        """Simple object representing a geological fault.

        Arguments:
            lat {float} -- Latitude of hypocentre.
            lon {float} -- Longitude of hypocentre
            depth {float} -- Hypocentral depth
            strike {float} -- Strike of the fault in degree.
            dip {float} -- Fault's dip in degree
            length {float} -- Length (lateral) in km.
        """
        
        # Assign stuff
        # Hypocentre
        self.lat = lat
        self.lon = lon
        self.depth = depth
        self.strike = strike
        self.dip = dip
        self.length = length
    
    def coords_at_depth(self, depth):
        """Computes the fault's trace's midpoint at a given depth. 

        Arguments:
            depth {float} -- Query depth in km.

        Returns:
            float -- Latitude of the fault's trace's midpoint.
            float -- Longitude of the fault's trace's midpoint.
        """
        if np.abs(depth-self.depth) < .1:
            return self.lat, self.lon
        
        # distance in x direction
        dx = np.abs(depth-self.depth)*np.tan(np.deg2rad(self.dip))
        
        # azimuth
        if (depth-self.depth) > 0:
            az = self.strike + 90
        else:
            az = self.strike - 90
            
        # Compute coordinates of faults trace
        coords = Geodesic.WGS84.ArcDirect(self.lat, self.lon, az, dx/DEG2KM)
        lat2, lon2 = coords['lat2'], coords['lon2']
        
        return lat2, lon2
