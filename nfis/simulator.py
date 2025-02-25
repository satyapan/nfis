import numpy as np
from lofarantpos import geo
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
import casacore.tables as ct
import multiprocessing as mp

from .funcs import *

class nf_sim:
    """
    Make simulator.

    Arguments:
    ms_file (str): Path to ms file where simulation is to be performed
    data_col (str): Data column to use. If column already present, the data will be replaced. If not, a new column will be created.
    fullpol (bool): If True, instrumental polarization effect is included in simulated data.
    """

    def __init__(self, ms_file, data_col='NFI_SIM', fullpol=True):
        self.ms_file = ms_file
        self.data_col = data_col
        self.fullpol = fullpol
        t = ct.table(self.ms_file, readonly=True)
        self.ant1 = t.getcol('ANTENNA1')
        self.ant2 = t.getcol('ANTENNA2')
        self.shape = t.getcol('DATA').shape
        self.uvw = t.getcol('UVW')
        t.close()
        self.freq_list = get_ms_freqs(self.ms_file)
        self.N_ch = len(self.freq_list)
        locs = get_ant_loc_enu(self.ms_file)
        self.x_ant = locs[:,0]
        self.y_ant = locs[:,1]
        self.z_ant = locs[:,2]
        self.N_ant = len(self.x_ant)
        self.locations = None
        self.intensities = None
        self.dipole_props = None

    def get_leakage(self, location, dipole_angle=0, dipole_direction=1, dipole_pattern=False):
        phi_y = -45*np.pi/180
        phi_x = (180+45)*np.pi/180
        y_vec = np.array([np.cos(phi_y), np.sin(phi_y)])
        x_vec = np.array([np.cos(phi_x), np.sin(phi_x)])
        att_ant_x = []
        att_ant_y = []
        for i in range(self.N_ant):
            x_ant = self.x_ant[i]
            y_ant = self.y_ant[i]
            angle_ant = np.arctan2((y_ant-location[1]),(x_ant-location[0]))
            # if dipole_angle*np.pi/180 <= angle_ant <= dipole_angle*np.pi/180+np.pi:            #Enable negative values based on the direction of the wavefront
            #     wavefront = dipole_direction*np.array([np.sin(angle_ant),-np.cos(angle_ant)])
            # else:
            #     wavefront = -dipole_direction*np.array([np.sin(angle_ant),-np.cos(angle_ant)])
            wavefront = dipole_direction*np.array([np.sin(angle_ant),-np.cos(angle_ant)])
            att_x = np.dot(wavefront, x_vec)
            att_y = np.dot(wavefront, y_vec)
            if dipole_pattern:
                att_x = att_x*abs(np.sin(angle_ant-dipole_angle*np.pi/180))
                att_y = att_y*abs(np.sin(angle_ant-dipole_angle*np.pi/180))
            att_ant_x.append(att_x)
            att_ant_y.append(att_y)
        return np.array(att_ant_x), np.array(att_ant_y)
        
    def sim_source(self, location, intensity, dipole_prop=None):
        data = np.zeros(self.shape, dtype='complex')
        x = location[0]
        y = location[1]
        z = location[2]+np.average(self.z_ant)
        print('Simulating source at %s,%s,%s'%(x,y,z))
        vis_val = get_vis(x,y,z,self.x_ant[self.ant1][:,None],self.y_ant[self.ant1][:,None],self.z_ant[self.ant1][:,None],self.x_ant[self.ant2][:,None],self.y_ant[self.ant2][:,None],self.z_ant[self.ant2][:,None],intensity[None,:],self.freq_list[None,:])
        geom_phase = np.exp(-(2j)*np.pi*self.uvw[:,2][:,None]*self.freq_list[None,:]/3.0e8)
        if self.fullpol:
            dipole_angle, dipole_direction, dipole_pattern = dipole_prop
            att_ant_x, att_ant_y = self.get_leakage(location, dipole_angle=dipole_angle, dipole_direction=dipole_direction, dipole_pattern=dipole_pattern)
            data[:,:,0] = vis_val*geom_phase*(att_ant_x[self.ant1]*att_ant_x[self.ant2])[:,None]
            data[:,:,1] = vis_val*geom_phase*(att_ant_x[self.ant1]*att_ant_y[self.ant2])[:,None]
            data[:,:,2] = vis_val*geom_phase*(att_ant_y[self.ant1]*att_ant_x[self.ant2])[:,None]
            data[:,:,3] = vis_val*geom_phase*(att_ant_y[self.ant1]*att_ant_y[self.ant2])[:,None]
        else:
            data[:,:,0] = vis_val*geom_phase
            data[:,:,3] = vis_val*geom_phase
        return data
    
    def sim_mp(self, source_id):
        location = self.locations[source_id]
        intensity = self.intensities[source_id]
        if self.fullpol:
            dipole_prop = self.dipole_props[source_id]
        else:
            dipole_prop=None
        return self.sim_source(location, intensity, dipole_prop=dipole_prop)

    def sim_sources(self, locations, intensities, dipole_props=None, maxthreads=12):
        if type(locations) == str:
            self.locations = np.loadtxt(locations)
        else:
            self.locations = locations
        if self.locations.shape == (3,):
            self.locations = self.locations.reshape([1,3])
        N_s = self.locations.shape[0]
        if type(intensities) == list:
            self.intensities = intensities
        else:
            self.intensities = [intensities for i in range(N_s)]
        if self.fullpol:
            if dipole_props is None:
                self.dipole_props = np.array([0,1,False])
            else:
                self.dipole_props = dipole_props
            if self.dipole_props.shape == (3,):
                self.dipole_props = self.dipole_props.reshape([1,3])
        source_ids = np.arange(N_s)
        numthreads = N_s
        if numthreads > maxthreads:
            numthreads = maxthreads
        pool = mp.Pool(numthreads)
        results = pool.map(self.sim_mp, source_ids)
        return sum(results)
    
    def sim_sources_ms(self, locations, intensities, dipole_props=None, maxthreads=12):
        """
        Simulate data.

        Arguments:
        locations (list or str): List of locations of near field sources. Each element in the list is a list of three elements with x,y,z locations. If str it refers to a text file containing the source locations, with each source in a line, and x,y,z separated by commas.
        intensities (list): List of intensity arrays for the sources. e.g. [a1, a2, a3] where ai is an array of length N_freq with the spectral powers
        dipole_props (list): If not None, this indicates the properties of the emitter: (dipole_angle,dipole_direction,dipole_pattern) where dipole_angle is the angle the dipole makes with the East, in deg, dipole_direction is 1 or -1, dipole_pattern is True or False based on whether to imprint a sin^2(theta) attenuation factor.
        maxthreads (int): Maximum number of threads over which to parallelize.
        """
        
        data = self.sim_sources(locations, intensities, dipole_props=dipole_props, maxthreads=maxthreads)
        t = ct.table(self.ms_file, readonly=False)
        if self.data_col in t.colnames():
            t.putcol(self.data_col,data)
            t.close()
        else:
            coldmi = t.getdminfo('DATA')
            coldmi['NAME'] = self.data_col
            coldesci = ct.makecoldesc(self.data_col,t.getcoldesc('DATA'))
            t.addcols(coldesci,coldmi)
            t.putcol(self.data_col,data)
            t.close()