import numpy as np
from lofarantpos import geo
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
import casacore.tables as ct
import multiprocessing as mp

from .funcs import *

class nf_sim:
    def __init__(self, ms_file, data_col='NFI_SIM', stokes='I'):
        self.ms_file = ms_file
        self.data_col = data_col
        self.stokes = stokes
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
        self.locations = None
        self.intensities = None
        
    def sim_source(self, location, intensity):
        data = np.zeros(self.shape, dtype='complex')
        x = location[0]
        y = location[1]
        z = location[2]+np.average(self.z_ant)
        print('Simulating source at %s,%s,%s'%(x,y,z))
        vis_val = get_vis(x,y,z,self.x_ant[self.ant1][:,None],self.y_ant[self.ant1][:,None],self.z_ant[self.ant1][:,None],self.x_ant[self.ant2][:,None],self.y_ant[self.ant2][:,None],self.z_ant[self.ant2][:,None],intensity,self.freq_list[None,:])
        geom_phase = np.exp(-(2j)*np.pi*self.uvw[:,2][:,None]*self.freq_list[None,:]/3.0e8)
        if self.stokes == 'I':
            data[:,:,0] = vis_val*geom_phase/2
            data[:,:,3] = vis_val*geom_phase/2
        elif self.stokes == 'XX':
            data[:,:,0] = vis_val*geom_phase
        elif self.stokes == 'YY':
            data[:,:,3] = vis_val*geom_phase
        return data
    
    def sim_mp(self, source_id):
        location = self.locations[source_id]
        intensity = self.intensities[source_id]
        return self.sim_source(location, intensity)
    
    def sim_sources(self, locations, intensities, maxthreads=12):
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
        source_ids = np.arange(N_s)
        numthreads = N_s
        if numthreads > maxthreads:
            numthreads = maxthreads
        pool = mp.Pool(numthreads)
        results = pool.map(self.sim_mp, source_ids)
        return sum(results)
    
    def sim_sources_ms(self, locations, intensities, maxthreads=12):
        data = self.sim_sources(locations, intensities, maxthreads=maxthreads)
        t = ct.table(self.ms_file, readonly=False)
        if self.data_col in t.colnames():
            t.putcol(self.data_col,data)
            t.close()
        else:
            coldmi = t.getdminfo('DATA')
            coldmi['NAME'] = self.data_col
            coldesci = makecoldesc(self.data_col,t.getcoldesc('DATA'))
            t.addcols(coldesci,coldmi)
            t.putcol(self.data_col,data)
            t.close()