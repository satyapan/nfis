import numpy as np
from nenucal import msutils
from scipy.signal import blackmanharris
from ps_eor import psutil

from .funcs import *

class delay_s_gen:
    def __init__(self, ms_file, source=None, data_col='DATA', n_timeavg=15, stokes='V'):
        self.ms_file = ms_file
        self.data_col = data_col
        self.n_timeavg = n_timeavg
        self.stokes = stokes
        ms = msutils.MsDataCube.load(self.ms_file, 0, 2000, data_col=self.data_col, n_time_avg=self.n_timeavg)
        self.data = ms.data
        self.ant1 = ms.ant1
        self.ant2 = ms.ant2
        self.shape = ms.data.shape
        self.N_freq = self.shape[0]
        self.N_t = self.shape[1]
        self.N_bl = self.shape[2]
        self.freq_list = ms.freq
        self.df = self.freq_list[1]-self.freq_list[0]
        self.delay_list = np.fft.fftshift(np.fft.fftfreq(self.N_freq, d=self.df))
        self.dt = ms.time[1]-ms.time[0]
        self.time_list = np.array([self.dt*i for i in range(self.N_t)])
        self.fringe_list = np.fft.fftshift(np.fft.fftfreq(self.N_t, d=self.dt))
        self.uvw = self.get_uvw()
        self.bh_taper_freq = blackmanharris(self.N_freq)
        self.bh_taper_time = blackmanharris(self.N_t)
        locs = get_ant_loc_enu(self.ms_file)
        self.x_ant = locs[:,0]
        self.y_ant = locs[:,1]
        self.z_ant = locs[:,2]
        if source == None:
            self.source = [27.75451691, -51.40993459, np.average(self.z_ant)]
        else:
            self.source = source
        self.exp_delay = self.get_exp_delay()
        self.data_zenith = self.data*np.exp((2j)*np.pi*self.uvw[:,2][None, None, :, None]*self.freq_list[:, None, None, None]/3.0e8)
        self.data_source = self.phase_data_source()

    def get_uvw(self):
        t = ct.table(self.ms_file, readonly=True)
        ant1 = t.getcol('ANTENNA1')
        ant2 = t.getcol('ANTENNA2')
        uvw = t.getcol('UVW')
        uvw_coords = np.zeros((self.N_bl,3))
        for i in range(self.N_bl):
            for j in range(np.shape(uvw)[0]):
                if self.ant1[i] == ant1[j] and self.ant2[i] == ant2[j]:
                    uvw_coords[i,0] = uvw[j,0]
                    uvw_coords[i,1] = uvw[j,1]
                    uvw_coords[i,2] = uvw[j,2]
                    break
        return(uvw_coords)
    
    def phase_data_source(self):
        return self.data_zenith*np.conjugate(get_nf_phase(self.source[0], self.source[1], self.source[2],self.x_ant[self.ant1][None, None, :, None],self.y_ant[self.ant1][None, None, :, None],self.z_ant[self.ant1][None, None, :, None],self.x_ant[self.ant2][None, None, :, None],self.y_ant[self.ant2][None, None, :, None],self.z_ant[self.ant2][None, None, :, None], self.freq_list[:, None, None, None]))
    
    def get_exp_delay(self):
        x_source = self.source[0]
        y_source = self.source[1]
        z_source = self.source[2]
        exp_delay = []
        for k in range(self.N_bl):
            ant1_ind = self.ant1[k]
            ant2_ind = self.ant2[k]
            x1 = self.x_ant[ant1_ind]
            y1 = self.y_ant[ant1_ind]
            x2 = self.x_ant[ant2_ind]
            y2 = self.y_ant[ant2_ind]
            dist1 = np.sqrt((x1-x_source)**2+(y1-y_source)**2)
            dist2 = np.sqrt((x2-x_source)**2+(y2-y_source)**2)
            path_diff = dist1-dist2
            w_term_delay = self.uvw[k,2]
            delay_with_w = (path_diff - w_term_delay)/3.0e8
            exp_delay.append(delay_with_w)
        exp_delay = np.array(exp_delay)*1e6
        return exp_delay
    
    def get_stokes_bu(self, data):
        if self.stokes == 'I':
            data_stokes = data[:,:,:,0]+data[:,:,:,3]
        elif self.stokes == 'Q':
            data_stokes = data[:,:,:,0]-data[:,:,:,3]
        elif self.stokes == 'U':
            data_stokes = data[:,:,:,1]+data[:,:,:,2]
        elif self.stokes == 'V':
            data_stokes = (-1j)*(data[:,:,:,1]-data[:,:,:,2])
        elif self.stokes == 'XX':
            data_stokes = data[:,:,:,0]
        elif self.stokes == 'YY':
            data_stokes = data[:,:,:,3]
        elif self.stokes == 'XY':
            data_stokes = data[:,:,:,1]
        elif self.stokes == 'YX':
            data_stokes = data[:,:,:,2]
        return data_stokes
    
    def get_delay_s(self, data_stokes_bu):
        dds = np.empty((self.N_freq,self.N_t), dtype='complex')
        for i in range(self.N_t):
            dds_temp = psutil.nudft(self.freq_list,data_stokes_bu[:,i],w=self.bh_taper_freq)
            dds[:,i] = dds_temp[1]
        #ddps = abs(dds)**2
        dfs = np.empty((self.N_freq,self.N_t), dtype='complex')
        for i in range(self.N_freq):
            dfs_temp = psutil.nudft(self.time_list,dds[i,:],w=self.bh_taper_time)
            dfs[i,:] = dfs_temp[1]
        #dfps = abs(dfs)**2
        return dds, dfs
    
    def plot_dps(self, bu_ind, ax=None, pc='ncp', **kargs):
        if pc == 'ncp':
            data = self.data
        elif pc == 'zenith':
            data = self.data_zenith
        elif pc == 'source':
            data = self.data_source
        
        data_stokes_bu = self.get_stokes_bu(data)[:,:,bu_ind]
        ddps = abs(self.get_delay_s(data_stokes_bu)[0])**2
        
        if ax == None:
            fig,ax = plt.subplots(figsize=(8,6))
        X,Y = np.meshgrid(self.time_list/60,self.delay_list*1e6)
        im = ax.pcolormesh(X,Y,ddps, norm=LogNorm())
        fig.colorbar(im, ax=ax)
        ax.axhline(y=self.exp_delay[bu_ind], color='black', linestyle='dashed', linewidth=3, alpha=0.5)
        ax.set_xlabel('Time (min)')
        ax.set_ylabel(r'Delay ($\mu$s)')
        fig.tight_layout()