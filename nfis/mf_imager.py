import numpy as np
import matplotlib.pyplot as plt
import casacore.tables as ct
import os
from tqdm import tqdm
import imageio
import numexpr as ne

from .funcs import *

class ms_data_mf:
    def __init__(self, ms_file, data_col='NFI_SIM', timerange='full'):
        self.ms_file = ms_file
        self.data_col = data_col
        self.timerange = timerange
        t = ct.table(ms_file, readonly=True)
        self.data = t.getcol(self.data_col)
        self.ant1 = t.getcol('ANTENNA1')
        self.ant2 = t.getcol('ANTENNA2')
        self.uvw = t.getcol('UVW')
        self.shape = self.data.shape
        self.freq_list = get_ms_freqs(self.ms_file)
        self.N_ch = len(self.freq_list)
        self.N_pol = self.shape[2]
        self.ant_ids = np.array(list(set(self.ant1)))
        self.N_ant = len(self.ant_ids)
        self.N_bl = int((self.N_ant*(self.N_ant-1)/2)+self.N_ant)
        self.N_t = int(self.shape[0]/self.N_bl)
        self.ant1_ids = self.ant1.reshape(self.N_t,self.N_bl)[0]
        self.ant2_ids = self.ant2.reshape(self.N_t,self.N_bl)[0]
        self.data_avg = self.apply_geom_timeavg(self.data)
        t.close()
        
    def apply_geom_timeavg(self, data):
        w = self.uvw[:,2]
        geom_phase = np.exp(2j*np.pi*w[:,None,None]*self.freq_list[None,:,None]/3.0e8)
        data_geom = data*geom_phase
        data_reshape = data_geom.reshape(self.N_t,self.N_bl,self.N_ch,self.N_pol)
        if self.timerange == 'full':
            data_avg = np.average(data_reshape, axis=0)
        else:    
            data_avg = np.average(data_reshape[self.timerange[0]:self.timerange[1],:,:], axis=0)
        return data_avg
    
    def get_nfi_gen(self, N_pix=100, dm=300, offset=(0,0,0), stokes='V', channels='all'):
        return nfi_gen_mf(self.ms_file, self.data_avg, self.ant1_ids, self.ant2_ids, self.freq_list, N_pix=N_pix, dm=dm, offset=offset, stokes=stokes, channels=channels)

class nfi_gen_mf:
    def __init__(self, ms_file, data_avg, ant1_ids, ant2_ids, freq_list, N_pix, dm, offset, stokes, channels='all'):
        self.data_avg = data_avg
        self.N_bl = data_avg.shape[0]
        self.ant1_ids = ant1_ids
        self.ant2_ids = ant2_ids
        self.freq_list = freq_list
        self.N_ch = freq_list.shape[0]
        locs = get_ant_loc_enu(ms_file)
        self.x_ant = locs[:,0]
        self.y_ant = locs[:,1]
        self.z_ant = locs[:,2]
        self.N_pix = N_pix
        self.dm = dm
        self.stokes = stokes
        self.channels = channels
        self.x, self.y, self.z = self.get_xy_grid(offset)
        self.phase_grid = self.get_phase_grid()
        
    def get_xy_grid(self, offset):
        x_grid = np.linspace(-self.dm+offset[0],self.dm+offset[0],self.N_pix)
        y_grid = np.linspace(-self.dm+offset[1],self.dm+offset[1],self.N_pix)
        z = np.average(self.z_ant)+offset[2]
        x, y = np.meshgrid(x_grid, y_grid)
        return x,y,z
    
    def get_phase(self,x,y,z,x1,y1,z1,x2,y2,z2,nu):
        dist1 = np.sqrt((x1-x)**2+(y1-y)**2+(z1-z)**2)
        dist2 = np.sqrt((x2-x)**2+(y2-y)**2+(z2-z)**2)
        delay = (dist2-dist1)/3.0e8
        j2pi = 1j*2*np.pi
        phase = ne.evaluate("exp(j2pi * nu * delay)")
        return phase
        
#     def get_phase(self,x,y,z,x1,y1,z1,x2,y2,z2,nu):
#         dist1 = np.sqrt((x1-x)**2+(y1-y)**2+(z1-z)**2)
#         dist2 = np.sqrt((x2-x)**2+(y2-y)**2+(z2-z)**2)
#         att = dist1*dist2
#         delay = (dist2-dist1)/3.0e8
#         j2pi = 1j*2*np.pi
#         phase = ne.evaluate("exp(j2pi * nu * delay)*att")
#         return phase
        
    def get_phase_grid(self):
        phase_grid = self.get_phase(self.x[None,None,:,:],self.y[None,None,:,:],self.z,self.x_ant[self.ant1_ids][:,None,None,None],self.y_ant[self.ant1_ids][:,None,None,None],self.z_ant[self.ant1_ids][:,None,None,None],self.x_ant[self.ant2_ids][:,None,None,None],self.y_ant[self.ant2_ids][:,None,None,None],self.z_ant[self.ant2_ids][:,None,None,None],self.freq_list[None,:,None,None])
        return phase_grid
    
    def make_image(self):
        if self.stokes == 'I':
            vis = self.data_avg[:,:,0]+self.data_avg[:,:,3]
        elif self.stokes == 'Q':
            vis = self.data_avg[:,:,0]-self.data_avg[:,:,3]
        elif self.stokes == 'U':
            vis = self.data_avg[:,:,1]+self.data_avg[:,:,2]
        elif self.stokes == 'V':
            vis = -1j*(self.data_avg[:,:,1]-self.data_avg[:,:,2])
        elif self.stokes == 'XX':
            vis = self.data_avg[:,:,0]
        elif self.stokes == 'YY':
            vis = self.data_avg[:,:,3]
        elif self.stokes == 'XY':
            vis = self.data_avg[:,:,1]
        elif self.stokes == 'YX':
            vis = self.data_avg[:,:,2]
            
#         autocorr_mask = np.zeros((self.N_bl))
#         for i in range(self.N_bl):
#             if self.ant1_ids[i] != self.ant2_ids[i]:
#                 autocorr_mask[i] = 1
        
        bar = tqdm(total=self.N_bl, position=0, leave='None') 
        img = 0
        for k in range(self.N_bl):
            v = np.nan_to_num(vis[k,:,None,None])
            p = self.phase_grid[k,:,:,:]
            h = ne.evaluate('v * p')
            img = ne.evaluate('img + h')
            bar.update(1)
        corr_avg = img/self.N_bl
        #corr = vis[:,:,None,None]*self.phase_grid
        #corr_avg = np.nanmean(corr, axis=0)
        return corr_avg
    
    def make_image_avg(self):
        if self.stokes == 'I':
            vis = self.data_avg[:,:,0]+self.data_avg[:,:,3]
        elif self.stokes == 'Q':
            vis = self.data_avg[:,:,0]-self.data_avg[:,:,3]
        elif self.stokes == 'U':
            vis = self.data_avg[:,:,1]+self.data_avg[:,:,2]
        elif self.stokes == 'V':
            vis = -1j*(self.data_avg[:,:,1]-self.data_avg[:,:,2])
        elif self.stokes == 'XX':
            vis = self.data_avg[:,:,0]
        elif self.stokes == 'YY':
            vis = self.data_avg[:,:,3]
        elif self.stokes == 'XY':
            vis = self.data_avg[:,:,1]
        elif self.stokes == 'YX':
            vis = self.data_avg[:,:,2]
            
        bar = tqdm(total=self.N_ch, position=0, leave='None') 
        img = 0
        for k in range(self.N_ch):
            v = np.nan_to_num(vis[:,k,None,None])
            p = self.phase_grid[:,k,:,:]
            h = ne.evaluate('v * p')
            img = ne.evaluate('img + h')
            bar.update(1)
        img = img/self.N_ch
        #img = np.average(np.nan_to_num(vis[:,:,None,None])*self.phase_grid, axis=1)
        corr_blavg = np.average(abs(img), axis=0)
        return corr_blavg
    
    def get_nfi(self, avg=True):
        if avg:
            corr = self.make_image_avg()
        else:
            corr = self.make_image()
        return nfi_mf(corr, avg, self.x, self.y, self.z, self.freq_list)    

class nfi_mf:
    def __init__(self, corr, avg, x_grid, y_grid, z_val, freq_list):
        self.corr = corr.real
        self.avg = avg
        if avg:
            self.corr = [corr]
        self.x_grid = x_grid
        self.y_grid = y_grid
        self.z_val = z_val
        self.freq_list = np.array(freq_list)
        self.N_ch = len(freq_list)
        
    def plot(self, ax=None, channel=0, fig_name='nfi_test', **kargs):
        if channel != 'all' or self.avg:
            if ax == None:
                fig,ax = plt.subplots(figsize=(8,6))
            im = ax.pcolormesh(self.x_grid,self.y_grid,self.corr[channel], **kargs)
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_title('Frequency = %0.2f MHz'%(self.freq_list[channel]*1e-6))
            if ax == None:
                cb = fig.colorbar(im)
                cb.set_label(r'Amplitude [Arbitrary]')
                fig.tight_layout()
                fig.savefig(fig_name+'.png', dpi=100)
            else:
                return im
        else:
            images = []
            for i in range(self.N_ch):
                fig, ax = plt.subplots()
                im = ax.pcolormesh(self.x_grid,self.y_grid,self.corr[i], **kargs)
                ax.set_xlabel('X (m)')
                ax.set_ylabel('Y (m)')
                ax.set_title('Frequency = %0.2f MHz'%(self.freq_list[i]*1e-6))
                cb = fig.colorbar(im)
                cb.set_label(r'Amplitude [arbitrary]')
                fig.tight_layout()
                fig.savefig(fig_name+'_%s.png'%(i), dpi=100)
                plt.close()
                images.append(imageio.imread(fig_name+'_%s.png'%(i)))
                os.remove(fig_name+'_%s.png'%(i))
            imageio.mimsave(fig_name+'.gif', images, fps=4)