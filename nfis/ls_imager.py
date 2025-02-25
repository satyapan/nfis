import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import casacore.tables as ct
import os
from tqdm import tqdm
import imageio
import pickle

from .funcs import *

class ms_data:
    """
    Load visibilities.

    Arguments:
    ms_file (str or list): Path to ms, or list of multiple paths.
    data_col (str): Data column to use  
    timerange (str or tuple): 'full' or tuple of two integers corresponding to start and end times 
    """

    def __init__(self, ms_file, data_col='DATA', timerange='full'):
        self.ms_file = ms_file
        self.data_col = data_col
        self.timerange = timerange
        if type(ms_file) != list:
            t = ct.table(ms_file, readonly=True)
            self.ant1 = t.getcol('ANTENNA1')
            self.ant2 = t.getcol('ANTENNA2')
            self.uvw = t.getcol('UVW')
            self.shape = t.getcol(self.data_col).shape
            self.freq_list = get_ms_freqs(self.ms_file)
            self.N_ch = len(self.freq_list)
            self.N_pol = self.shape[2]
            self.ant_ids = np.array(list(set(self.ant1)))
            self.N_ant = len(self.ant_ids)
            self.N_bl = int((self.N_ant*(self.N_ant-1)/2)+self.N_ant)
            self.N_t = int(self.shape[0]/self.N_bl)
            self.ant1_ids = self.ant1.reshape(self.N_t,self.N_bl)[0]
            self.ant2_ids = self.ant2.reshape(self.N_t,self.N_bl)[0]
            self.data_avg = self.apply_geom_timeavg(t.getcol(self.data_col), self.uvw, self.N_t)
            t.close()
        else:
            uvw_list = []
            data_avg_list = []
            shape_list = []
            for i in range(len(ms_file)):
                ms = ms_file[i]
                t = ct.table(ms, readonly=True)
                data = t.getcol(self.data_col)
                uvw = t.getcol('UVW')
                uvw_list.append(uvw)
                shape = data.shape
                print(shape)
                shape_list.append(shape)
                if i==0:
                    self.ant1 = t.getcol('ANTENNA1')
                    self.ant2 = t.getcol('ANTENNA2')
                    self.freq_list = get_ms_freqs(ms)
                    self.N_ch = len(self.freq_list)
                    self.N_pol = shape[2]
                    self.ant_ids = np.array(list(set(self.ant1)))
                    self.N_ant = len(self.ant_ids)
                    self.N_bl = int((self.N_ant*(self.N_ant-1)/2)+self.N_ant)
                    N_t = int(shape[0]/self.N_bl)
                    self.ant1_ids = self.ant1.reshape(N_t,self.N_bl)[0]
                    self.ant2_ids = self.ant2.reshape(N_t,self.N_bl)[0]
                N_t = int(shape[0]/self.N_bl)
                data_avg = self.apply_geom_timeavg(data, uvw, N_t)
                data_avg_list.append(data_avg)
                t.close()
            self.uvw = np.concatenate(uvw_list, axis=0)
            self.data_avg = np.average(data_avg_list, axis=0)
            self.shape = (sum([shape_list[i][0] for i in range(len(ms_file))]),shape[1],shape[2])
            self.N_t = int(self.shape[0]/self.N_bl)
        
    def apply_geom_timeavg(self, data, uvw, N_t):
        w = uvw[:,2]
        geom_phase = np.exp(2j*np.pi*w[:,None,None]*self.freq_list[None,:,None]/3.0e8)
        data_geom = data*geom_phase
        data_reshape = data_geom.reshape(N_t,self.N_bl,self.N_ch,self.N_pol)
        if self.timerange == 'full':
            data_avg = np.average(data_reshape, axis=0)
        else:
            if type(self.timerange) == tuple:
                data_avg = np.average(data_reshape[self.timerange[0]:self.timerange[1],:,:,:], axis=0)
            else:
                data_avg = np.zeros((self.N_bl,self.N_ch,self.N_pol), dtype='complex')
                for i in range(len(self.timerange)):
                    data_avg += np.average(data_reshape[self.timerange[i][0]:self.timerange[i][1],:,:,:], axis=0)
                data_avg = data_avg/len(self.timerange)
        return data_avg
    
    def get_nfi_gen(self, N_pix=50, dm=200, offset=(0,0,0), stokes='V', N_ch_set=1, bl_cut=None, fullpol=False, dipole_prop=None):
        return nfi_gen(self.ms_file, self.data_avg, self.ant1_ids, self.ant2_ids, self.freq_list, N_pix=N_pix, dm=dm, offset=offset, stokes=stokes, N_ch_set=N_ch_set, bl_cut=bl_cut, fullpol=fullpol, dipole_prop=dipole_prop)


class nfi_gen:
    """
    Make image generator.

    Arguments:

    N_pix (int): Number of pixels along each side
    dm (float): Total width of grid in meter
    offset (tuple): (x,y,z) indicating x,y,z location of the center of the image with respect to the average x,y,z values of the array layout
    stokes (str): Stokes parameter to use. Options are 'I','Q','U','V','XX','XY','YX','YY'.
    N_ch_set (int): Number of channels to group together to perform inversion independently.
    bl_cut (str): Only use baseline lengths below this limit
    fullpol (bool): If True, the instrumental polarization is accounted for in inversion
    dipole_prop (tuple): If not None, this indicates the properties of the emitter: (dipole_angle,dipole_direction,dipole_pattern) where dipole_angle is the angle the dipole makes with the East, in deg, dipole_direction is 1 or -1, dipole_pattern is True or False based on whether to imprint a sin^2(theta) attenuation factor.
    """

    def __init__(self, ms_file, data_avg, ant1_ids, ant2_ids, freq_list, N_pix, dm, offset, stokes, N_ch_set, bl_cut, fullpol, dipole_prop):
        self.data_avg = data_avg
        self.N_bl = data_avg.shape[0]
        self.ant1_ids = ant1_ids
        self.ant2_ids = ant2_ids
        self.fullpol = fullpol
        self.dipole_prop = dipole_prop
        self.freq_list = freq_list
        self.N_ch = freq_list.shape[0]
        if type(ms_file) != list:
            locs = get_ant_loc_enu(ms_file)
        else:
            locs = get_ant_loc_enu(ms_file[0])
        self.x_ant = locs[:,0]
        self.y_ant = locs[:,1]
        self.z_ant = locs[:,2]
        self.N_ant = len(self.x_ant)
        self.N_pix = N_pix
        self.dm = dm
        self.stokes = stokes
        self.N_ch_set = N_ch_set
        self.x_grid, self.y_grid, self.xy_grid, self.z_val = self.get_xy_grid(offset)
        self.N_n = self.xy_grid.shape[0]
        self.M_arr = self.get_M_arr(self.xy_grid)
        self.freq_set = None
        self.griddata = True
        self.bl_cut = bl_cut
        if self.bl_cut is not None:
            self.data_avg *= self.get_bl_sel()

    def get_bl_sel(self):
        bl_sel = np.ones(self.N_bl)
        for i in range(self.N_bl):
            if np.sqrt((self.x_ant[self.ant1_ids[i]]-self.x_ant[self.ant2_ids[i]])**2+(self.y_ant[self.ant1_ids[i]]-self.y_ant[self.ant2_ids[i]])**2) > self.bl_cut:
                bl_sel[i] = 0
        return bl_sel[:,None,None]

    def get_phase_att(self,x,y,z,x1,y1,z1,x2,y2,z2,nu):
        dist1 = np.sqrt((x1-x)**2+(y1-y)**2+(z1-z)**2)
        dist2 = np.sqrt((x2-x)**2+(y2-y)**2+(z2-z)**2)
        path_diff = dist2-dist1
        att = 1/(dist1*dist2)
        phase = np.exp(-(2j)*np.pi*nu*path_diff/3.0e8)
        return att*phase
        
    def get_xy_grid(self, offset):
        x_grid = np.linspace(-self.dm+offset[0],self.dm+offset[0],self.N_pix)
        y_grid = np.linspace(-self.dm+offset[1],self.dm+offset[1],self.N_pix)
        z_val = np.average(self.z_ant)+offset[2]
        xy_grid = []
        for i in range(self.N_pix):
            for j in range(self.N_pix):
                xy_grid.append((x_grid[i],y_grid[j]))
        xy_grid = np.array(xy_grid)
        return x_grid, y_grid, xy_grid, z_val
    
    def get_leakage(self, location):
        dipole_angle=self.dipole_prop[0]
        dipole_direction=self.dipole_prop[1]
        dipole_pattern=self.dipole_prop[2]
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

    def get_leakage_arr(self):
        leakage_arr = np.ones((self.N_bl, self.N_pix**2, self.N_ch))
        leakage_x_pix = []
        leakage_y_pix = []
        for i in range(self.N_pix**2):
            location = [self.xy_grid[i][0],self.xy_grid[i][1],self.z_val]
            att_ant_x, att_ant_y = self.get_leakage(location)
            leakage_x_pix.append(att_ant_x)
            leakage_y_pix.append(att_ant_y)
        leakage_x_pix = np.array(leakage_x_pix).T
        leakage_y_pix = np.array(leakage_y_pix).T
        if self.stokes == 'XX':
            leakage_arr *= leakage_x_pix[self.ant1_ids,:,None]*leakage_x_pix[self.ant2_ids,:,None]
        elif self.stokes == 'XY':
            leakage_arr *= leakage_x_pix[self.ant1_ids,:,None]*leakage_y_pix[self.ant2_ids,:,None]
        elif self.stokes == 'YX':
            leakage_arr *= leakage_y_pix[self.ant1_ids,:,None]*leakage_x_pix[self.ant2_ids,:,None]
        elif self.stokes == 'YY':
            leakage_arr *= leakage_y_pix[self.ant1_ids,:,None]*leakage_y_pix[self.ant2_ids,:,None]
        return leakage_arr

    
    def get_M_arr(self, xy_grid):
        M_arr = self.get_phase_att(self.xy_grid[:,0][None,:,None], self.xy_grid[:,1][None,:,None], self.z_val, self.x_ant[self.ant1_ids][:,None,None],self.y_ant[self.ant1_ids][:,None,None],self.z_ant[self.ant1_ids][:,None,None],self.x_ant[self.ant2_ids][:,None,None],self.y_ant[self.ant2_ids][:,None,None],self.z_ant[self.ant2_ids][:,None,None],self.freq_list[None,None,:])
        if self.fullpol:
            M_arr *= self.get_leakage_arr()
        M_arr = np.concatenate((M_arr.real,M_arr.imag), axis=0)
        return M_arr
    
    def get_v_arr(self):
        if self.stokes == 'I':
            v = 0.5*(self.data_avg[:,:,0]+self.data_avg[:,:,3])
        elif self.stokes == 'Q':
            v = 0.5*(self.data_avg[:,:,0]-self.data_avg[:,:,3])
        elif self.stokes == 'U':
            v = 0.5*(self.data_avg[:,:,1]+self.data_avg[:,:,2])
        elif self.stokes == 'V':
            v = -0.5*1j*(self.data_avg[:,:,1]-self.data_avg[:,:,2])
        elif self.stokes == 'XX':
            v = self.data_avg[:,:,0]
        elif self.stokes == 'YY':
            v = self.data_avg[:,:,3]
        elif self.stokes == 'XY':
            v = self.data_avg[:,:,1]
        elif self.stokes == 'YX':
            v = self.data_avg[:,:,2]
        v_arr = np.zeros((2*self.N_bl,self.N_ch))
        v_arr[0:self.N_bl,:] = v.real
        v_arr[self.N_bl:2*self.N_bl,:] = v.imag
        return v_arr
    
    def apply_nan_mask(self, v_arr, M_arr):
        for i in range(2*self.N_bl):
            for j in range(self.N_ch):
                if np.isnan(v_arr[i,j]):
                    v_arr[i,j] = 0
                    M_arr[i,:,j] = 0
        return v_arr, M_arr
    
    def apply_autocorr_mask(self, v_arr, M_arr):
        autocorr_mask = np.zeros((2*self.N_bl))
        for i in range(self.N_bl):
            if self.ant1_ids[i] != self.ant2_ids[i]:
                autocorr_mask[i] = 1
                autocorr_mask[self.N_bl+i] = 1
        v_arr = v_arr*autocorr_mask[:,None]
        M_arr = M_arr*autocorr_mask[:,None,None]
        return v_arr, M_arr
    
    def eqn_solver(self, M_arr, v_arr):
        return sp.optimize.nnls(M_arr, v_arr)[0]
    
    def leastsq_solve(self, apply_nan_mask=True, apply_autocorr_mask=True, N_ch_max=None, eqn_solver=None):
        v_arr = self.get_v_arr()
        M_arr = self.M_arr
        if apply_nan_mask:
            v_arr, M_arr = self.apply_nan_mask(v_arr, M_arr)
        if apply_autocorr_mask:
            v_arr, M_arr = self.apply_autocorr_mask(v_arr, M_arr)
        if N_ch_max is None:
            N_ch_max = self.N_ch
        if eqn_solver is None:
            eqn_solver = self.eqn_solver
        N_set = N_ch_max//self.N_ch_set
        bar = tqdm(total=N_set)
        nfi_data_list = []
        freq_set = []
        for i in range(N_set):
            M_arr_set = M_arr[:,:,self.N_ch_set*i]
            v_arr_set = v_arr[:,self.N_ch_set*i]
            freq_set.append(np.average(self.freq_list[self.N_ch_set*i:self.N_ch_set*(i+1)]))
            if self.N_ch_set != 1:
                for j in range(1,self.N_ch_set):
                    M_arr_set = np.concatenate((M_arr_set, M_arr[:,:,self.N_ch_set*i+j]), axis=0)
                    v_arr_set = np.concatenate((v_arr_set, v_i_arr[:,self.N_ch_set*i+j]), axis=0)
            nfi_data_list.append(eqn_solver(M_arr_set, v_arr_set))
            bar.update(1)
        self.freq_set = freq_set
        if self.griddata == False:
            return nfi_data_list
        else:
            nfi_gridded_list = []
            for k in range(N_set):
                nfi_gridded = np.zeros((self.N_pix,self.N_pix))
                for i in range(self.N_pix):
                    for j in range(self.N_pix):
                        nfi_gridded[j,i] = nfi_data_list[k][self.N_pix*i+j]
                nfi_gridded_list.append(nfi_gridded)
            return nfi_gridded_list
    
    def get_nfi(self, apply_nan_mask=True, apply_autocorr_mask=True, N_ch_max=None, eqn_solver=None):
        nfi_gridded_list = self.leastsq_solve(apply_nan_mask=apply_nan_mask, apply_autocorr_mask=apply_autocorr_mask, N_ch_max=N_ch_max, eqn_solver=eqn_solver)
        return nfi(nfi_gridded_list, self.x_grid, self.y_grid, self.z_val, self.freq_set)


class nfi:
    """
    Make image.

    Arguments:

    apply_nan_mask (bool): Mask out nans
    apply_autocorr_mask (bool): Mask out autocorrelation
    N_ch_max (int): Maximum channel to use. If None, all channels are used
    eqn_solver (func): The algorithm to perform inversion. Default is eqn_solver_fun which uses scipy.linalg.lstsq. Custom functions can be used instead.
    """

    def __init__(self, nfi_gridded_list, x_grid, y_grid, z_val, freq_set):
        self.nfi_gridded_list = nfi_gridded_list
        self.x_grid = x_grid
        self.y_grid = y_grid
        self.z_val = z_val
        self.freq_set = np.array(freq_set)
        self.N_set = len(freq_set)

    def save(self, filename):
        with open(filename, 'wb') as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)
        
    def plot(self, ax=None, channel='avg', fig_name='nfi_test', **kargs):
        """
        Plot image.

        Arguments:

        ax (obj): Axis object on which to plot. If None, a new figure and axis is created.
        channel (str or int): 'avg' will plot frequency averaged image. Integer value will plot specified channel. 'all' will make a movie and save it as a gif file.
        fig_name (str): Path to save the plot.
        """
        
        X,Y = np.meshgrid(self.x_grid,self.y_grid)
        if type(channel) == int or len(self.nfi_gridded_list)==1:
            if ax == None:
                fig,ax = plt.subplots(figsize=(8,6))
            im = ax.pcolormesh(X,Y,self.nfi_gridded_list[channel], **kargs)
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_title('Frequency = %0.2f MHz'%(self.freq_set[channel]*1e-6))
            cb = fig.colorbar(im)
            cb.set_label(r'Power [$10^{-26}\,\mathrm{W/Hz}$]')
            fig.tight_layout()
            fig.savefig(fig_name+'.png', dpi=100)
        elif channel == 'avg':
            if ax == None:
                fig,ax = plt.subplots(figsize=(8,6))
                im = ax.pcolormesh(X,Y,np.average(self.nfi_gridded_list, axis=0))
                ax.set_xlabel('X (m)')
                ax.set_ylabel('Y (m)')
                ax.set_title('Averaged over frequency')
                cb = fig.colorbar(im, ax=ax)
                cb.set_label(r'Power [$10^{-26}\,\mathrm{W/Hz}$]')    
                fig.tight_layout()
                fig.savefig(fig_name+'.png', dpi=100)
            else:
                im = ax.pcolormesh(X,Y,np.average(self.nfi_gridded_list, axis=0))
                ax.set_xlabel('X (m)')
                ax.set_ylabel('Y (m)')
                ax.set_title('Averaged over frequency')
                return im
        elif channel == 'all':
            images = []
            for i in range(self.N_set):
                fig, ax = plt.subplots()
                im = ax.pcolormesh(X,Y,self.nfi_gridded_list[i], **kargs)
                ax.set_xlabel('X (m)')
                ax.set_ylabel('Y (m)')
                ax.set_title('Frequency = %0.2f MHz'%(self.freq_set[i]*1e-6))
                cb = fig.colorbar(im)
                cb.set_label(r'Power [$10^{-26}\,\mathrm{W/Hz}$]')
                fig.tight_layout()
                fig.savefig(fig_name+'_%s.png'%(i), dpi=100)
                plt.close()
                images.append(imageio.imread(fig_name+'_%s.png'%(i)))
                os.remove(fig_name+'_%s.png'%(i))
            imageio.mimsave(fig_name+'.gif', images, fps=4)


def eqn_solver_fun(M_arr, v_arr, lamda = 1.3e-3):
    M_arr_reg = np.concatenate((M_arr, lamda*np.identity(M_arr.shape[1])), axis=0)
    v_arr_reg = np.concatenate((v_arr, np.zeros(M_arr.shape[1])), axis=0)
    return sp.linalg.lstsq(M_arr_reg, v_arr_reg)[0]