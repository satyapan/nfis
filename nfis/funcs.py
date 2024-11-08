import numpy as np
from lofarantpos import geo
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
import astropy.units as u
import warnings
import casacore.tables as ct
import os
import pickle

def load(filename):
    with open(filename, 'rb') as inp:
        obj = pickle.load(inp)
    return obj

def XYZ_from_LatLonAlt(latitude, longitude, altitude):
    gps_b = 6356752.31424518
    gps_a = 6378137
    e_squared = 6.69437999014e-3
    e_prime_squared = 6.73949674228e-3
    latitude = np.array(latitude)
    longitude = np.array(longitude)
    altitude = np.array(altitude)
    Npts = latitude.size
    if longitude.size != Npts:
        raise ValueError(
            'latitude, longitude and altitude must all have the same length')
    if altitude.size != Npts:
        raise ValueError(
            'latitude, longitude and altitude must all have the same length')
    gps_N = gps_a / np.sqrt(1 - e_squared * np.sin(latitude)**2)
    xyz = np.zeros((Npts, 3))
    xyz[:, 0] = ((gps_N + altitude) * np.cos(latitude) * np.cos(longitude))
    xyz[:, 1] = ((gps_N + altitude) * np.cos(latitude) * np.sin(longitude))
    xyz[:, 2] = ((gps_b**2 / gps_a**2 * gps_N + altitude) * np.sin(latitude))
    xyz = np.squeeze(xyz)
    return xyz

def ENU_from_ECEF(xyz, latitude, longitude, altitude):
    xyz = np.array(xyz)
    if xyz.ndim > 1 and xyz.shape[1] != 3:
        if xyz.shape[0] == 3:
            warnings.warn('The expected shape of ECEF xyz array is (Npts, 3). '
                          'Support for arrays shaped (3, Npts) will go away in '
                          'version 1.5', DeprecationWarning)
            xyz_in = xyz.T
            transpose = True
        else:
            raise ValueError('The expected shape of ECEF xyz array is (Npts, 3).')
    else:
        xyz_in = xyz
        transpose = False
    if xyz.shape == (3, 3):
        warnings.warn('The xyz array in ENU_from_ECEF is being '
                      'interpreted as (Npts, 3). Historically this function '
                      'has supported (3, Npts) arrays, please verify that '
                      'array ordering is as expected. This warning will be '
                      'removed in version 1.5', DeprecationWarning)
    if xyz_in.ndim == 1:
        xyz_in = xyz_in[np.newaxis, :]
    ecef_magnitudes = np.linalg.norm(xyz_in, axis=1)
    sensible_radius_range = (6.35e6, 6.39e6)
    if (np.any(ecef_magnitudes <= sensible_radius_range[0])
            or np.any(ecef_magnitudes >= sensible_radius_range[1])):
        raise ValueError(
            'ECEF vector magnitudes must be on the order of the radius of the earth')
    xyz_center = XYZ_from_LatLonAlt(latitude, longitude, altitude)
    xyz_use = np.zeros_like(xyz_in)
    xyz_use[:, 0] = xyz_in[:, 0] - xyz_center[0]
    xyz_use[:, 1] = xyz_in[:, 1] - xyz_center[1]
    xyz_use[:, 2] = xyz_in[:, 2] - xyz_center[2]

    enu = np.zeros_like(xyz_use)
    enu[:, 0] = (-np.sin(longitude) * xyz_use[:, 0]
                 + np.cos(longitude) * xyz_use[:, 1])
    enu[:, 1] = (-np.sin(latitude) * np.cos(longitude) * xyz_use[:, 0]
                 - np.sin(latitude) * np.sin(longitude) * xyz_use[:, 1]
                 + np.cos(latitude) * xyz_use[:, 2])
    enu[:, 2] = (np.cos(latitude) * np.cos(longitude) * xyz_use[:, 0]
                 + np.cos(latitude) * np.sin(longitude) * xyz_use[:, 1]
                 + np.sin(latitude) * xyz_use[:, 2])
    if len(xyz.shape) == 1:
        enu = np.squeeze(enu)
    elif transpose:
        return enu.T
    return enu

def get_ant_loc_enu(ms_file):
    nenufarcentre_geo = (np.deg2rad(2.192400), np.deg2rad(47.376511), 150)
    nenufarcentre_xyz = geo.xyz_from_geographic(*nenufarcentre_geo)
    nenufar_location = EarthLocation(lat=nenufarcentre_geo[1] * u.rad, lon=nenufarcentre_geo[0] * u.rad)
    ant_pos = ct.table(ms_file + '/ANTENNA', ack=False).getcol('POSITION')
    return ENU_from_ECEF(ant_pos.T, nenufar_location.lat.rad, nenufar_location.lon.rad, 150).T

def get_ms_freqs(ms_file):
    with ct.table(os.path.join(ms_file, 'SPECTRAL_WINDOW'), readonly=True, ack=False) as t_spec_win:
        freqs = t_spec_win.getcol('CHAN_FREQ').squeeze()
    return freqs

def get_vis(x,y,z,x1,y1,z1,x2,y2,z2,I,nu):
    dist1 = np.sqrt((x1-x)**2+(y1-y)**2+(z1-z)**2)
    dist2 = np.sqrt((x2-x)**2+(y2-y)**2+(z2-z)**2)
    path_diff = dist2-dist1
    intensity = I/(dist1*dist2)
    phase = np.exp(-(2j)*np.pi*nu*path_diff/3.0e8)
    return intensity*phase

def get_nf_phase(x,y,z,x1,y1,z1,x2,y2,z2,nu):
    dist1 = np.sqrt((x1-x)**2+(y1-y)**2+(z1-z)**2)
    dist2 = np.sqrt((x2-x)**2+(y2-y)**2+(z2-z)**2)
    path_diff = dist2-dist1
    phase = np.exp(-(2j)*np.pi*nu*path_diff/3.0e8)
    return phase