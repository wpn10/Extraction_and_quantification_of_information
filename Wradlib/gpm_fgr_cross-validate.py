import warnings
warnings.filterwarnings('ignore')
import wradlib as wrl
import matplotlib.pyplot as pl
import matplotlib as mpl
from matplotlib.collections import PatchCollection
from matplotlib.colors import from_levels_and_colors
from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib.cm as cm
try:
    get_ipython().magic("matplotlib inline")
except:
    pl.ion()
import numpy as np
import datetime as dt
from osgeo import osr
# define GPM data set
gpm_file = wrl.util.get_wradlib_data_file('2A-CS-151E24S154E30S.GPM.Ku.V7-20170308.20141206-S095002-E095137.004383.V05A.HDF5')
# define matching ground radar file
gr2gpm_file = wrl.util.get_wradlib_data_file('IDR66_20141206_094829.vol.h5')

# define TRMM data sets
trmm_2a23_file = wrl.util.get_wradlib_data_file('2A-CS-151E24S154E30S.TRMM.PR.2A23.20100206-S111425-E111526.069662.7.HDF')
trmm_2a25_file = wrl.util.get_wradlib_data_file('2A-CS-151E24S154E30S.TRMM.PR.2A25.20100206-S111425-E111526.069662.7.HDF')

# define matching ground radar file
gr2trmm_file = wrl.util.get_wradlib_data_file('IDR66_20100206_111233.vol.h5')
# Space-born precipitation radar parameters
sr_pars = {"trmm": {
    "zt": 402500.,  # orbital height of TRMM (post boost)   APPROXIMATION!
    "dr": 250.,     # gate spacing of TRMM
    "gr_file": gr2trmm_file,
}, "gpm": {
    "zt": 407000.,  # orbital height of GPM                 APPROXIMATION!
    "dr": 125.,      # gate spacing of GPM
    "gr_file": gr2gpm_file,
}}
# Set parameters for this procedure
bw_sr = 0.71                  # SR beam width
platf = "gpm"                 # SR platform/product: one out of ["gpm", "trmm"]
zt = sr_pars[platf]["zt"]     # SR orbit height (meters)
dr_sr = sr_pars[platf]["dr"]  # SR gate length (meters)
gr_file = sr_pars[platf]["gr_file"]
ee = 2                        # Index that points to the GR elevation angle to be used
def _get_tilts(dic):
    i = 0
    for k in dic.keys():
        if 'dataset' in k:
            i += 1
    return i


def read_gr(filename, loaddata=True):

    gr_data = wrl.io.read_generic_netcdf(filename)
    dat = gr_data['what']['date']
    tim = gr_data['what']['time']
    date = dt.datetime.strptime(dat + tim, "%Y%d%m%H%M%S")
    source = gr_data['what']['source']

    lon = gr_data['where']['lon']
    lat = gr_data['where']['lat']
    alt = gr_data['where']['height']

    if gr_data['what']['object'] == 'PVOL':
        ntilt = _get_tilts(gr_data)
    else:
        raise ValueError('GR file is no PPI/Volume File')

    ngate = np.zeros(ntilt, dtype=np.int16)
    nbeam = np.zeros(ntilt)
    elang = np.zeros(ntilt)
    r0 = np.zeros(ntilt)
    dr = np.zeros(ntilt)
    a0 = np.zeros(ntilt)

    for i in range(0, ntilt):
        dset = gr_data['dataset{0}'.format(i+1)]
        a0[i] = dset['how']['astart']
        elang[i] = dset['where']['elangle']
        ngate[i] = dset['where']['nbins']
        r0[i] = dset['where']['rstart']
        dr[i] = dset['where']['rscale']
        nbeam[i] = dset['where']['nrays']

    if ((len(np.unique(r0)) != 1) |
            (len(np.unique(dr)) != 1) |
            (len(np.unique(a0)) != 1) |
            (len(np.unique(nbeam)) != 1) |
            (nbeam[0] != 360)):
        raise ValueError('GroundRadar Data layout dos not match')

    gr_dict = {}
    gr_dict.update({'source': source, 'date': date, 'lon': lon, 'lat': lat,
                    'alt': alt, 'ngate': ngate, 'nbeam': nbeam, 'ntilt': ntilt,
                    'r0': r0, 'dr': dr, 'a0': a0, 'elang': elang})
    if not loaddata:
        return gr_dict

    sdate = []
    refl = []
    for i in range(0, ntilt):
        dset = gr_data['dataset{0}'.format(i+1)]
        dat = dset['what']['startdate']
        tim = dset['what']['starttime']
        date = dt.datetime.strptime(dat + tim, "%Y%d%m%H%M%S")
        sdate.append(date)
        data = dset['data1']
        quantity = data['what']['quantity']
        factor = data['what']['gain']
        offset = data['what']['offset']
        if quantity == 'DBZH':
            dat = data['variables']['data']['data'] * factor + offset
            refl.append(dat)

    sdate = np.array(sdate)
    refl = np.array(refl)

    gr_dict.update({'sdate': sdate, 'refl': refl})

    return gr_dict
# read matching GR data
gr_data = read_gr(gr_file)
# number of rays in gr sweep
nray_gr = gr_data['nbeam'].astype("i4")[ee]
# number of gates in gr beam
ngate_gr = gr_data['ngate'].astype("i4")[ee]
# number of sweeps
nelev = gr_data['ntilt']
# elevation of sweep (degree)
elev_gr = gr_data['elang'][ee]
# gate length (meters)
dr_gr = gr_data['dr'][ee]
# reflectivity array of sweep
ref_gr = gr_data['refl'][ee]
# sweep datetime stamp
date_gr = gr_data['sdate'][ee]
# range of first gate
r0_gr = gr_data['r0'][ee]
# azimuth angle of first beam
a0_gr = gr_data['a0'][ee]
# Longitude of GR
lon0_gr = gr_data['lon']
# Latitude of GR
lat0_gr = gr_data['lat']
# Altitude of GR (meters)
alt0_gr = gr_data['alt']
# Beam width of GR (degree)
bw_gr = 1.
print(elev_gr, lon0_gr)

coord = wrl.georef.sweep_centroids(nray_gr, dr_gr, ngate_gr, elev_gr)
coords = wrl.georef.spherical_to_proj(coord[..., 0],
                                      coord[..., 1],
                                      coord[..., 2],
                                      (lon0_gr, lat0_gr, alt0_gr))
lon = coords[..., 0]
lat = coords[..., 1]
alt = coords[..., 2]
bbox = wrl.zonalstats.get_bbox(lon, lat)
print("Radar bounding box:\n\t%.2f\n%.2f           %.2f\n\t%.2f" %
      (bbox['top'], bbox['left'], bbox['right'], bbox['bottom']))
# read spaceborn SR data
if platf == "gpm":
    sr_data = wrl.io.read_gpm(gpm_file, bbox)
elif platf == "trmm":
    sr_data = wrl.io.read_trmm(trmm_2a23_file, trmm_2a25_file, bbox)
else:
    raise("Invalid platform")
refl = sr_data['refl']
print(refl)
# Longitudes of SR scans
sr_lon = sr_data['lon']
# Latitudes of SR scans
sr_lat = sr_data['lat']
# Precip flag
pflag = sr_data['pflag']
# Number of scans on SR data
nscan_sr= sr_data['nscan']
# Number of rays in one SR scan
nray_sr = sr_data['nray']
# Number of gates in one SR ray
ngate_sr = sr_data['nbin']
print(sr_lon.shape)
# Calculate equivalent earth radius
wgs84 = wrl.georef.get_default_projection()
re1 = wrl.georef.get_earth_radius(lat0_gr, wgs84)
print("Earth radius 1:", re1)
a = wgs84.GetSemiMajor()
b = wgs84.GetSemiMinor()
print("SemiMajor, SemiMinor:", a, b)

# Set up aeqd-projection gr-centered
rad = wrl.georef.proj4_to_osr(('+proj=aeqd +lon_0={lon:f} ' +
                                   '+lat_0={lat:f} +a={a:f} ' +
                                   '+b={b:f}').format(lon=lon0_gr,
                                                      lat=lat0_gr,
                                                      a=a, b=b))
re2 = wrl.georef.get_earth_radius(lat0_gr, rad)
print("Earth radius 2:", re2)
# create gr range and azimuth arrays
rmax_gr = r0_gr + ngate_gr * dr_gr
r_gr = np.arange(0, ngate_gr) * dr_gr + dr_gr/2.
az_gr = np.arange(0, nray_gr) - a0_gr
print("Range/Azi-Shape:", r_gr.shape, az_gr.shape)

# create gr polar grid and calculate aeqd-xyz coordinates
gr_polargrid = np.meshgrid(r_gr, az_gr)
gr_xyz, rad = wrl.georef.spherical_to_xyz(gr_polargrid[0], gr_polargrid[1], elev_gr, (lon0_gr, lat0_gr, alt0_gr ),
                                          squeeze=True)
print("XYZ-Grid-Shape:", gr_xyz.shape)

# create gr poygon array in aeqd-xyz-coordinates
gr_poly, rad1 = wrl.georef.spherical_to_polyvert(r_gr, az_gr, elev_gr, (lon0_gr, lat0_gr, alt0_gr))
print(gr_poly.shape, 360 * 600)
gr_poly.shape = (nray_gr, ngate_gr, 5, 3)

# get radar domain (outer ring)
gr_domain = gr_xyz[:,-1,0:2]
gr_domain = np.vstack((gr_domain, gr_domain[0]))
print("Domain-Shape:", gr_domain.shape)
sr_x, sr_y = wrl.georef.reproject(sr_lon, sr_lat,
                                      projection_source=wgs84,
                                      projection_target=rad)
sr_xy = np.dstack((sr_x, sr_y))
print("SR-GRID-Shapes:", sr_x.shape, sr_y.shape, sr_xy.shape)
sr_x, sr_y = wrl.georef.reproject(sr_lon, sr_lat,
                                      projection_source=wgs84,
                                      projection_target=rad)
sr_xy = np.dstack((sr_x, sr_y))
print("SR-GRID-Shapes:", sr_x.shape, sr_y.shape, sr_xy.shape)
# Create ZonalData for spatial subsetting (inside GR range domain)

# get precip indexes
print(sr_xy.shape)
print(pflag.shape)

precip_mask = (pflag == 2) & wrl.zonalstats.get_clip_mask(sr_xy, gr_domain, rad)

# get iscan/iray boolean arrays
print(precip_mask.shape, sr_xy.shape)
print(pflag.shape, sr_xy.reshape(-1, sr_xy.shape[-1]).shape)
pl.imshow(precip_mask)
print("NRAY", nray_sr)
print("NBIN", ngate_sr)

# use localZenith Angle
alpha = sr_data['zenith']
beta = abs(-17.04 + np.arange(nray_sr) * bw_sr)

# Correct for parallax, get 3D-XYZ-Array
#   xyzp_sr: Parallax corrected xyz coordinates
#   r_sr_inv: range array from ground to SR platform
#   zp: SR bin altitudes
xyp_sr, r_sr_inv, z_sr = wrl.georef.correct_parallax(sr_xy, ngate_sr, dr_sr, alpha)
print(xyp_sr.shape, r_sr_inv.shape, z_sr.shape)
xyzp_sr = np.concatenate((xyp_sr, z_sr[..., np.newaxis]),
                   axis=-1)
print(sr_xy.shape)
print("SR_XYP:", xyp_sr.shape, xyzp_sr.shape, r_sr_inv.shape, z_sr.shape)
r_sr, az_sr, elev_sr = wrl.georef.xyz_to_spherical(xyzp_sr, alt0_gr, proj=rad)
#TODO: hardcoded 1.0
mask = (elev_sr > (1.0 - bw_gr/2.)) & (elev_sr < (1.0 + bw_gr/2.))
pl.figure()
pl.pcolormesh(mask[40,:,:].T)
rs = wrl.georef.dist_from_orbit(zt, alpha, beta, r_sr_inv, re1)
# Small anngle approximation
vol_sr2  = np.pi * dr_sr * rs**2 * np.radians(bw_sr / 2.)**2

# Or using wradlib's native function
vol_sr = wrl.qual.pulse_volume(rs, dr_sr, bw_sr)
# Evaluate difference between both approaches
print("Min. difference (m3):", (vol_sr - vol_sr2).min())
print("Max. difference (m3): ", (vol_sr - vol_sr2).max())
print("Average rel. difference (%):", round(np.mean(vol_sr-vol_sr2)*100./np.mean(np.mean(vol_sr2)), 4))

# Verdict: differences are negligble - use wradlibs's native function!
# GR pulse volumes
#   along one beam
vol_gr = wrl.qual.pulse_volume(r_gr, dr_gr, bw_gr)
#   with shape (nray_gr, ngate_gr)
vol_gr = np.repeat(vol_gr, nray_gr).reshape((nray_gr, ngate_gr), order="F")
Rs = 0.5 * (1 +  np.cos(np.radians(alpha)))[:,:,np.newaxis] * rs * np.tan(np.radians(bw_sr/2.))
Ds = dr_sr / np.cos(np.radians(alpha))
Ds = np.broadcast_to(Ds[..., np.newaxis], Rs.shape)
print(z_sr.shape)
print(sr_data['zbb'].shape, sr_data['bbwidth'].shape, sr_data['quality'].shape)
ratio, ibb = wrl.qual.get_bb_ratio(sr_data['zbb'], sr_data['bbwidth'], sr_data['quality'], z_sr)
print(ratio.shape)
zbb = sr_data['zbb'].copy()
zbb[~ibb] = np.nan
print(np.nanmin(ratio[..., 9]), np.nanmax(ratio[..., 9]))
#pl.imshow(ratio, vmin=-10, vmax=30)
ref_sr = sr_data['refl'].filled(np.nan)
print(ref_sr.shape, ratio.shape)
ref_sr_ss = np.zeros_like(ref_sr) * np.nan
ref_sr_sh = np.zeros_like(ref_sr) * np.nan

a_s, a_h = (wrl.trafo.KuBandToS.snow, wrl.trafo.KuBandToS.hail)

ia = (ratio >= 1)
ref_sr_ss[ia] = ref_sr[ia] + wrl.util.calculate_polynomial(ref_sr[ia], a_s[:,10])
ref_sr_sh[ia] = ref_sr[ia] + wrl.util.calculate_polynomial(ref_sr[ia], a_h[:,10])
ib = (ratio <= 0)
ref_sr_ss[ib] = ref_sr[ib] + wrl.util.calculate_polynomial(ref_sr[ib], a_s[:,0])
ref_sr_sh[ib] = ref_sr[ib] + wrl.util.calculate_polynomial(ref_sr[ib], a_h[:,0])
im = (ratio > 0) & (ratio < 1)
ind = np.round(ratio[im] * 10).astype(np.int)
ref_sr_ss[im] = ref_sr[im] + wrl.util.calculate_polynomial(ref_sr[im], a_s[:,ind])
ref_sr_sh[im] = ref_sr[im] + wrl.util.calculate_polynomial(ref_sr[im], a_h[:,ind])

# Jackson Tan's fix for C-band
is_cband = False
if (is_cband):
    deltas = (ref_sr_ss - ref_sr) * 5.3 / 10.0
    ref_sr_ss = outref_sr + deltas
    deltah = (ref_sr_sh - ref_sr) * 5.3 / 10.0
    ref_sr_sh = ref_sr + deltah

ref_sr_ss[ref_sr < 0] = np.nan
# Convert S-band GR reflectivities to Ku-band using method of Liao and Meneghini (2009)
ref_gr_ku = np.zeros_like(ref_gr) * np.nan

# Which zbb value should we take here???
#    Q'n'Dirty: just take the mean of all SR profiles
#    TODO: Consider zbb for each profile during the matching process

# Snow
ia = ( gr_xyz[...,2] >= np.nanmean(zbb) )
#ref_gr_ku[ia] = wrl.trafo.KuBandToS.snow[0] + wrl.trafo.KuBandToS.snow[1]*ref_gr[ia] + wrl.trafo.KuBandToS.snow[2]*ref_gr[ia]**2
ref_gr_ku[ia] = wrl.util.calculate_polynomial(ref_gr[ia], wrl.trafo.SBandToKu.snow)
# Rain
ib = ( gr_xyz[...,2] < np.nanmean(zbb) )
#ref_gr_ku[ib] = wrl.trafo.KuBandToS.rain[0] + wrl.trafo.KuBandToS.rain[1]*ref_gr[ia] + wrl.trafo.KuBandToS.rain[2]*ref_gr[ia]**2
ref_gr_ku[ib] = wrl.util.calculate_polynomial(ref_gr[ib], wrl.trafo.SBandToKu.rain)

# Jackson Tan's fix for C-band
is_cband = False
if (is_cband):
    delta = (ref_gr_ku - ref_gr) * 5.3/10.0
    ref_gr_ku = ref_gr + delta
# First assumption: no valid SR bins (all False)
valid = np.asarray(elev_sr, dtype=np.bool)==False
print(valid.shape, precip_mask.shape)
# SR is inside GR range and is precipitating
iscan = precip_mask.nonzero()[0]
iray = precip_mask.nonzero()[1]
valid[iscan,iray] = True
# SR bins intersect with GR sweep
valid = valid & (elev_sr >= (elev_gr-bw_gr/2.)) & (elev_sr <= (elev_gr+bw_gr/2.))
# Number of matching SR bins per profile
nvalids = np.sum(valid, axis=2)
# scan and ray indices for profiles with at least one valid bin
vscan, vray = np.where(nvalids>0)
# number of profiles with at least one valid bin
nprof = len(vscan)
print(vscan.shape)
print(valid.shape)
# average coordinates
xyz_v1 = xyzp_sr.copy()
print(xyz_v1.shape)
xyz_v1[~valid] = np.nan
xyz_c1 = xyzp_sr.filled(0)
xyz_c1[~valid] = 0
c = np.count_nonzero(xyz_c1[..., 0], axis=2)
ntotsr = c[vscan, vray]
xyz_m1 = np.nanmean(xyz_v1,axis=2)
xyz = xyz_m1[vscan, vray]
print(xyz.shape, c.shape)

# approximate Rs
rs_v1 = Rs.copy()
rs_v1[~valid] = np.nan
rs_m1 = np.nanmax(rs_v1, axis=2)
rs_prof = rs_m1[vscan, vray]
ds = rs_prof

# approximate Ds
ds_v1 = Ds.copy()
ds_v1[~valid] = np.nan
ds_m1 = np.nansum(ds_v1, axis=2)
ds_prof = ds_m1[vscan, vray]
dz = ds_prof

# approximate Vs
vs_v1 = vol_sr.copy()
vs_v1[~valid] = np.nan
vs_m1 = np.nansum(vs_v1, axis=2)
vs_prof = vs_m1[vscan, vray]
volsr1 = vs_prof

from mpl_toolkits.mplot3d import Axes3D
fig = pl.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xyz_m1[..., 0], xyz_m1[..., 1], xyz_m1[..., 2], c=c.ravel())
r_sr, az_sr, el_rs = wrl.georef.xyz_to_spherical(xyz, alt0_gr, proj=rad)
ref_sr_1 = wrl.trafo.idecibel(ref_sr)
ref_sr_1[~valid] = np.nan
refsr1a = np.nanmean(ref_sr_1, axis=2)[vscan,vray]
refsr1a = wrl.trafo.decibel(refsr1a)

ref_sr_2 = wrl.trafo.idecibel(ref_sr_ss)
ref_sr_2[~valid] = np.nan
refsr2a = np.nanmean(ref_sr_2, axis=2)[vscan,vray]
refsr2a = wrl.trafo.decibel(refsr2a)

ref_sr_3 = wrl.trafo.idecibel(ref_sr_sh)
ref_sr_3[~valid] = np.nan
refsr3a = np.nanmean(ref_sr_3, axis=2)[vscan,vray]
refsr3a = wrl.trafo.decibel(refsr3a)


print(refsr1a.shape)
zds = wrl.zonalstats.DataSource(xyz[:, 0:2].reshape(-1, 2), rad)
print(zds.ds.GetLayer().GetFeatureCount())
tmp_trg_lyr = zds.ds.GetLayer()
trg_poly = []
for i, feat in enumerate(tmp_trg_lyr):
    geom = feat.GetGeometryRef().Buffer(rs_prof[i])
    poly = wrl.georef.ogr_to_numpy(geom)
    trg_poly.append(poly)

#%%time
print("creating")
zdp = wrl.zonalstats.ZonalDataPoly(gr_poly[..., 0:2].reshape(-1, 5, 2), trg_poly, srs=rad)
zdp.dump_vector('m3d_zonal_poly_{0}'.format(platf))
#%%time
print("loading")
obj3 = wrl.zonalstats.ZonalStatsPoly('m3d_zonal_poly_{0}'.format(platf))

#%%time
print(obj3.ix.shape)
volgr1 = np.array([np.sum(vol_gr.ravel()[obj3.ix[i]])
                   for i in np.arange(len(obj3.ix))[~obj3.check_empty()]])
print(volgr1.shape)

ref_gr_i = wrl.trafo.idecibel(ref_gr.ravel())
ref_gr_ku_i = wrl.trafo.idecibel(ref_gr_ku.ravel())
refgr1a = np.array([np.nanmean(ref_gr_i[obj3.ix[i]])
             for i in np.arange(len(obj3.ix))[~obj3.check_empty()]])
refgr1a = wrl.trafo.decibel(refgr1a)
refgr2a = np.array([np.nanmean(ref_gr_ku_i[obj3.ix[i]])
             for i in np.arange(len(obj3.ix))[~obj3.check_empty()]])
refgr2a = wrl.trafo.decibel(refgr2a)
fig = pl.figure(figsize=(12,5))
ax = fig.add_subplot(121, aspect="equal")
pl.scatter(refgr1a, refsr1a, marker="+", c="black")
pl.plot([0,60],[0,60], linestyle="solid", color="black")
pl.xlim(10,50)
pl.ylim(10,50)
pl.xlabel("GR reflectivity (dBZ)")
pl.ylabel("SR reflectivity (dBZ)")
ax = fig.add_subplot(122)
pl.hist(refgr1a[refsr1a>10], bins=np.arange(-10,50,5), edgecolor="None", label="GR")
pl.hist(refsr1a[refsr1a>-10], bins=np.arange(-10,50,5), edgecolor="red", facecolor="None", label="SR")
pl.xlabel("Reflectivity (dBZ)")
pl.legend()
fig.suptitle("uncorrected GR vs uncorrected SR")
fig = pl.figure(figsize=(12,5))
ax = fig.add_subplot(121, aspect="equal")
pl.scatter(refgr2a, refsr1a, marker="+", c="black")
pl.plot([0,60],[0,60], linestyle="solid", color="black")
pl.xlim(10,50)
pl.ylim(10,50)
pl.xlabel("GR reflectivity (dBZ)")
pl.ylabel("SR reflectivity (dBZ)")
ax = fig.add_subplot(122)
pl.hist(refgr1a[refsr1a>10], bins=np.arange(-10,50,5), edgecolor="None", label="GR")
pl.hist(refsr1a[refsr1a>-10], bins=np.arange(-10,50,5), edgecolor="red", facecolor="None", label="SR")
pl.xlabel("Reflectivity (dBZ)")
pl.legend()
fig.suptitle("corrected GR vs uncorrected SR")
fig = pl.figure(figsize=(12,5))
ax = fig.add_subplot(121, aspect="equal")
pl.scatter(refgr2a, refsr2a, marker="+", c="black")
pl.plot([0,60],[0,60], linestyle="solid", color="black")
pl.xlim(10,50)
pl.ylim(10,50)
pl.xlabel("GR reflectivity (dBZ)")
pl.ylabel("SR reflectivity (dBZ)")
ax = fig.add_subplot(122)
pl.hist(refgr2a[refsr2a>10], bins=np.arange(-10,50,5), edgecolor="None", label="GR")
pl.hist(refsr2a[refsr2a>-10], bins=np.arange(-10,50,5), edgecolor="red", facecolor="None", label="SR")
pl.xlabel("Reflectivity (dBZ)")
pl.legend()
fig.suptitle("corrected GR vs corrected SR snow")
fig = pl.figure(figsize=(12,5))
ax = fig.add_subplot(121, aspect="equal")
pl.scatter(refgr2a, refsr3a, marker="+", c="black")
pl.plot([0,60],[0,60], linestyle="solid", color="black")
pl.xlim(10,50)
pl.ylim(10,50)
pl.xlabel("GR reflectivity (dBZ)")
pl.ylabel("SR reflectivity (dBZ)")
ax = fig.add_subplot(122)
pl.hist(refgr2a[refsr3a>10], bins=np.arange(-10,50,5), edgecolor="None", label="GR")
pl.hist(refsr3a[refsr3a>-10], bins=np.arange(-10,50,5), edgecolor="red", facecolor="None", label="SR")
pl.xlabel("Reflectivity (dBZ)")
pl.legend()
fig.suptitle("corrected GR vs corrected SR hail")
fig = pl.figure(figsize=(12,8))
ax = fig.add_subplot(121, aspect="equal")
pl.scatter(xyz[..., 0], xyz[...,1], c=refsr1a, cmap=pl.cm.viridis, vmin=10, vmax=50, edgecolor="None")
pl.title("SR reflectivity")
pl.xlim(-100000, 150000)
pl.ylim(-150000, 150000)
pl.grid()
ax = fig.add_subplot(122, aspect="equal")
pl.scatter(xyz[..., 0], xyz[...,1], c=refgr1a, cmap=pl.cm.viridis, vmin=10, vmax=50, edgecolor="None")
pl.title("GR reflectivity")
pl.xlim(-100000, 150000)
pl.ylim(-150000, 150000)
pl.grid()
fig.suptitle("uncorrected GR vs uncorrected SR")
pl.tight_layout()
