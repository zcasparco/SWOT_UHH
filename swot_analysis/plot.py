import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean as cm


def plot_segments(lon,lat,dat,title,vmin,vmax):
    proj = ccrs.PlateCarree()
    extent = [np.nanmin(lon),
              np.nanmax(lon),
              np.nanmin(lat),
              np.nanmax(lat)]
    
    # Add the scatter plot
    lon=lon.flatten()
    lat=lat.flatten()
    
    fig = plt.figure(figsize=(10,10))
    ax = plt.axes(projection=proj)
    
    im=ax.scatter(lon, lat, c=dat, s=1, cmap=plt.cm.bwr,
                  transform=ccrs.PlateCarree(),
                  vmin=vmin,vmax=vmax )
    # Add the colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', 
                        pad=0.02, aspect=40, shrink=0.8,
                        extend='both' , label='meter')
    ax.coastlines(resolution='10m')
    #ax.add_feature(cfeature.LAND, color='lightgrey')
    #ax.add_feature(cfeature.OCEAN, color='lightblue')
    #ax.add_feature(cfeature.RIVERS)
    gl=ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    gl.top_labels = gl.right_labels = False
    ax.set_extent(extent, proj)
    ax.set_title(title)
    return ax
