from matplotlib.animation import PillowWriter
import matplotlib.pyplot as plt
import numpy as np
from spotter import Star, show

star = Star.from_sides(100, inc=70.0, u=(0.3, 0.2), obl=35.0, period=4.86)
# show(star)

fig,axes = plt.subplots(1,1,figsize=(6, 6), dpi=100)

starmap_ax = axes

writer = PillowWriter(fps=20)

phases = np.linspace(0,2*4.86,100)

with writer.saving(fig, 'spotrot.gif', dpi=100):
    for ph in phases:
        for ax in [starmap_ax]:
            ax.clear()
            # ax.set_facecolor('black')
            # ax.set_bgcolor("k")
        
        spot_lat = 0.35
        spot_long = 0
        
        star = star.set(y=1.0 + 0.3 * np.random.rand(star.size))  # noise
        star = star.set(y=star.y - 0.8 * star.spot(-spot_lat, spot_long, 0.3, 20))  # spot
        # star = star.set(y=star.y)
        show(star,phase=ph,ax=starmap_ax)

        writer.grab_frame()