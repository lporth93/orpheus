# In this script we generate the workflow plot from the readme

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch, FancyArrow, Circle, Rectangle
from matplotlib.lines import Line2D


###################
# PARAMETER SETUP # 
###################

# Paths
basepath = "/vol/euclidraid4/data/lporth/HigherOrderLensing/Estimator/orpheus/figures/workflow/"
PATH_DATA = basepath + "data/t17_ggg_forfig.npz"
PATH_SAVEFIG = basepath + "orpheus_workflow.png"

# Colors
BG = "#0b1020" # Main figure background.
CARD = "#151a2e" # Background of the three cards.
EDGE = "#262e4b" # Borders / subtle dividers.
CY = "#35c8f0" # Main cyan accent colours.
CY2 = "#5ee6dc" # Alternative cyan accent
FG = "#e9edf6" # Main text color
MUT = "#98a3bb" # Muted text color
PINK = "#ff6b8a" # Accent used for the aperture-statistics plot.

# Font setup
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "axes.unicode_minus": False,})

FONTSIZE_HEADER = 18
FONTSIZE_EMPH = 16
FONTSIZE_BASE = 14
FONTSIZE_SMALL = 12

#############
# LOAD DATA #
#############

# Load 3pcf and Map3 data
d = np.load(PATH_DATA)
phi = d["phi"]
gamma0 = d["gamma0"]
theta1, theta2 = float(d["theta1"]), float(d["theta2"])
ap = d["apradii"]
map3 = np.real(d["map3"][0, 0])
map2mx = np.mean(d["map3"][1:4, 0],axis=0)
mapmx2 = np.mean(d["map3"][4:7, 0],axis=0)
mx3 = np.real(d["map3"][7, 0])

################
# HELPER FUNCS #
################

# Draw a card
def card(x0, y0, w, h):
    fig.add_artist(FancyBboxPatch( (x0, y0), w, h, transform=fig.transFigure, 
    boxstyle="round,pad=0,rounding_size=.014", fc=CARD, ec=EDGE, lw=1.2, zorder=0,))

# Draw image inside a cart; rect = [left, bottom, width, height]; Caption above image
def img(rect, path, caption, ret_ax=False, zorder=3):
    a = fig.add_axes(rect, zorder=zorder)
    # Transparent parts of the image show the card colour.
    a.set_facecolor(CARD)
    a.imshow(mpimg.imread(path), interpolation="none")
    a.set_xticks([])
    a.set_yticks([])
    for sp in a.spines.values():
        sp.set_linewidth(0.0)
    fig.text(rect[0]+rect[2]/2, rect[1] + rect[3] + .008, caption,
        fontsize=FONTSIZE_SMALL, color=MUT, va="bottom",)
    if ret_ax:return a

# Plot setup for orpheus measurements; rect = [left, bottom, width, height]
def plot_axes(rect):
    a = fig.add_axes(rect, zorder=3)
    a.set_facecolor("#10152a")
    for s in a.spines.values():
        s.set_color(EDGE)
    a.tick_params(colors=MUT, labelsize=8.5, direction="in", top=True, right=True,)
    for lab in (list(a.get_xticklabels()) + list(a.get_yticklabels())):
        lab.set_color(MUT)
    return a

# Add text for header, title and subtitle; x, y = position of the number.
def boxhead(x, y, header):
    fig.text(x, y, header, fontsize=FONTSIZE_HEADER, color=FG, weight="bold", va="center", ha="center", 
             bbox=dict(boxstyle="round, pad=0.5", facecolor="none", edgecolor=FG, linewidth=1.5,))

###############
# DO THE PLOT #
###############

fig = plt.figure(figsize=(18.0, 10.0),facecolor=BG,)

## Set full figure background
# General structure
bg = fig.add_axes([0, 0, 1, 1],zorder=-10,)
bg.set_facecolor(BG)
bg.set_xlim(0, 1)
bg.set_ylim(0, 1)
bg.axis("off")
# Add some "stars" as background --> Set to 1 to remove
rng = np.random.default_rng(5)
nstars = 400
bg.scatter(rng.random(nstars), rng.random(nstars), s=rng.uniform(.4, 5.5, nstars),
    c="w", alpha=rng.uniform(.10, .70, nstars), lw=0, zorder=-9)

## Layout of the three cards and the arrows between them
X0, X1, X2 = .05, .375, .7 # Lhs of each card
W, Y, H = .25, .15, .8 # Width, spacing, height; shared by all panels
_headfromboxtop = 0.05 # Header separation from top of box
for x in (X0, X1, X2):
    card(x, Y, W, H)
_arrlength = 0.08
for x in (X0 + W - .015, X1 + W - .015):
    fig.add_artist(FancyArrow(x, Y + .5*H, _arrlength, 0.0, width=.02, head_width=.040,
                              head_length=.017, color=CY, transform=fig.transFigure,
                              zorder=5,))

## Card 1
_cardcenter = X0+W/2
_imwidth = 0.32
boxhead(_cardcenter, Y + H - _headfromboxtop, "INPUT DATA")
img([_cardcenter-_imwidth/2, Y+0.39, _imwidth, _imwidth], basepath + "footprint.png", "")
la = img([_cardcenter-_imwidth/2, Y+.034, _imwidth, _imwidth], basepath + "triangles.png", "", ret_ax=True, zorder=5)
for spine in la.spines.values():
    spine.set_visible(True); spine.set_edgecolor(CY); spine.set_linewidth(2.2)
fig.add_artist(Line2D([_cardcenter-.28*_imwidth, _cardcenter-.004], [0.5, 0.705], transform=fig.transFigure, 
                      color=CY, ls=':', lw=2, alpha=1, zorder=5))
fig.add_artist(Line2D([_cardcenter+.28*_imwidth, _cardcenter+.004], [0.5, 0.705], transform=fig.transFigure, 
                       color=CY, ls=':', lw=2, alpha=1, zorder=5))
fig.add_artist(Line2D([_cardcenter-.28*_imwidth, _cardcenter-.004], [0.18, 0.685], transform=fig.transFigure, 
                      color=CY, ls=':', lw=2, alpha=1, zorder=4))
fig.add_artist(Line2D([_cardcenter+.28*_imwidth, _cardcenter+.004], [0.18, 0.685], transform=fig.transFigure, 
                       color=CY, ls=':', lw=2, alpha=1, zorder=4))


## Card 2
_cardcenter = X1+W/2
_imwidth = 0.35
# Plot orpheus logo section with extra frame around it
boxhead(_cardcenter, Y + H - _headfromboxtop, "ORPHEUS PROCESSING")
la = img([_cardcenter-_imwidth/2, Y+.257, _imwidth, _imwidth], basepath + "logo_disc.png", "", ret_ax=True)
la.add_patch(Circle((.5, .5),.495,transform=la.transAxes,fill=False,ec=CY,lw=2.2,))
fig.text(X1+W/2,Y+.1,"N-point correlations",fontsize=FONTSIZE_EMPH,color=FG,weight="bold",ha="center",)
fig.text(X1+W/2,Y+.06,"in multipole basis",fontsize=FONTSIZE_BASE,color=MUT,ha="center",)
# Add buzzwords; tuple is (text, horizontal_offset, vertical_offset)
CHIPS = [("spherical patch\n decomposition", -.075, .628), 
         ("spatial hashing", .075, .628), 
         ("hierarchical tree-based\naccelerations", -.053, .225), 
         ("parallelised \n C kernels", .075, .225),]
for text, dx, dy in CHIPS:
    fig.text(X1+W/2+dx, Y+dy, text, fontsize=FONTSIZE_BASE, color=FG, ha="center", va="center", zorder=6,
            bbox=dict(boxstyle="round,pad=.45", fc="#1e2540", ec=EDGE, lw=1.))

## Card 3
_cardcenter = X2+W/2
_imwidth = 0.2
boxhead(_cardcenter, Y+H-_headfromboxtop, "OUTPUT STATISTICS")
fig.text(_cardcenter, Y+.650,  "Higher-order correlation functions", 
        fontsize=FONTSIZE_EMPH, color=FG, va="bottom", ha="center")
# 3PCF plot
a = plot_axes([_cardcenter-.09,Y+.41,_imwidth,.22])
a.plot(phi, 1e7 * np.real(gamma0), "-", lw=2.0, color=CY, label=r"Re$(\Gamma^0)$")
a.plot(phi, 1e7 * np.imag(gamma0), "--", lw=2.0, color=CY, label=r"Im$(\Gamma^0)$")
a.axhline(0.0,color=EDGE,lw=.8)
a.set_xlim(phi[0],phi[-1])
a.set_xlabel(r"$\phi$  [rad]",fontsize=FONTSIZE_SMALL,color=MUT)
a.set_ylabel(r"$10^{7}\,\times\,\Gamma^{0}(%.1f',\,%.1f')$" % (theta1, theta2),
    fontsize=FONTSIZE_SMALL, color=MUT)
a.legend(fontsize=FONTSIZE_SMALL,frameon=False,labelcolor=MUT,handlelength=1.5,loc="lower left")
a.tick_params(axis='both', labelsize=FONTSIZE_SMALL)
# Aperture stats plot
fig.text(_cardcenter, Y+.31, "Higher-order aperture statistics", 
         fontsize=FONTSIZE_EMPH, color=FG, va="bottom", ha="center")
b = plot_axes([_cardcenter-.09,Y+.06,_imwidth,.22])
b.plot(ap, 1e7 * ap * map3, "o-", ms=3.6, lw=1.8, color=CY, label="E-Mode")
b.plot(ap, 1e7 * ap * map2mx, "o-", ms=3.6, lw=1.8, color=PINK, label="Other Modes")
b.plot(ap, 1e7 * ap * mapmx2, "o-", ms=3.6, lw=1.8, color=PINK)
b.plot(ap, 1e7 * ap * mx3, "o-", ms=3.6, lw=1.8, color=PINK)
b.axhline(0.0,color=EDGE,lw=.8)
b.set_xscale("log")
b.set_xlabel("Aperture radius  [arcmin]", fontsize=FONTSIZE_SMALL, color=MUT)
b.set_ylabel(r"$10^{7}\,\times\,\theta\,\langle M_{\rm ap}^{3}\rangle$", fontsize=FONTSIZE_SMALL, color=MUT)
b.legend(loc="upper right", fontsize=FONTSIZE_SMALL, frameon=False, labelcolor=MUT)
b.tick_params(axis='both', labelsize=FONTSIZE_SMALL)

## Footer
# Get the baseline
fig.add_artist(FancyBboxPatch((.042, .045),.916,.066,transform=fig.transFigure,
                                boxstyle="round,pad=0,rounding_size=.012",fc="none",ec=EDGE,lw=1.2,zorder=0,))
x0 = .14; dx = .170; y = .078
texts = []
for k, t in enumerate(("2-, 3-, 4-point estimators", "integrated statistics", 
                       "direct estimators","scalar & spinor fields",)):
    x = x0 + k * dx
    texts.append(fig.text(x, y, t, fontsize=FONTSIZE_BASE, color=FG, weight="bold", va="center",ha="center"))
fig.text(.935,.078,"github.com/lporth93/orpheus",fontsize=FONTSIZE_SMALL,color=MUT,ha="right",va="center")
# Add dots in the middle; to get the actual middle we need to render once so mpl knows the actual text dimensions
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
inv = fig.transFigure.inverted()
for left, right in zip(texts[:-1], texts[1:]):
    lb = left.get_window_extent(renderer)
    rb = right.get_window_extent(renderer)
    x_left = inv.transform((lb.x1, lb.y0))[0]
    x_right = inv.transform((rb.x0, rb.y0))[0]
    x_dot = (x_left + x_right) / 2
    bg.plot(x_dot, y, "o", ms=4.5, color=CY, zorder=10)

# Save
fig.savefig(PATH_SAVEFIG, dpi=200, facecolor=BG)
print("wrote", os.path.normpath(PATH_SAVEFIG))