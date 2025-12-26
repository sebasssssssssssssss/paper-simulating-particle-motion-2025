import os, re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

# -----------------------------------------------------------------------------
# 1) Read mesh & build VX, VY, EToV 
# -----------------------------------------------------------------------------
with open("basic_Trixi_mesh.txt",'r') as f:
    raw = f.read().split()
clean = [s.replace('VX',' ').replace('VY',' ').replace('velocity',' ')
           .replace(';','').replace('[','').replace(']','')
           .replace('x','').replace('y','') for s in raw]
clean = [c for c in clean if c.strip()]
data = np.array(list(map(float, clean)))

n_elems = 724
block   = 4 * n_elems
vx_flat = data[0*block:1*block].reshape(4, n_elems)
vy_flat = data[1*block:2*block].reshape(4, n_elems)

def build_VX_VY_EToV(vx, vy):
    node_map = {}; VX_list=[]; VY_list=[]; EToV=[]
    cnt = 0
    for i in range(vx.shape[1]):
        quad = [0,1,3,2]
        elem=[]
        for j in quad:
            coord=(vx[j,i],vy[j,i])
            if coord not in node_map:
                node_map[coord]=cnt
                VX_list.append(coord[0]); VY_list.append(coord[1])
                cnt+=1
            elem.append(node_map[coord])
        EToV.append(elem)
    return np.array(VX_list), np.array(VY_list), np.array(EToV)

VX, VY, EToV = build_VX_VY_EToV(vx_flat, vy_flat)

# -----------------------------------------------------------------------------
# 2) Inverse bilinear mapping + element lookup
# -----------------------------------------------------------------------------
def invert_bilinear_map(x_t,y_t,xn,yn,tol=1e-12,max_iter=20):
    xh=yh=0.0
    for _ in range(max_iter):
        phi = np.array([.25*(1-xh)*(1-yh), .25*(1+xh)*(1-yh),
                        .25*(1+xh)*(1+yh), .25*(1-xh)*(1+yh)])
        dphidx = np.array([-.25*(1-yh),.25*(1-yh),.25*(1+yh),-.25*(1+yh)])
        dphidy = np.array([-.25*(1-xh),-.25*(1+xh),.25*(1+xh),.25*(1-xh)])
        x = phi.dot(xn); y = phi.dot(yn)
        fx,fy = x-x_t, y-y_t
        if np.hypot(fx,fy)<tol: 
            return xh,yh
        J = np.array([[dphidx.dot(xn),dphidy.dot(xn)],
                      [dphidx.dot(yn),dphidy.dot(yn)]])
        dxh,dyh = np.linalg.solve(J,[fx,fy])
        xh -= dxh; yh -= dyh
    raise RuntimeError("invert failed")

def lookup_element(xp,yp):
    tol=1e-6
    for e,nodes in enumerate(EToV):
        xs,ys = VX[nodes], VY[nodes]
        if xp<xs.min()-tol or xp>xs.max()+tol or yp<ys.min()-tol or yp>ys.max()+tol:
            continue
        for (i0,i1,i2) in [(0,1,2),(0,2,3)]:
            x0,y0 = xs[i0], ys[i0]
            x1,y1 = xs[i1], ys[i1]
            x2,y2 = xs[i2], ys[i2]
            den = (y1-y2)*(x0-x2)+(x2-x1)*(y0-y2)
            l1 = ((y1-y2)*(xp-x2)+(x2-x1)*(yp-y2))/den
            l2 = ((y2-y0)*(xp-x2)+(x0-x2)*(yp-y2))/den
            l3 = 1 - l1 - l2
            if l1>=-tol and l2>=-tol and l3>=-tol:
                xh,yh = invert_bilinear_map(xp,yp, xs, ys)
                if -1-tol<=xh<=1+tol and -1-tol<=yh<=1+tol:
                    return e, xh, yh
    return None

# -----------------------------------------------------------------------------
# 3) Bilinear‐in‐space interpolation (fixed signature)
# -----------------------------------------------------------------------------
def velocity_bilinear_interpolation(x, y, t, xvel, yvel):
    """Interpolate at (x,y) in the CURRENT velocity field xvel,yvel."""
    res = lookup_element(x, y)
    if res is None:
        return None
    e, xh, yh = res
    u_loc = xvel[:, e]
    v_loc = yvel[:, e]
    phi = .25 * np.array([
        (1-xh)*(1-yh), (1+xh)*(1-yh),
        (1+xh)*(1+yh), (1-xh)*(1+yh)
    ])
    return float(phi.dot(u_loc)), float(phi.dot(v_loc))

# -----------------------------------------------------------------------------
# 4) Midpoint‐in‐time RK2
# -----------------------------------------------------------------------------
def interpolate_velocity_fields(xv0, yv0, xv1, yv1, α=0.5):
    return (1-α)*xv0 + α*xv1, (1-α)*yv0 + α*yv1

def RK2_time_dependent(x, y, t, dt,
                       xvel0, yvel0,
                       xvel1, yvel1):
    # Stage 1
    v1 = velocity_bilinear_interpolation(x, y, t, xvel0, yvel0)
    if v1 is None: 
        return x, y, t, False
    u1, w1 = v1

    # Half‐step
    xm = x + 0.5*dt*u1
    ym = y + 0.5*dt*w1
    tm = t + 0.5*dt

    # Blend fields at midpoint
    xv_half, yv_half = interpolate_velocity_fields(xvel0, yvel0, xvel1, yvel1, 0.5)

    # Stage 2
    v2 = velocity_bilinear_interpolation(xm, ym, tm, xv_half, yv_half)
    if v2 is None:
        return x, y, t, False
    u2, w2 = v2

    # Final
    return x + dt*u2, y + dt*w2, t + dt, True

# -----------------------------------------------------------------------------
# 5) Load all 100 files into memory
# -----------------------------------------------------------------------------
data_dir = "Trixi_mesh_100_timesteps"
files = sorted([f for f in os.listdir(data_dir) if f.endswith(".txt")],
               key=lambda fn:int(re.search(r"(\d+)",fn).group(1)))
n_files = len(files)
dt = 30.0/(n_files-1)

all_xvel = []
all_yvel = []
for fn in files:
    L = [l.strip() for l in open(os.path.join(data_dir,fn)) if l.strip()]
    idx, blk = 0, []
    for _ in range(4):
        idx += 1
        vals = L[idx].strip("[]").replace(";","").split()
        idx += 1
        blk.append(np.array(list(map(float,vals))))
    _,_, xv_f, yv_f = blk
    all_xvel.append(xv_f.reshape(4,-1))
    all_yvel.append(yv_f.reshape(4,-1))

# -----------------------------------------------------------------------------
# 6) Seed “oil‐spill” particles as upward-stacked circles
# -----------------------------------------------------------------------------
N_CIRCLES = 40            # how many stacked layers
PARTS_PER_CIRCLE = 50     # how many particles per circle
RADIUS = 0.05             # radius of each circular layer
x_center, y_start = 6.50, 1.50
circle_spacing = 0.012    # vertical gap between circles

initial_positions = []

for i in range(N_CIRCLES):
    angle = np.linspace(0, 2*np.pi, PARTS_PER_CIRCLE, endpoint=False)
    x_circle = x_center + RADIUS * np.cos(angle)
    y_circle = y_start + i * circle_spacing + RADIUS * np.sin(angle)
    initial_positions.extend(zip(x_circle, y_circle))

# -----------------------------------------------------------------------------
# 7) March each particle from step=1..n_files-1
# -----------------------------------------------------------------------------
# initialize
xvel_prev, yvel_prev = all_xvel[0], all_yvel[0]
trajectories = [[p] for p in initial_positions]

for step in range(1, n_files):
    xv_cur = all_xvel[step]
    yv_cur = all_yvel[step]
    t0     = (step-1)*dt

    for i, tr in enumerate(trajectories):
        x, y = tr[-1]
        x2, y2, t2, ok = RK2_time_dependent(
            x, y, t0, dt,
            xvel_prev, yvel_prev,
            xv_cur,      yv_cur
        )
        if ok:
            tr.append((x2, y2))
        else:
            tr.append((x, y))

    xvel_prev, yvel_prev = xv_cur, yv_cur

# -----------------------------------------------------------------------------
# 8) Plot “oil spill” on unstructured mesh
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6,6))

# mesh
for elem in EToV:
    ids = list(elem)+[elem[0]]
    ax.plot(VX[ids], VY[ids], 'k-', lw=0.4)

# oil‑spill colormap
cmap = LinearSegmentedColormap.from_list('oil',['#000000','#555555','#bbbbbb'])

# draw each trajectory with a 3‑color normalized time gradient
for tr in trajectories:
    pts  = np.array(tr)
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    lc   = LineCollection(segs,
                          array=np.linspace(0,1,len(segs)),
                          cmap=cmap, linewidths=0.6, alpha=0.8)
    ax.add_collection(lc)

ax.set_aspect('equal','box')
ax.set_xlim(VX.min()-0.05, VX.max()+0.05)
ax.set_ylim(VY.min()-0.05, VY.max()+0.05)
ax.axis('off')

# normalized‐time colorbar
sm   = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0,1))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, orientation='horizontal',
                    fraction=0.04, pad=0.02,
                    ticks=[0,1])
cbar.outline.set_visible(False)
for spine in cbar.ax.spines.values():
    spine.set_visible(False)
cbar.set_ticklabels(['t=0','t=30'])
cbar.set_label('Normalized time', labelpad=8)

plt.tight_layout()
plt.savefig("oil.png", dpi=300, bbox_inches='tight')
plt.show()

