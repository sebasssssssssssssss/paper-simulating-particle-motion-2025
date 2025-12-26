import os, re
import numpy as np
import matplotlib.pyplot as plt

# 1) Which instant to show?
SNAPSHOT_TIME = 30.0   # ← choose 5.0, 15.0 or 30.0

# 2) Three starting positions for your particles
INIT_POSITIONS = [
    (12, 1.2),
    (6, 1),
    (7.5, 2.00),
]

# 3) Read mesh & build VX, VY, EToV (unchanged)
with open("basic_Trixi_mesh.txt", 'r') as f:
    raw = f.read().split()
clean = [s.replace('VX',' ').replace('VY',' ').replace('velocity',' ')
           .replace(';','').replace('[','').replace(']','')
           .replace('x','').replace('y','')
         for s in raw]
clean = [x for x in clean if x.strip()]
data = np.array(list(map(float, clean)))

n_elems = 724
block   = 4 * n_elems

vx_flat = data[0*block:1*block]
vy_flat = data[1*block:2*block]
vx = vx_flat.reshape(4, n_elems)
vy = vy_flat.reshape(4, n_elems)

def build_VX_VY_EToV(vx, vy):
    node_map = {}
    VX_list, VY_list, EToV = [], [], []
    cnt = 0
    for i in range(vx.shape[1]):
        quad = [0,1,3,2]
        elem = []
        for j in quad:
            coord = (vx[j,i], vy[j,i])
            if coord not in node_map:
                node_map[coord] = cnt
                VX_list.append(coord[0])
                VY_list.append(coord[1])
                cnt += 1
            elem.append(node_map[coord])
        EToV.append(elem)
    return np.array(VX_list), np.array(VY_list), np.array(EToV)

VX, VY, EToV = build_VX_VY_EToV(vx, vy)

# 4) Interpolation + RK2 (as before)
def invert_bilinear_map(x_t, y_t, x_nodes, y_nodes, tol=1e-12, max_iter=20):
    xh = yh = 0.0
    for _ in range(max_iter):
        phi   = [.25*(1-xh)*(1-yh), .25*(1+xh)*(1-yh),
                 .25*(1+xh)*(1+yh), .25*(1-xh)*(1+yh)]
        dphidx= [-.25*(1-yh), .25*(1-yh), .25*(1+yh), -.25*(1+yh)]
        dphidy= [-.25*(1-xh),-.25*(1+xh), .25*(1+xh),  .25*(1-xh)]
        x = sum(p*n for p,n in zip(phi, x_nodes))
        y = sum(p*n for p,n in zip(phi, y_nodes))
        fx, fy = x - x_t, y - y_t
        if np.hypot(fx, fy) < tol:
            return xh, yh
        J = np.array([[sum(d*nn for d,nn in zip(dphidx,x_nodes)),
                       sum(d*nn for d,nn in zip(dphidy,x_nodes))],
                      [sum(d*nn for d,nn in zip(dphidx,y_nodes)),
                       sum(d*nn for d,nn in zip(dphidy,y_nodes))]])
        dxh, dyh = np.linalg.solve(J, [fx, fy])
        xh -= dxh; yh -= dyh
    raise RuntimeError

def lookup_element_barycentric(xp, yp, VX, VY, EToV):
    tol = 1e-6
    for e, nodes in enumerate(EToV):
        xs, ys = VX[nodes], VY[nodes]
        if xp < xs.min()-tol or xp>xs.max()+tol or yp<ys.min()-tol or yp>ys.max()+tol:
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
                xh, yh = invert_bilinear_map(xp, yp, xs, ys)
                if -1-tol<=xh<=1+tol and -1-tol<=yh<=1+tol:
                    return e, xh, yh
    return None

# Bilinear Interpolation of velocity
def velocity_bilinear_interpolation(x, y, t, VX, VY, EToV, xvel, yvel):
    res = lookup_element_barycentric(x, y, VX, VY, EToV)
    if res is None:
        return None
    e, xh, yh = res
    u_loc = xvel[:, e]
    v_loc = yvel[:, e]
    phi = .25 * np.array([
        (1-xh)*(1-yh), (1+xh)*(1-yh),
        (1+xh)*(1+yh), (1-xh)*(1+yh)
    ])
    return phi.dot(u_loc), phi.dot(v_loc)

#V_half = (1 - α) * V0 + α * V1,   where α = 0.5 (since tm is halfway between t0 and t1)
#linearly interpolate velocity fields at midpoint time
def interpolate_velocity_fields(xvel0, yvel0, xvel1, yvel1, alpha):
    xvel_interp = (1 - alpha) * xvel0 + alpha * xvel1
    yvel_interp = (1 - alpha) * yvel0 + alpha * yvel1

    return xvel_interp, yvel_interp
# Runge-Kutta 2nd order time stepping
def RK2(x, y, t, dt, VX, VY, EToV, xvel0, yvel0, xvel1, yvel1):
  #interpolate to get midpoint velocity field
    alpha = 0.5
    xvel_half, yvel_half = interpolate_velocity_fields(xvel0, yvel0, xvel1, yvel1, alpha)
    #evaluate velocity at initial position
    v1 = velocity_bilinear_interpolation(x, y, t, VX, VY, EToV, xvel0, yvel0)
    if v1 is None:
        return x, y, t, False
    u1, w1 = v1
    #midpoint estimate
    xm = x + 0.5 * dt * u1
    ym = y + 0.5 * dt * w1

#use interpolated field at midpoint
    v2 = velocity_bilinear_interpolation(xm, ym, t + 0.5 * dt, VX, VY, EToV, xvel_half, yvel_half)
    if v2 is None:
        return x, y, t, False
    u2, w2 = v2

#return update
    return x + dt * u2, y + dt * w2, t + dt, True

# 5) Load timestep files
data_dir = "Trixi_mesh_100_timesteps"
files = sorted([f for f in os.listdir(data_dir) if f.endswith(".txt")],
               key=lambda fn: int(re.search(r"(\d+)", fn).group(1)))
n_files = len(files)
dt_file = 30.0/(n_files-1)

# 6) March **each** particle until SNAPSHOT_TIME
all_trajs = [[pos] for pos in INIT_POSITIONS]
vel_blocks = []

# Initialize previous velocities
xvel_prev = None
yvel_prev = None

for i, fn in enumerate(files):
    t_now = i * dt_file
    if t_now > SNAPSHOT_TIME + 1e-8:
        break

    # parse this file’s velocities
    lines = [L.strip() for L in open(os.path.join(data_dir, fn)) if L.strip()]
    idx, blocks = 0, []
    for _ in range(4):
        idx += 1
        vals = lines[idx].strip("[]").replace(";", "").split()
        idx += 1
        blocks.append(np.array(list(map(float, vals))))
    vx_flat, vy_flat, xvel_flat, yvel_flat = blocks

    xvel_f = xvel_flat.reshape(4, -1)
    yvel_f = yvel_flat.reshape(4, -1)

    vel_blocks.append((np.mean(xvel_f, axis=0),
                       np.mean(yvel_f, axis=0)))

    # Skip RK2 on first step (we need both prev and current fields)
    if xvel_prev is None:
        xvel_prev = xvel_f
        yvel_prev = yvel_f
        continue

    # step every particle
    for traj in all_trajs:
        x, y = traj[-1]
        x_new, y_new, _, ok = RK2(
            x, y, t_now, dt_file,
            VX, VY, EToV,
            xvel_prev, yvel_prev,
            xvel_f, yvel_f
        )
        if not ok or lookup_element_barycentric(x_new, y_new, VX, VY, EToV) is None:
            traj.append((x, y))  # stay put
        else:
            traj.append((x_new, y_new))

    # update for next step
    xvel_prev = xvel_f
    yvel_prev = yvel_f

# 7) Plot mesh + all 3 trajectories + quiver at SNAPSHOT_TIME
fig, ax = plt.subplots(figsize=(6,6))
# mesh
for elem in EToV:
    ids = list(elem)+[elem[0]]
    ax.plot(VX[ids], VY[ids], 'k-', lw=0.5)

# trajectories
colors = ['Blue','Green','Orange']
for traj, c, pos in zip(all_trajs, colors, INIT_POSITIONS):
    traj = np.array(traj)
    ax.plot(traj[:,0], traj[:,1], '-o', color=c, markersize=4)
           # label=f"Particle at t={SNAPSHOT_TIME:.0f}")

# quiver from the last block
u_plot, v_plot = vel_blocks[-1]
centers_x = np.mean(vx, axis=0)
centers_y = np.mean(vy, axis=0)
ax.quiver(centers_x, centers_y, u_plot, v_plot,
          angles='xy', scale_units='xy', scale=1,
          color='red') 

# styling
ax.set_aspect('equal', 'box')
ax.set_xlim(VX.min()-0.1, VX.max()+0.1)
ax.set_ylim(VY.min()-0.1, VY.max()+0.1)
ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values(): sp.set_visible(False)

ax.set_title(f"Trajectories & Velocity at t = {SNAPSHOT_TIME:.0f}")
ax.legend(loc='upper right')
ax.get_legend().remove()
plt.tight_layout()
plt.savefig("figures/trixi_multi.png", dpi=300, bbox_inches='tight')
plt.show()
