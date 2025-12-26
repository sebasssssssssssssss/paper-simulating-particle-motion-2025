# PLOTTING PARTICLE TRAJECTORY IN UNSTRUCTURED MESH WITH FINAL DATA
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import zipfile
import os

zip_path = "Trixi_mesh_100_timesteps.zip"
extract_dir = "Trixi_mesh_100_timesteps"

# Unzip if the folder doesn't already exist
if not os.path.isdir(extract_dir):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

print(f"Extracted to: {extract_dir}")

# Read mesh file & build VX, VY, EToV
with open("basic_Trixi_mesh.txt", 'r') as f:
    raw = f.read().split()

#strip labels/punctuation, convert to float
clean = [s.replace('VX',' ')
           .replace('VY',' ')
           .replace('velocity',' ')
           .replace(';','')
           .replace('[','')
           .replace(']','')
           .replace('x','')
           .replace('y','')
         for s in raw]
clean = [x for x in clean if x.strip()]
data = np.array(list(map(float, clean)))

n_elems = 724
block = 4 * n_elems

vx_flat = data[0*block:1*block]
vy_flat = data[1*block:2*block]
# Placeholders (we'll overwrite these each timestep)
xvel_flat = data[2*block:3*block]
yvel_flat = data[3*block:4*block]

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

# Invert Mapping
def invert_bilinear_map(x_t, y_t, x_nodes, y_nodes, tol=1e-12, max_iter=20):
    xh = yh = 0.0
    for _ in range(max_iter):
        phi = np.array([
            .25*(1-xh)*(1-yh), .25*(1+xh)*(1-yh),
            .25*(1+xh)*(1+yh), .25*(1-xh)*(1+yh)
        ])
        dphidx = np.array([-.25*(1-yh), .25*(1-yh), .25*(1+yh), -.25*(1+yh)])
        dphidy = np.array([-.25*(1-xh), -.25*(1+xh), .25*(1+xh), .25*(1-xh)])
        x = phi.dot(x_nodes); y = phi.dot(y_nodes)
        fx, fy = x - x_t, y - y_t
        if np.hypot(fx, fy) < tol:
            return xh, yh
        J = np.array([
            [dphidx.dot(x_nodes), dphidy.dot(x_nodes)],
            [dphidx.dot(y_nodes), dphidy.dot(y_nodes)]
        ])
        dxh, dyh = np.linalg.solve(J, [fx, fy])
        xh -= dxh; yh -= dyh
    raise RuntimeError("invert_bilinear_map did not converge")

# Lookup if particle is in an element
def lookup_element_barycentric(xp, yp, VX, VY, EToV):
    tol = 1e-6
    for e, nodes in enumerate(EToV):
        xs = VX[nodes]; ys = VY[nodes]
        if xp < xs.min()-tol or xp > xs.max()+tol or yp < ys.min()-tol or yp > ys.max()+tol:
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
                xh,yh = invert_bilinear_map(xp, yp, xs, ys)
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

# Approximate particle trajectory through each of the 100 timesteps
data_dir = "Trixi_mesh_100_timesteps"
if not os.path.isdir(data_dir):
    raise RuntimeError(f"Expected folder {data_dir} to exist")

def sort_key(fn):
    m = re.search(r"(\d+)", fn)
    return int(m.group(1)) if m else -1

files = sorted([f for f in os.listdir(data_dir) if f.endswith(".txt")], key=sort_key)
n_files = len(files)
dt_file = 30.0 / (n_files - 1)

# Sums of velocities
u_sum = np.zeros(n_elems)
v_sum = np.zeros(n_elems)

# Particle parameters and trajectory
x, y, t = 7, 1.75, 0.0
traj = [(x, y)]
xvel_prev = yvel_prev = None

for i, fn in enumerate(files):
    full = os.path.join(data_dir, fn)
    with open(full) as f:
        lines = [L.strip() for L in f if L.strip()]

#parse the four blocks
    idx, blocks = 0, []
    for _ in range(4):
        idx += 1
        vals = lines[idx].strip("[]").replace(";", "").split()
        idx += 1
        blocks.append(np.array(list(map(float, vals))))
    vx_flat, vy_flat, xvel_flat, yvel_flat = blocks
    #reshape into (4, n_elems)
    xvel_f = xvel_flat.reshape(4, -1)
    yvel_f = yvel_flat.reshape(4, -1)

    if i == 0:
        xvel_prev, yvel_prev = xvel_f, yvel_f
        continue

# accumulate element average velocity
    u_sum += np.mean(xvel_f, axis=0)
    v_sum += np.mean(yvel_f, axis=0)

#perform one Rk2 step
    print(f"Stepping with file {fn!r}, dt = {dt_file:.3f}")
    x_new, y_new, t_new, ok = RK2(x, y, t, dt_file,
                                  VX, VY, EToV,
                                  xvel_prev, yvel_prev,
                                  xvel_f, yvel_f)
# stop if either stage 1/2 failed or final point is outside
    if (not ok) or (lookup_element_barycentric(x_new, y_new, VX, VY, EToV) is None):
        print("↘ particle stopped at", fn)
        break
#accept the step
    x, y, t = x_new, y_new, t_new
    traj.append((x, y))
    xvel_prev, yvel_prev = xvel_f, yvel_f

#compute the time average
u_avg = u_sum / len(traj)
v_avg = v_sum / len(traj)

traj = np.array(traj)
fig, ax = plt.subplots(figsize=(6,6))

#draw unstructured mesh
for elem in EToV:
    ids = list(elem) + [elem[0]]
    ax.plot(VX[ids], VY[ids], 'k-', lw=0.5)

#draw particle path
ax.plot(traj[:,0], traj[:,1],
         'o-', color='blue', markersize=4,
         label='Particle Path')

#draw time average velocity field
centers_x = np.mean(vx, axis=0)
centers_y = np.mean(vy, axis=0)
ax.quiver(centers_x, centers_y,
          u_avg, v_avg,
          angles='xy', scale_units='xy', scale=1,
          color='red', label='Time‑Average Velocity')

#enforce equal aspect and fix the limits of what you had
ax.set_aspect('equal', 'box')
ax.set_xlim(VX.min() - 0.1, VX.max() + 0.1)
ax.set_ylim(VY.min() - 0.1, VY.max() + 0.1)

#now hide everything axis related
ax.set_xticks([]); ax.set_yticks([]) #no x and y ticks
for spine in ax.spines.values(): # no axis lines
    spine.set_visible(False)

plt.tight_layout()
plt.savefig("trixi_single.png", dpi=300)  # High-res PNG
plt.show()





