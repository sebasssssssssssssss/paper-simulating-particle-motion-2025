import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

# Mesh data (generate 17x17 grid to match EToV)
x_vals = np.linspace(-2.0, 2.0, 17)
y_vals = np.linspace(-2.0, 2.0, 17)
VX, VY = np.meshgrid(x_vals, y_vals, indexing='ij')
VX = VX.flatten()
VY = VY.flatten()

EToV = """1 18 2 19; 18 35 19 36; 35 52 36 53; 52 69 53 70; 69 86 70 87; 86 103 87 104; 103 120 104 121; 120 137 121 138; 137 154 138 155; 154 171 155 172; 171 188 172 189; 188 205 189 206; 205 222 206 223; 222 239 223 240; 239 256 240 257; 256 273 257 274; 2 19 3 20; 19 36 20 37; 36 53 37 54; 53 70 54 71; 70 87 71 88; 87 104 88 105; 104 121 105 122; 121 138 122 139; 138 155 139 156; 155 172 156 173; 172 189 173 190; 189 206 190 207; 206 223 207 224; 223 240 224 241; 240 257 241 258; 257 274 258 275; 3 20 4 21; 20 37 21 38; 37 54 38 55; 54 71 55 72; 71 88 72 89; 88 105 89 106; 105 122 106 123; 122 139 123 140; 139 156 140 157; 156 173 157 174; 173 190 174 191; 190 207 191 208; 207 224 208 225; 224 241 225 242; 241 258 242 259; 258 275 259 276; 4 21 5 22; 21 38 22 39; 38 55 39 56; 55 72 56 73; 72 89 73 90; 89 106 90 107; 106 123 107 124; 123 140 124 141; 140 157 141 158; 157 174 158 175; 174 191 175 192; 191 208 192 209; 208 225 209 226; 225 242 226 243; 242 259 243 260; 259 276 260 277; 5 22 6 23; 22 39 23 40; 39 56 40 57; 56 73 57 74; 73 90 74 91; 90 107 91 108; 107 124 108 125; 124 141 125 142; 141 158 142 159; 158 175 159 176; 175 192 176 193; 192 209 193 210; 209 226 210 227; 226 243 227 244; 243 260 244 261; 260 277 261 278; 6 23 7 24; 23 40 24 41; 40 57 41 58; 57 74 58 75; 74 91 75 92; 91 108 92 109; 108 125 109 126; 125 142 126 143; 142 159 143 160; 159 176 160 177; 176 193 177 194; 193 210 194 211; 210 227 211 228; 227 244 228 245; 244 261 245 262; 261 278 262 279; 7 24 8 25; 24 41 25 42; 41 58 42 59; 58 75 59 76; 75 92 76 93; 92 109 93 110; 109 126 110 127; 126 143 127 144; 143 160 144 161; 160 177 161 178; 177 194 178 195; 194 211 195 212; 211 228 212 229; 228 245 229 246; 245 262 246 263; 262 279 263 280; 8 25 9 26; 25 42 26 43; 42 59 43 60; 59 76 60 77; 76 93 77 94; 93 110 94 111; 110 127 111 128; 127 144 128 145; 144 161 145 162; 161 178 162 179; 178 195 179 196; 195 212 196 213; 212 229 213 230; 229 246 230 247; 246 263 247 264; 263 280 264 281; 9 26 10 27; 26 43 27 44; 43 60 44 61; 60 77 61 78; 77 94 78 95; 94 111 95 112; 111 128 112 129; 128 145 129 146; 145 162 146 163; 162 179 163 180; 179 196 180 197; 196 213 197 214; 213 230 214 231; 230 247 231 248; 247 264 248 265; 264 281 265 282; 10 27 11 28; 27 44 28 45; 44 61 45 62; 61 78 62 79; 78 95 79 96; 95 112 96 113; 112 129 113 130; 129 146 130 147; 146 163 147 164; 163 180 164 181; 180 197 181 198; 197 214 198 215; 214 231 215 232; 231 248 232 249; 248 265 249 266; 265 282 266 283; 11 28 12 29; 28 45 29 46; 45 62 46 63; 62 79 63 80; 79 96 80 97; 96 113 97 114; 113 130 114 131; 130 147 131 148; 147 164 148 165; 164 181 165 182; 181 198 182 199; 198 215 199 216; 215 232 216 233; 232 249 233 250; 249 266 250 267; 266 283 267 284; 12 29 13 30; 29 46 30 47; 46 63 47 64; 63 80 64 81; 80 97 81 98; 97 114 98 115; 114 131 115 132; 131 148 132 149; 148 165 149 166; 165 182 166 183; 182 199 183 200; 199 216 200 217; 216 233 217 234; 233 250 234 251; 250 267 251 268; 267 284 268 285; 13 30 14 31; 30 47 31 48; 47 64 48 65; 64 81 65 82; 81 98 82 99; 98 115 99 116; 115 132 116 133; 132 149 133 150; 149 166 150 167; 166 183 167 184; 183 200 184 201; 200 217 201 218; 217 234 218 235; 234 251 235 252; 251 268 252 269; 268 285 269 286; 14 31 15 32; 31 48 32 49; 48 65 49 66; 65 82 66 83; 82 99 83 100; 99 116 100 117; 116 133 117 134; 133 150 134 151; 150 167 151 168; 167 184 168 185; 184 201 185 202; 201 218 202 219; 218 235 219 236; 235 252 236 253; 252 269 253 270; 269 286 270 287; 15 32 16 33; 32 49 33 50; 49 66 50 67; 66 83 67 84; 83 100 84 101; 100 117 101 118; 117 134 118 135; 134 151 135 152; 151 168 152 169; 168 185 169 186; 185 202 186 203; 202 219 203 220; 219 236 220 237; 236 253 237 254; 253 270 254 271; 270 287 271 288; 16 33 17 34; 33 50 34 51; 50 67 51 68; 67 84 68 85; 84 101 85 102; 101 118 102 119; 118 135 119 136; 135 152 136 153; 152 169 153 170; 169 186 170 187; 186 203 187 204; 203 220 204 221; 220 237 221 238; 237 254 238 255; 254 271 255 272; 271 288 272 289"""
# Data conversion
raw_lines = EToV.strip().replace(";", "\n").splitlines()
etov_clean = []
for line in raw_lines:
    tokens = line.strip().split()
    if len(tokens) == 4:
        etov_clean.append([int(t) for t in tokens])
EToV = np.array(etov_clean) - 1

# Infer number of columns
bottom_y = VY[EToV].min(axis=1)
min_y = bottom_y.min()
bottom_quads = [i for i, y in enumerate(bottom_y) if y == min_y]
num_cols = len(bottom_quads)
num_rows = len(EToV) // num_cols

# Taylor-Green Vortex velocity field
v = 1.0
def F(v, t): return np.exp(-2 * v * t)
def u_field(x, y, t): return np.sin(x)*np.cos(y)*F(v, t)
def v_field(x, y, t): return -np.cos(x)*np.sin(y)*F(v, t)

def invert_bilinear_map(x_target, y_target, x_nodes, y_nodes, tol=1e-12, max_iter=20):
    x_hat, y_hat = 0.0, 0.0
    for _ in range(max_iter):
        phi = np.array([
            0.25 * (1 - x_hat) * (1 - y_hat),
            0.25 * (1 + x_hat) * (1 - y_hat),
            0.25 * (1 + x_hat) * (1 + y_hat),
            0.25 * (1 - x_hat) * (1 + y_hat)
        ])
        x = phi.dot(x_nodes)
        y = phi.dot(y_nodes)
        fx, fy = x - x_target, y - y_target
        if np.hypot(fx, fy) < tol:
            return x_hat, y_hat

        dphidx = np.array([
            -0.25 * (1 - y_hat),
             0.25 * (1 - y_hat),
             0.25 * (1 + y_hat),
            -0.25 * (1 + y_hat)
        ])
        dphidy = np.array([
            -0.25 * (1 - x_hat),
            -0.25 * (1 + x_hat),
             0.25 * (1 + x_hat),
             0.25 * (1 - x_hat)
        ])
        J = np.array([
            [dphidx.dot(x_nodes), dphidy.dot(x_nodes)],
            [dphidx.dot(y_nodes), dphidy.dot(y_nodes)]
        ])
        try:
            delta = np.linalg.solve(J, [fx, fy])
        except np.linalg.LinAlgError:
            raise RuntimeError("Singular Jacobian in invert_bilinear_map")
        x_hat -= delta[0]
        y_hat -= delta[1]

    raise RuntimeError("Newton did not converge in invert_bilinear_map")


def lookup_element(x_target, y_target, VX, VY, EToV):
    tol = 1e-8
    for elem, nodes in enumerate(EToV):
        quad_ids = [nodes[0], nodes[1], nodes[3], nodes[2]]
        x_nodes = VX[quad_ids]
        y_nodes = VY[quad_ids]

        if not (x_nodes.min() - tol <= x_target <= x_nodes.max() + tol and
                y_nodes.min() - tol <= y_target <= y_nodes.max() + tol):
            continue

        try:
            x_hat, y_hat = invert_bilinear_map(x_target, y_target, x_nodes, y_nodes)
        except RuntimeError:
            continue

        if -1 - tol <= x_hat <= 1 + tol and -1 - tol <= y_hat <= 1 + tol:
            return elem, x_hat, y_hat

    return None


def velocity_bilinear_interpolation(x, y, t, xvel, yvel):
    # pass the mesh into lookup_element
    res = lookup_element(x, y, VX, VY, EToV)
    if res is None:
        return None
    elem, xh, yh = res

    node_ids = EToV[elem][[0,1,3,2]]
    u_loc = [xvel[i] for i in node_ids]
    v_loc = [yvel[i] for i in node_ids]

    phi = np.array([
        0.25 * (1 - xh) * (1 - yh),
        0.25 * (1 + xh) * (1 - yh),
        0.25 * (1 + xh) * (1 + yh),
        0.25 * (1 - xh) * (1 + yh)
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

# Time step and particles
dt = 0.05
n_steps = 50
a, b, n_particles = 0.2, 0.1, 2000
thetas = np.linspace(0, 4*np.pi, n_particles)
initial_positions = [((a + b*t)*np.cos(t), (a + b*t)*np.sin(t)) for t in thetas]

# Build interpolated trajectories
trajectories = [[pos] for pos in initial_positions]
for idx, (x0, y0) in enumerate(initial_positions):
    x, y, t = x0, y0, 0.0

    for _ in range(n_steps):
        # 1) Snapshot the nodal velocity at current time t
        u0 = u_field(VX, VY, t)
        v0 = v_field(VX, VY, t)

        # 2) Snapshot the nodal velocity at time t + dt
        u1 = u_field(VX, VY, t + dt)
        v1 = v_field(VX, VY, t + dt)

        # 3) Advance one RK2 step using the two snapshots
        x, y, t, _ = RK2_time_dependent(
            x, y, t, dt,
            u0, v0,    # velocity field at time t
            u1, v1     # velocity field at time t+dt
        )

        # 4) Store the new position
        trajectories[idx].append((x, y))

# Convert trajectories to numpy arrays
traj_np = [np.array(traj) for traj in trajectories]

# Analytic RK2 function (define first so you can use it)
def RK2_analytic(x, y, t, dt):
    u1, v1 = u_field(x, y, t), v_field(x, y, t)
    x_mid  = x + 0.5 * dt * u1
    y_mid  = y + 0.5 * dt * v1
    t_mid  = t + 0.5 * dt
    u2, v2 = u_field(x_mid, y_mid, t_mid), v_field(x_mid, y_mid, t_mid)
    return x + dt * u2, y + dt * v2

# Build analytic trajectories for 2000 particles (matching approximated)
traj_analytic_full = []
for x0, y0 in initial_positions:
    x, y, t = x0, y0, 0.0
    single = [(x, y)]
    for _ in range(n_steps):
        x, y = RK2_analytic(x, y, t, dt)
        t += dt
        single.append((x, y))
    traj_analytic_full.append(np.array(single))

# Convert lists to numpy arrays for vectorized error computation
traj_num_arr = np.array(traj_np)           # shape (2000, n_steps+1, 2)
traj_exact_arr = np.array(traj_analytic_full)  # shape (2000, n_steps+1, 2)

# Compute Euclidean distances at each time step for each particle
distances = np.linalg.norm(traj_num_arr - traj_exact_arr, axis=2)  # shape (2000, n_steps+1)

# Mean error per particle (average over time steps)
mean_error_per_particle = np.mean(distances, axis=1)

# Aggregate mean and max error over all particles and all time steps
mean_error = np.mean(mean_error_per_particle)
max_error = np.max(distances)

print(f"Mean Euclidean error over all trajectories: {mean_error:.4e}")
print(f"Max Euclidean error over all trajectories:  {max_error:.4e}")

fig, ax = plt.subplots(figsize=(6,6))
ax.set_title("Taylor-Green Vortex Fluid Flow", fontsize=16, pad=15)
ax.set_facecolor('white')

# Velocity field arrows
nx, ny = 15, 15
xg = np.linspace(VX.min(), VX.max(), nx)
yg = np.linspace(VY.min(), VY.max(), ny)
Xg, Yg = np.meshgrid(xg, yg)
Ug = u_field(Xg, Yg, 0.0)
Vg = v_field(Xg, Yg, 0.0)
ax.quiver(Xg, Yg, Ug, Vg, color='gray', alpha=0.5)

# Draw mesh
for quad in EToV:
    ids = [quad[0], quad[1], quad[3], quad[2], quad[0]]
    ax.plot(VX[ids], VY[ids], color='lightgrey', linewidth=0.8)

# Plot interpolated trajectories
cmap = LinearSegmentedColormap.from_list('blue_green', ["#1023f8", "#44c9bc", "#74F498"])
for traj in traj_np:
    N = len(traj)
    pts  = traj.reshape(-1,1,2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    t_norm = np.linspace(0, 1, N-1)
    lc = LineCollection(segs, array=t_norm, cmap=cmap, linewidth=1.5)
    ax.add_collection(lc)
    interval = max(N//20, 1)
    markers = traj[::interval]
    t_mark  = np.linspace(0,1,len(markers))
    ax.scatter(markers[:,0], markers[:,1], c=t_mark, cmap=cmap, s=10, edgecolors='none')

# Blue→green normalized‐time bar for the RK2 (interpolated) trajectories
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
sm.set_array([])
cbar = fig.colorbar(
    sm, ax=ax,
    orientation='horizontal',
    fraction=0.04,
    pad=0.10,
    aspect=30,
    shrink=0.8,
    ticks=[0, 1]  
)
cbar.set_ticklabels(['t = 0', 't = 2.5'])  # actual time labels
cbar.set_label('Approximated Paths (RK2)', labelpad=0.1, fontsize=12)
cbar.ax.xaxis.set_tick_params(labelsize=12, width=1, length=3, direction='out')
cbar.outline.set_linewidth(0)

ax.set_aspect('equal','box')
ax.set_xlim(VX.min()-0.1, VX.max()+0.1)
ax.set_ylim(VY.min()-0.1, VY.max()+0.1)
ax.axis('off')

plt.tight_layout()
plt.savefig("taylor.png", dpi=300, bbox_inches='tight')
plt.show()