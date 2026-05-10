import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import os

# ============================================================
# Save Directory
# ============================================================
save_dir = r"D:\HuaweiMoveData\Users\gaoyu\Desktop"

# 自动创建文件夹（保险）
os.makedirs(save_dir, exist_ok=True)
# ============================================================
# Cart-Pole Swing-Up
# Direct Transcription + RK4 + TV-LQR Tracking
# ============================================================

# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------
mc = 1.0
m = 0.1
l = 0.5
g = 9.81

nx = 4
nu = 1

T = 10
N = 60  # 保持 N=100
dt = T / N

x_start = np.array([0.0, 0.0, 0.0, 0.0])
x_goal = np.array([0, np.pi, 0.0, 0.0])

x_max = 2.0
u_max = 25

# ============================================================
# Real system parameters (model mismatch)
# ============================================================
m_real = 0.11
l_real = 0.52
mc_real = 1.05

# ============================================================
# Dynamics
# ============================================================
def dynamics(x, u):
    X, th, dX, dth = x
    s = np.sin(th)
    c = np.cos(th)
    D = mc + m * s**2
    ddX = (u + m * l * dth**2 * s - m * g * s * c) / D
    ddth = (u * c + m * l * dth**2 * s * c - (mc + m) * g * s) / (l * D)
    return np.array([dX, dth, ddX, ddth])

# ============================================================
# Real Dynamics (Model Mismatch)
# ============================================================
def dynamics_real(x, u):
    X, th, dX, dth = x
    s = np.sin(th)
    c = np.cos(th)
    D = mc_real + m_real * s**2
    ddX = (u + m_real * l_real * dth**2 * s - m_real * g * s * c) / D
    ddth = (u * c + m_real * l_real * dth**2 * s * c - (mc_real + m_real) * g * s) / (l_real * D)
    ddX -= 0.15 * dX
    ddth -= 0.05 * dth
    return np.array([dX, dth, ddX, ddth])

# ============================================================
# RK4 Integrator (保持不变)
# ============================================================
def rk4_step(x, u):
    k1 = dynamics(x, u)
    k2 = dynamics(x + 0.5 * dt * k1, u)
    k3 = dynamics(x + 0.5 * dt * k2, u)
    k4 = dynamics(x + dt * k3, u)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def rk4_step_real(x, u):
    k1 = dynamics_real(x, u)
    k2 = dynamics_real(x + 0.5 * dt * k1, u)
    k3 = dynamics_real(x + 0.5 * dt * k2, u)
    k4 = dynamics_real(x + dt * k3, u)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# ============================================================
# 解析雅可比 (优化：避免重复计算)
# ============================================================
def dynamics_jacobian_analytic(x, u):
    x1, x2, x3, x4 = x
    s = np.sin(x2)
    c = np.cos(x2)
    D = mc + m * s**2
    D2 = D * D  # 预计算 D²
    
    # B 矩阵
    B = np.zeros((4, 1))
    B[2, 0] = 1.0 / D
    B[3, 0] = c / (l * D)
    
    # A 矩阵
    A = np.zeros((4, 4))
    A[0, 2] = 1.0
    A[1, 3] = 1.0
    
    # 第三行
    df3_dx4 = 2.0 * m * l * x4 * s / D
    term1 = (m * l * x4**2 * c - m * g * (c**2 - s**2)) / D
    term2 = (2.0 * m * s * c * (u + m * l * x4**2 * s - m * g * s * c)) / D2
    df3_dx2 = term1 - term2
    A[2, 1] = df3_dx2
    A[2, 3] = df3_dx4
    
    # 第四行
    df4_dx4 = 2.0 * m * x4 * s * c / D
    term_a = (-u * s + m * l * x4**2 * (c**2 - s**2) - (mc + m) * g * c) / (l * D)
    term_b = (2.0 * m * s * c * (u * c + m * l * x4**2 * s * c - (mc + m) * g * s)) / (l * D2)
    df4_dx2 = term_a - term_b
    A[3, 1] = df4_dx2
    A[3, 3] = df4_dx4
    
    return A, B

# ============================================================
# 打包/解包函数
# ============================================================
def pack(X, U):
    return np.hstack([X.ravel(), U.ravel()])

def unpack(z):
    X = z[:(N+1)*nx].reshape(N+1, nx)
    U = z[(N+1)*nx:].reshape(N, nu)
    return X, U

# ============================================================
# 目标函数 (优化：使用向量化计算)
# ============================================================
Q = np.diag([1.0, 10, 0.1 ,0.1])
R = 0.03
Qf = np.diag([50, 300, 30, 30])

def objective(z):
    X, U = unpack(z)
    
    # 向量化计算代价
    dx = X - x_goal
    stage_cost = np.sum(dx @ Q * dx) + R * np.sum(U**2)
    
    # 终端代价
    dxf = X[-1] - x_goal
    terminal_cost = dxf @ Qf @ dxf
    
    return stage_cost + terminal_cost

# ============================================================
# 约束函数 (优化：预分配数组)
# ============================================================
def constraints(z):
    X, U = unpack(z)
    
    # 预分配数组
    ceq = np.zeros((N+1, nx))
    
    # 初始条件
    ceq[0] = X[0] - x_start
    
    # 动力学约束
    for k in range(N):
        x_next = rk4_step(X[k], U[k, 0])
        ceq[k+1] = X[k+1] - x_next
    
    return ceq.ravel()

# ============================================================
# 初始猜测 (优化：更合理的初始轨迹)
# ============================================================
def initial_guess():
    X = np.zeros((N+1, nx))
    U = np.zeros((N, nu))
    
    # 更平滑的初始轨迹
    for k in range(N+1):
        alpha = k / N
        # 摆角：三次多项式从0到π
        X[k, 1] = np.pi * (3*alpha**2 - 2*alpha**3)
        # 小车位置：正弦波
        X[k, 0] = 0.5 * np.sin(2 * np.pi * alpha)
        # 速度：从零开始
        X[k, 2] = 0.0
        X[k, 3] = 0.0
    
    # 控制输入：正弦波
    for k in range(N):
        U[k, 0] = 5.0 * np.sin(2 * np.pi * k / N)
    
    return pack(X, U)

# ============================================================
# 边界
# ============================================================
def bounds():
    z0 = initial_guess()
    lb = np.full_like(z0, -np.inf)
    ub = np.full_like(z0, np.inf)
    
    # 状态边界
    for k in range(N+1):
        idx = slice(k*nx, k*nx + nx)
        lb[idx] = [-x_max, -4*np.pi, -20, -20]
        ub[idx] = [x_max, 4*np.pi, 20, 20]
    
    # 控制边界
    uidx = slice((N+1)*nx, None)
    lb[uidx] = -u_max
    ub[uidx] = u_max
    
    return opt.Bounds(lb, ub)

# ============================================================
# 轨迹优化 (优化：调整优化器参数)
# ============================================================
def solve_trajectory():
    z0 = initial_guess()
    
    cons = {'type': 'eq', 'fun': constraints}
    
    print("Solving trajectory optimization...")
    
    # 优化器参数调整
    res = opt.minimize(
        objective,
        z0,
        method='SLSQP',
        bounds=bounds(),
        constraints=cons,
        options={
            'maxiter': 300,      # 减少最大迭代次数
            'ftol': 1e-4,       # 放宽收敛容差
            'disp': True,
            'eps': 1e-6         # 调整数值差分步长
        }
    )
    
    print(f"Optimization status: {res.message}")
    print(f"Number of iterations: {res.nit}")
    print(f"Final cost: {res.fun:.6f}")
    
    X, U = unpack(res.x)
    return X, U

# ============================================================
# TV-LQR
# ============================================================
def tvlqr(X_ref, U_ref):
    Qlqr = np.diag([500, 10, 50,100])
    Rlqr = np.array([[0.2]])
    P = Qf.copy()
    Klist = []
    
    for k in reversed(range(N)):
        Acont, Bcont = dynamics_jacobian_analytic(X_ref[k], U_ref[k, 0])
        A = np.eye(nx) + dt * Acont
        B = dt * Bcont
        
        G = Rlqr + B.T @ P @ B
        K = np.linalg.solve(G, B.T @ P @ A)
        P = Qlqr + A.T @ P @ A - A.T @ P @ B @ K
        Klist.append(K)
    
    return Klist[::-1]

# ============================================================
# 闭环仿真
# ============================================================
def simulate(X_ref, U_ref, Klist):
    x = x_start.copy()
    x += np.array([0.05, 0.12, 0.0, 0.0])  # 初始扰动
    
    sim = [x.copy()]
    u_hist = []
    
    for k in range(N):
        dx = x - X_ref[k]
        u = U_ref[k, 0] - Klist[k] @ dx
        u = np.clip(u.item(), -u_max, u_max)
        u_hist.append(u)
        
        x = rk4_step_real(x, u)
        
        # 过程噪声
        noise = np.array([
            0.002 * np.random.randn(),
            0.003 * np.random.randn(),
            0.02 * np.random.randn(),
            0.02 * np.random.randn()
        ])
        x += noise
        sim.append(x.copy())
    
    return np.array(sim), np.array(u_hist)

# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    print("\n" + "="*50)
    print("DIRECT TRANSCRIPTION + TVLQR")
    print("="*50 + "\n")
    
    # 1. 求解参考轨迹
    print("Step 1: Solving trajectory optimization...")
    X_ref, U_ref = solve_trajectory()
    
    # 2. 计算 TV-LQR 增益
    print("\nStep 2: Computing TV-LQR gains...")
    Klist = tvlqr(X_ref, U_ref)
    print(f"Generated {len(Klist)} gain matrices.")
    
    # 3. 闭环仿真
    print("\nStep 3: Running closed-loop simulation...")
    X_sim, U_sim = simulate(X_ref, U_ref, Klist)
    print("Simulation completed.")
    
    # 4. 性能评估
    final_error = np.linalg.norm(X_sim[-1] - x_goal)
    max_control = np.max(np.abs(U_sim))
    print(f"\nPerformance Metrics:")
    print(f"  Final state error: {final_error:.6f}")
    print(f"  Max control input: {max_control:.2f} N")
    print(f"  Max reference control: {np.max(np.abs(U_ref)):.2f} N")
    
    # 5. 绘图
    print("\nGenerating plots...")
    t = np.linspace(0, T, N+1)
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('State Response under TV‑LQR', fontsize=14)
    
    # 小车位移
    axs[0, 0].plot(t, X_ref[:, 0], 'r--', lw=2, label='Reference')
    axs[0, 0].plot(t, X_sim[:, 0], 'b', lw=2, label='TV‑LQR')
    axs[0, 0].set_ylabel('x [m]')
    axs[0, 0].grid(True)
    axs[0, 0].legend()
    
    # 摆角
    axs[0, 1].plot(t, X_ref[:, 1], 'r--', lw=2, label='Reference')
    axs[0, 1].plot(t, X_sim[:, 1], 'b', lw=2, label='TV‑LQR')
    axs[0, 1].set_ylabel(r'$\theta$ [rad]')
    axs[0, 1].grid(True)
    axs[0, 1].legend()
    
    # 小车速度
    axs[1, 0].plot(t, X_ref[:, 2], 'r--', lw=2)
    axs[1, 0].plot(t, X_sim[:, 2], 'b', lw=2)
    axs[1, 0].set_ylabel(r'$\dot{x}$ [m/s]')
    axs[1, 0].set_xlabel('Time [s]')
    axs[1, 0].grid(True)
    
    # 角速度
    axs[1, 1].plot(t, X_ref[:, 3], 'r--', lw=2)
    axs[1, 1].plot(t, X_sim[:, 3], 'b', lw=2)
    axs[1, 1].set_ylabel(r'$\dot{\theta}$ [rad/s]')
    axs[1, 1].set_xlabel('Time [s]')
    axs[1, 1].grid(True)
    
    plt.tight_layout()
    fig.savefig(
    os.path.join(save_dir, "state_response.png"),
    dpi=600,
    bbox_inches='tight'
    )
    plt.show()
    
    # 控制输入
    plt.figure(figsize=(10, 4))
    plt.plot(t[:-1], U_ref[:, 0], 'r--', lw=2, drawstyle='steps-post', label='Reference')
    plt.plot(t[:-1], U_sim, 'b', lw=2, drawstyle='steps-post', label='TV‑LQR')
    plt.ylabel('u [N]')
    plt.xlabel('Time [s]')
    plt.title('Control Input')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
    os.path.join(save_dir, "control_input.png"),
    dpi=600,
    bbox_inches='tight'
)
    plt.show()
    
    print("\n" + "="*50)
    print("All tasks completed successfully!")
    print("="*50)
    # ============================================================
# 3. 能量曲线 / 相平面轨迹（极限环）
# ============================================================
#3.1 能量曲线
theta = X_sim[:,1]
dtheta = X_sim[:,3]
T_energy = 0.5 * m_real * (l_real * dtheta)**2
V_energy = m_real * g * l_real * (1 - np.cos(theta))
E = T_energy + V_energy

plt.figure(figsize=(9, 4))
plt.plot(t, E, lw=2, color='teal')
plt.xlabel('Time [s]')
plt.ylabel('Mechanical Energy [J]')
plt.title('Energy Evolution')
plt.grid(True)
plt.tight_layout()
plt.savefig(
    os.path.join(save_dir, "energy_evolution.png"),
    dpi=600,
    bbox_inches='tight'
)
plt.show()

# 3.2 相平面（极限环）
plt.figure(figsize=(7, 6))
plt.plot(X_sim[:,1], X_sim[:,3], lw=2, color='darkorange', label='TV‑LQR trajectory')
plt.scatter(X_sim[0,1], X_sim[0,3], s=80, color='red', label='Start')
plt.scatter(X_sim[-1,1], X_sim[-1,3], s=80, color='green', label='End')
plt.xlabel(r'$\theta$ [rad]')
plt.ylabel(r'$\dot{\theta}$ [rad/s]')
plt.title('Phase Portrait (Limit Cycle)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(save_dir, "phase_portrait.png"),
    dpi=600,
    bbox_inches='tight')
plt.show()

# ============================================================
# 动画（内嵌跟踪误差显示）
# ============================================================
from matplotlib.animation import FuncAnimation

err = np.linalg.norm(X_sim - X_ref, axis=1)

fig_anim = plt.figure(figsize=(14,6))
ax_anim = plt.subplot(121)
ax_anim.set_xlim(-2.5, 2.5)
ax_anim.set_ylim(-1.2, 1.2)
ax_anim.set_aspect('equal')
ax_anim.grid(True)
ax_anim.plot([-3,3],[0,0],'k',lw=2)
cart_width,cart_height=0.3,0.18
cart=plt.Rectangle((-cart_width/2,-cart_height/2),cart_width,cart_height,fc='tab:blue')
ax_anim.add_patch(cart)
rod,=ax_anim.plot([],[],'r-',lw=3)
mass,=ax_anim.plot([],[],'ko',ms=10)
ref_rod,=ax_anim.plot([],[],'g--',lw=2,alpha=0.4)
ref_mass,=ax_anim.plot([],[],'go',alpha=0.4)
time_text=ax_anim.text(0.02,0.95,'',transform=ax_anim.transAxes)

ax_err=plt.subplot(122)
ax_err.set_xlim(0,T)
ax_err.set_ylim(0,1.1*np.max(err))
ax_err.set_title('Tracking Error')
ax_err.set_xlabel('Time [s]')
ax_err.set_ylabel(r'$\|x-x_{\mathrm{ref}}\|$')
err_line,=ax_err.plot([],[],'b-',lw=2)
err_point,=ax_err.plot([],[],'ro')

def init():
    rod.set_data([],[]);mass.set_data([],[])
    ref_rod.set_data([],[]);ref_mass.set_data([],[])
    err_line.set_data([],[]);err_point.set_data([],[])
    time_text.set_text('')
    return cart,rod,mass,ref_rod,ref_mass,err_line,err_point,time_text

def animate(i):
    x,th = X_sim[i,0],X_sim[i,1]   
    px=x+l*np.sin(th);py=-l*np.cos(th)
    cart.set_xy((x-cart_width/2,-cart_height/2))
    rod.set_data([x,px],[0,py])
    mass.set_data([px],[py])
    xr,thr = X_ref[i,0],X_ref[i,1]
    pxr=xr+l*np.sin(thr);pyr=-l*np.cos(thr)
    ref_rod.set_data([xr,pxr],[0,pyr])
    ref_mass.set_data([pxr],[pyr])
    err_line.set_data(t[:i+1],err[:i+1])
    err_point.set_data([t[i]], [err[i]])
    time_text.set_text(f't = {t[i]:.2f}s')
    return cart,rod,mass,ref_rod,ref_mass,err_line,err_point,time_text

ani = FuncAnimation(
    fig_anim,
    animate,
    init_func=init,
    frames=len(X_sim),
    interval=40,
    blit=False
)

fig_anim.tight_layout()

video_path = os.path.join(save_dir, "cartpole_tvlqr.mp4")

ani.save(
    video_path,
    writer='ffmpeg',
    fps=int(1/dt)
)

# 5. 独立跟踪误差图
plt.figure(figsize=(8,4))
plt.plot(t, err, lw=2, color='blue')
plt.xlabel('Time [s]')
plt.ylabel(r'$\|x-x_{\mathrm{ref}}\|$')
plt.title('Tracking Error over Time')
plt.grid(True)
plt.tight_layout()
plt.tight_layout()

plt.savefig(
    os.path.join(save_dir, "tracking_error.png"),
    dpi=600,
    bbox_inches='tight'
)

plt.show()
plt.show()