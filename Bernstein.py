import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
from filterpy.kalman import KalmanFilter

def bernstein_poly(n, i, t):
    """计算伯恩斯坦多项式。"""
    return comb(n, i) * (t**i) * (1 - t)**(n - i)

def bernstein_curve(points, t):
    """使用伯恩斯坦多项式计算曲线上的点。"""
    n = len(points) - 1
    
    # 检查t是标量还是数组
    if np.isscalar(t):
        # t是标量，返回曲线上的一个点
        curve_point = np.zeros_like(points[0], dtype=float)
        for i, point in enumerate(points):
            curve_point += bernstein_poly(n, i, t) * point
        return curve_point
    else:
        # t是数组，返回曲线上的一组点
        t = np.asarray(t)
        curve_points = np.zeros((len(t), points.shape[1]), dtype=float)
        
        for ti_idx, ti in enumerate(t):
            for i, point in enumerate(points):
                curve_points[ti_idx] += bernstein_poly(n, i, ti) * point
                
        return curve_points

def fit_bernstein_curve(trajectory, degree=5):
    """使用最小二乘法拟合轨迹到伯恩斯坦曲线。"""
    num_points = int(len(trajectory))
    t = np.linspace(0, 1, num_points)

    # 构建设计矩阵
    design_matrix = np.zeros((num_points, degree + 1))
    for i in range(num_points):
        for j in range(degree + 1):
            design_matrix[i, j] = bernstein_poly(degree, j, t[i])
    
    # 使用最小二乘法求解控制点
    control_points = np.linalg.lstsq(design_matrix, trajectory, rcond=None)[0]
    return control_points

def generate_realistic_trajectory(num_points=50):
    """生成更像真实车辆轨迹的 2D 轨迹点（左转 + 避让）。"""
    np.random.seed(0)  # 保证结果可重复

    # 初始位置和方向
    x = 0.0
    y = 0.0
    heading = 0.0  # 初始方向为水平向右 (0 度)

    trajectory = []
    for i in range(num_points):
        # 模拟左转（前 20 个点向左转向）
        if i < 20:
            heading += np.deg2rad(1.5)  # 每次左转 1.5 度 (角度转换为弧度)
        elif i < 35:
            heading += np.deg2rad(-1.5)
        else:
            heading += np.deg2rad(0)
        # 模拟避让（在中间阶段向右偏移）
        if 15 <= i <= 25:
            y += 0.1
        
        
        # 计算下一步的位置
        dx = np.cos(heading) * 0.1 + np.random.randn() * 0.01 #添加一些随机性
        dy = np.sin(heading) * 0.1 + np.random.randn() * 0.01
        x += dx
        y += dy
        
        trajectory.append((x, y))

    return np.array(trajectory)

def apply_kalman_filter(trajectory):
    """初始化卡尔曼滤波器 应用卡尔曼滤波器平滑轨迹。"""
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.x = np.array([trajectory[0,0], 0, trajectory[0,1], 0], dtype=float)  # 初始化状态 (x, dx, y, dy)
    kf.F = np.array([[1, 1, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 1],
                     [0, 0, 0, 1]], dtype=float)  # 状态转移矩阵
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 0, 1, 0]], dtype=float)  # 观测矩阵
    kf.P *= 100. # 较大的初始协方差
    kf.R = 1 #观测噪声方差
    kf.Q = np.eye(4) * 0.05 #过程噪声方差
    
    filtered_trajectory = []
    for point in trajectory:
        kf.predict()
        kf.update(point)
        filtered_trajectory.append(kf.x[[0,2]]) #只保留 x 和 y
    return np.array(filtered_trajectory)

if __name__ == "__main__":
    # 1. 生成随机轨迹
    trajectory = generate_realistic_trajectory(50)
    
    # 2. 应用卡尔曼滤波
    filtered_trajectory = apply_kalman_filter(trajectory)

    # 3. 使用伯恩斯坦多项式拟合曲线
    degree = 5
    control_points = fit_bernstein_curve(filtered_trajectory, degree)

    # 4. 生成拟合曲线上的点
    t = np.linspace(0, 1, 100)
    fitted_curve = np.array([bernstein_curve(control_points, ti) for ti in t])

    # 5. 可视化
    plt.figure(figsize=(10, 6))
    
    # 绘制原始轨迹点
    plt.scatter(trajectory[:, 0], trajectory[:, 1], label='Original Points', color='lightgray', marker='o', s=20)

    # 绘制卡尔曼滤波后的轨迹点
    plt.scatter(filtered_trajectory[:, 0], filtered_trajectory[:, 1], label='Filtered Points', color='blue', marker='x', s=20)

    # 绘制拟合曲线
    plt.plot(fitted_curve[:, 0], fitted_curve[:, 1], label='Fitted Curve', color='red', linewidth=2)

    # 绘制控制点
    plt.scatter(control_points[:, 0], control_points[:, 1], label='Control Points', color='green', marker='^', s=50)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Bernstein Curve Fitting with Kalman Filter')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()