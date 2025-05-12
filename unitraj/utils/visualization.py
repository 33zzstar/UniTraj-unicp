import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os ,sys
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch
import matplotlib.transforms as transforms
parentdir = '/zzs/UniTraj/unitraj'

sys.path.insert(0,parentdir) 
from Bernstein import (
    bernstein_poly,
    bernstein_curve,
    fit_bernstein_curve,
    apply_kalman_filter
)
# input
# ego: (16,3)
# agents: (16,n,3)
# map: (150,n,3)

# visualize all of the agents and ego in the map, the first dimension is the time step,
# the second dimension is the number of agents, the third dimension is the x,y,theta of the agent
# visualize ego and other in different colors, visualize past and future in different colors,past is the first 4 time steps, future is the last 12 time steps
# visualize the map, the first dimension is the lane number, the second dimension is the x,y,theta of the lane
# you can discard the last dimension of all the elements

def visualize_prediction(batch, prediction, model_cfg, draw_index=0):
    # 保留原有函数内容
    def draw_line_with_mask(point1, point2, color, line_width=1.5, label=None):
        """绘制带掩码的线段"""
        ax.plot([point1[0], point2[0]], [point1[1], point2[1]], linewidth=line_width, color=color, label=label)

    def draw_line_with_point(point1, point2, color, line_width=0.5, label=None):
        """绘制带掩码的线段和点标记"""
        # 绘制线段
        ax.plot([point1[0], point2[0]], [point1[1], point2[1]], 
                linewidth=line_width, zorder=4,
                color=color, label=label)
        
        # 绘制端点圆圈
        ax.plot(point1[0], point1[1], 
                'o',                    # 'o' 表示圆形标记
                color=color,            # 圆点颜色
                zorder=4,
                markersize=0.2,          # 圆点大小
                markerfacecolor=color,  # 圆点填充颜色
                markeredgecolor=color)  # 圆点边框颜色
        
        ax.plot(point2[0], point2[1], 
                'o', 
                color=color, 
                zorder=4,
                markersize=0.2,
                markerfacecolor=color,
                markeredgecolor=color)

    def interpolate_color(t, total_t):
        """非自车轨迹的颜色插值(浅绿到深绿)"""
        start_color = (0.56, 0.93, 0.56)  # 浅绿色
        end_color = (0, 0.5, 0)  # 深绿色
        return tuple((1 - t/total_t) * s + (t/total_t) * e 
                    for s, e in zip(start_color, end_color))

    def interpolate_color_ego(t, total_t):
        # Start is red, end is blue """自车轨迹的颜色插值(红到蓝)"""
        return (1 - t / total_t, 0, t / total_t)

    def draw_trajectory(trajectory, line_width=2, color=None, ego=False):
        points = trajectory[:, :2]
        valid_mask = (points[:, 0] != 0)
        valid_points = points[valid_mask]

        if len(valid_points) < 2:
            return

        # 构建分段线段
        segments = np.array([valid_points[:-1], valid_points[1:]]).transpose(1, 0, 2)

        num_segments = len(segments)

        if color is not None:
            colors = [color] * num_segments
        else:
            if ego:
                # 红到蓝
                colors = [(1 - i / num_segments, 0, i / num_segments) for i in range(num_segments)]
            else:
                # 绿到蓝
                colors = [(0, 1 - i / num_segments, i / num_segments) for i in range(num_segments)]

        # 使用 LineCollection 批量绘制渐变线段
        lc = LineCollection(segments, colors=colors, linewidths=line_width, alpha=0.6, zorder=3)
        ax.add_collection(lc)

    # 添加控制点相关函数
    def fit_trajectory_to_control_points(trajectory):
        """将轨迹转换为控制点并生成拟合曲线"""
        # 过滤掉无效点
        valid_points = trajectory[trajectory[:, 0] != 0]
        num_points = len(valid_points)
        if len(valid_points) < 2:  # 确保有足够的点
            return None, None
            
        # 应用卡尔曼滤波
        filtered_trajectory = apply_kalman_filter(valid_points)
        
        # 拟合控制点
        degree = 5  # 控制点数量
        control_points = fit_bernstein_curve(filtered_trajectory, degree)
        
        # 生成拟合曲线点
        t = np.linspace(0, 1, num_points)
        fitted_curve = np.array([bernstein_curve(control_points, ti) for ti in t])
        
        return control_points, fitted_curve

    def calculate_view_range():
        """计算合适的视野范围，以自车轨迹为重点"""
        # 收集自车所有相关轨迹点
        ego_points = []
        
        # 添加历史轨迹点
        ego_hist_traj = past_traj[ego_index, :, :2]
        ego_points.extend(ego_hist_traj[ego_hist_traj[:, 0] != 0])
        
        # 添加真实未来轨迹点
        ego_future_traj = future_traj[ego_index, :, :2]
        ego_points.extend(ego_future_traj[ego_future_traj[:, 0] != 0])
        
        # 添加预测轨迹点
        pred_points = pred_future_traj[max_prob_idx, :, :2]
        ego_points.extend(pred_points[pred_points[:, 0] != 0])
        
        # 转换为numpy数组
        ego_points = np.array(ego_points)
        
        # 计算自车轨迹的范围
        max_x = np.max(ego_points[:, 0])
        min_x = np.min(ego_points[:, 0])
        max_y = np.max(ego_points[:, 1])
        min_y = np.min(ego_points[:, 1])
        
        # 计算中心点（使用自车当前位置）
        center_x = ego_last_pos[0]
        center_y = ego_last_pos[1]
        
        # 计算需要的视野范围（考虑边距）
        margin = 15  # 减小边距到15米
        range_x = max(abs(max_x - center_x), abs(min_x - center_x)) + margin
        range_y = max(abs(max_y - center_y), abs(min_y - center_y)) + margin
        
        # 取较大的范围确保视野是正方形，但设置上限
        view_range = min(max(range_x, range_y), 20)  # 限制最大视野范围为35米
        
        return center_x, center_y, view_range
    
    # 提取数据
    batch = batch['input_dict']
    map_lanes = batch['map_polylines'][draw_index].cpu().numpy()
    map_mask = batch['map_polylines_mask'][draw_index].cpu().numpy()
    past_traj = batch['obj_trajs'][draw_index].cpu().numpy()
    future_traj = batch['obj_trajs_future_state'][draw_index].cpu().numpy()
    past_traj_mask = batch['obj_trajs_mask'][draw_index].cpu().numpy()
    future_traj_mask = batch['obj_trajs_future_mask'][draw_index].cpu().numpy()
    pred_future_prob = prediction['predicted_probability'][draw_index].detach().cpu().numpy()
    pred_future_traj = prediction['predicted_trajectory'][draw_index].detach().cpu().numpy()
    ego_index = batch['track_index_to_predict'][draw_index].item()  # 获取自车索引
    ego_last_pos = pred_future_traj[0, 1, :2]  # 获取最后一帧位置
    map_xy = map_lanes[..., :2]
    ego_history_traj = None
    ego_future_traj = None

    # 新增：从batch中提取映射轨迹数据
    history_mapped_trajectories = None
    future_mapped_trajectories = None
    if 'history_ctrl_mapped_trajectories' in batch:
        history_mapped_trajectories = batch['history_ctrl_mapped_trajectories'][draw_index].cpu().numpy()
    if 'history_ctrl_mapped_trajectories' in batch:
        future_mapped_trajectories = batch['future_ctrl_mapped_trajectories'][draw_index].cpu().numpy()

    # 在提取控制点的代码段添加条件判断
    if model_cfg.get('unicp', False):
        # 提取控制点数据
        history_control_points = batch['history_control_points'][draw_index].cpu().numpy() 
        future_control_points = batch['future_control_points'][draw_index].cpu().numpy()
    else:
        history_control_points = None
        future_control_points = None

    map_type = map_lanes[..., 0, -20:]

    # draw map
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    first_map_element = True  # 添加标志变量
    # Plot the map with mask check
    for idx, lane in enumerate(map_xy):
        lane_type = map_type[idx]
        # convert onehot to index
        lane_type = np.argmax(lane_type)
        if lane_type in [1, 2, 3]:
            continue
        for i in range(len(lane) - 1):
            if map_mask[idx, i] and map_mask[idx, i + 1]:
                if first_map_element:  # 只有第一次绘制时添加图例
                    draw_line_with_mask(lane[i], lane[i + 1], 
                                    color='grey', 
                                    line_width=1.5,
                                    label='Map Elements')
                    first_map_element = False  # 设置标志为False，之后不再添加图例
                else:
                    draw_line_with_mask(lane[i], lane[i + 1], 
                                    color='grey', 
                                    line_width=1.5)
    
    # 绘制历史轨迹
    for idx, traj in enumerate(past_traj[:,:,:2]):
        if idx == ego_index:
            ego_history_traj = traj
            # 自车历史轨迹用黑色，使用点表示
            history_GT_control_points, fitted_curve = fit_trajectory_to_control_points(ego_history_traj)
            if history_GT_control_points is not None:
                # 绘制控制点
                ax.scatter(history_GT_control_points[:, 0], history_GT_control_points[:, 1], 
                color='red', marker='^', s=0.3, zorder=5,
                label='Historical Control Points')
                        
                # 绘制拟合曲线
                ax.plot(fitted_curve[:, 0], fitted_curve[:, 1],
                    color='red', linewidth=0.5, zorder=5,
                    label='Historical Fitted Trajectory')
            # 新增：绘制自车历史轨迹离散点
            valid_history_points = traj[traj[:, 0] != 0]
            if len(valid_history_points) > 0:
                ax.scatter(valid_history_points[:, 0], valid_history_points[:, 1],
                        color='red',  # 使用红色，与拟合曲线一致
                        marker='o',
                        s=1.0,  # 点稍大，以便于区分
                        alpha=0.9,
                        zorder=6,  # 确保点在最上层
                        label='Ego History GT Points')
                
            
            # 添加自车历史轨迹起点长方形标记
            valid_points = traj[traj[:, 0] != 0]
            if len(valid_points) >= 2:
                # 获取起点位置
                x, y = valid_points[0]
                
                # 获取方向向量并计算角度
                direction = valid_points[1] - valid_points[0]
                angle = np.arctan2(direction[1], direction[0])
                
                # 创建长方形（长宽比3:1，长度为2米）
                length, width = 2.0, 0.8
                rect = plt.Rectangle((-length/2, -width/2), length, width, 
                                angle=0, 
                                facecolor='red', 
                                edgecolor='black', 
                                alpha=0.7,
                                zorder=5,
                                label='Ego Start')
                
                # 应用变换
                t = ax.transData
                rot = transforms.Affine2D().rotate_around(0, 0, angle).translate(x, y) + t
                rect.set_transform(rot)
                
                # 添加到画布
                ax.add_patch(rect)
        elif idx == 0:  # 只在第一个其他车辆添加图例
            draw_trajectory(traj, line_width=2, color=None, ego=False,
                        label='Other Vehicles')
        else:
            # 其他车辆使用默认渐变色
            draw_trajectory(traj, line_width=2, color=None, ego=False)

    # 绘制实际未来轨迹
    for idx, traj in enumerate(future_traj[:,:,:2]):
        if idx == ego_index:
            ego_future_traj = traj
            # 对未来轨迹进行控制点拟合
            future_GT_control_points, fitted_curve_future = fit_trajectory_to_control_points(ego_future_traj)
            
            if future_GT_control_points is not None and fitted_curve_future is not None:
                # 绘制未来轨迹的控制点 - 使用黑色
                ax.scatter(future_GT_control_points[:, 0], future_GT_control_points[:, 1], 
                         color='gray', marker='^', s=0.3, zorder=5,
                         label='Future Control Points')
                
                # 绘制未来轨迹的拟合曲线 - 使用黑色
                ax.plot(fitted_curve_future[:, 0], fitted_curve_future[:, 1],
                       color='gray', linewidth=0.2, zorder=6,
                       label='Future Fitted Trajectory')
            # 新增：绘制自车未来轨迹离散点
            valid_future_points = traj[traj[:, 0] != 0]
            if len(valid_future_points) > 0:
                ax.scatter(valid_future_points[:, 0], valid_future_points[:, 1],
                        color='blue',  # 使用蓝色，表示未来
                        marker='o',
                        s=1.0,  # 点稍大，以便于区分
                        alpha=0.9,
                        zorder=6,  # 确保点在最上层
                        label='Ego Future GT Points')
        else:
            # 其他车辆使用默认渐变色
            draw_trajectory(traj, line_width=2, color=None, ego=False)

    # 绘制历史映射轨迹时添加显示离散点
    if history_mapped_trajectories is not None:
        # 确保是有效的数据格式
        if isinstance(history_mapped_trajectories, np.ndarray):
            # 如果是单轨迹
            if history_mapped_trajectories.ndim == 2:
                # 绘制轨迹，使用紫色
                valid_points = history_mapped_trajectories[history_mapped_trajectories[:, 0] != 0]
                if len(valid_points) >= 2:
                    # 创建分段线段
                    segments = np.array([valid_points[:-1, :2], valid_points[1:, :2]]).transpose(1, 0, 2)
                    # 使用固定颜色或颜色梯度
                    colors = [(0.5, 0, 0.5)] * len(segments)  # 紫色
                    lc = LineCollection(segments, colors=colors, linewidths=1.5, alpha=0.7, zorder=4)
                    ax.add_collection(lc)
                    
                    # 新增：绘制离散点
                    ax.scatter(valid_points[:, 0], valid_points[:, 1], 
                            color=(0.5, 0, 0.5),  # 与线段相同的紫色
                            marker='o', 
                            s=2.5,  # 点的大小
                            alpha=0.9,  # 透明度稍高于线段
                            zorder=5,  # 确保点在线段上方
                            label='History Mapped Points')
                    
                    # 添加线段图例
                    ax.plot([], [], color=(0.5, 0, 0.5), linewidth=1.5, label='History Mapped')
                    
            # 如果是多轨迹
            elif history_mapped_trajectories.ndim == 3:
                first_traj = True
                first_points = True
                for traj in history_mapped_trajectories:
                    valid_points = traj[traj[:, 0] != 0]
                    if len(valid_points) >= 2:
                        segments = np.array([valid_points[:-1, :2], valid_points[1:, :2]]).transpose(1, 0, 2)
                        colors = [(0.5, 0, 0.5)] * len(segments)  # 紫色
                        lc = LineCollection(segments, colors=colors, linewidths=1.5, alpha=0.7, zorder=4)
                        ax.add_collection(lc)
                        
                        # 新增：绘制离散点
                        if first_points:
                            ax.scatter(valid_points[:, 0], valid_points[:, 1], 
                                    color=(0.5, 0, 0.5),
                                    marker='o', 
                                    s=2.5,
                                    alpha=0.9,
                                    zorder=5,
                                    label='History Mapped Points')
                            first_points = False
                        else:
                            ax.scatter(valid_points[:, 0], valid_points[:, 1], 
                                    color=(0.5, 0, 0.5),
                                    marker='o', 
                                    s=2.5,
                                    alpha=0.9,
                                    zorder=5)
                        
                        # 只为第一条轨迹添加图例
                        if first_traj:
                            ax.plot([], [], color=(0.5, 0, 0.5), linewidth=1.5, label='History Mapped')
                            first_traj = False

    # 绘制未来映射轨迹时添加显示离散点
    if future_mapped_trajectories is not None:
        # 确保是有效的数据格式
        if isinstance(future_mapped_trajectories, np.ndarray):
            # 如果是单轨迹
            if future_mapped_trajectories.ndim == 2:
                # 绘制轨迹，使用青色
                valid_points = future_mapped_trajectories[future_mapped_trajectories[:, 0] != 0]
                if len(valid_points) >= 2:
                    # 创建分段线段
                    segments = np.array([valid_points[:-1, :2], valid_points[1:, :2]]).transpose(1, 0, 2)
                    # 使用固定颜色或颜色梯度
                    colors = [(0, 0.5, 0.5)] * len(segments)  # 青色
                    lc = LineCollection(segments, colors=colors, linewidths=1.5, alpha=0.7, zorder=4)
                    ax.add_collection(lc)
                    
                    # 新增：绘制离散点
                    ax.scatter(valid_points[:, 0], valid_points[:, 1], 
                            color=(0, 0.5, 0.5),  # 与线段相同的青色
                            marker='o', 
                            s=2.5,  # 点的大小
                            alpha=0.9,  # 透明度稍高于线段
                            zorder=5,  # 确保点在线段上方
                            label='Future Mapped Points')
                    
                    # 添加图例
                    ax.plot([], [], color=(0, 0.5, 0.5), linewidth=1.5, label='Future Mapped')
                    
                    # 为最后一个点添加箭头
                    if len(valid_points) >= 2:
                        arrow = FancyArrowPatch(
                            posA=valid_points[-2, :2], 
                            posB=valid_points[-1, :2],
                            arrowstyle='->', 
                            color=(0, 0.5, 0.5),
                            mutation_scale=10, 
                            linewidth=1.5, 
                            zorder=5
                        )
                        ax.add_patch(arrow)
                        
            # 如果是多轨迹
            elif future_mapped_trajectories.ndim == 3:
                first_traj = True
                first_points = True
                for traj in future_mapped_trajectories:
                    valid_points = traj[traj[:, 0] != 0]
                    if len(valid_points) >= 2:
                        segments = np.array([valid_points[:-1, :2], valid_points[1:, :2]]).transpose(1, 0, 2)
                        colors = [(0, 0.5, 0.5)] * len(segments)  # 青色
                        lc = LineCollection(segments, colors=colors, linewidths=1.5, alpha=0.7, zorder=4)
                        ax.add_collection(lc)
                        
                        # 新增：绘制离散点
                        if first_points:
                            ax.scatter(valid_points[:, 0], valid_points[:, 1], 
                                    color=(0, 0.5, 0.5),
                                    marker='o', 
                                    s=2.5,
                                    alpha=0.9,
                                    zorder=5,
                                    label='Future Mapped Points')
                            first_points = False
                        else:
                            ax.scatter(valid_points[:, 0], valid_points[:, 1], 
                                    color=(0, 0.5, 0.5),
                                    marker='o', 
                                    s=2.5,
                                    alpha=0.9,
                                    zorder=5)
                        
                        # 只为第一条轨迹添加图例
                        if first_traj:
                            ax.plot([], [], color=(0, 0.5, 0.5), linewidth=1.5, label='Future Mapped')
                            first_traj = False
                        
                        # 为每条轨迹的最后一个点添加箭头
                        if len(valid_points) >= 2:
                            arrow = FancyArrowPatch(
                                posA=valid_points[-2, :2], 
                                posB=valid_points[-1, :2],
                                arrowstyle='->', 
                                color=(0, 0.5, 0.5),
                                mutation_scale=10, 
                                linewidth=1.5, 
                                zorder=5
                            )
                            ax.add_patch(arrow)

    # 找出概率最高的轨迹索引
    max_prob_idx = np.argmax(pred_future_prob)
    pred_traj = pred_future_traj[max_prob_idx]

    # 在右上角添加图例
    ax.legend(loc='upper right', 
             fontsize=3,
             bbox_to_anchor=(1.15, 1.0),
             frameon=True,
             fancybox=True,
             shadow=True)
    
    # 计算视野范围
    center_x, center_y, view_range = calculate_view_range()
    
    # 设置坐标轴范围，以自车为中心
    ax.set_xlim(center_x - view_range, center_x + view_range)
    ax.set_ylim(center_y - view_range, center_y + view_range)
    
    # 设置其他属性
    ax.set_aspect('equal')  # 保持横纵比例相等
    ax.axis('off')  # 隐藏坐标轴
    ax.grid(True)  # 显示网格

    # 保存路径
    save_path = "/data1/data_zzs/plt_sample" 
    #子目录
    save_commit = '4.15.2_ctrl'
    #创建完整目录
    save_dir = os.path.join(save_path)
    # 确保目录存在
    os.makedirs(save_dir, exist_ok=True)
    # 生成文件名和完整的保存路径
    filename = f'trajectory_plot_{save_commit}.png'
    save_path = os.path.join(save_dir, filename)

    plt.savefig(save_path, dpi=500, bbox_inches='tight')
    # 清理图像以释放内存
    # plt.close()
    
    return plt, ego_history_traj, ego_future_traj, history_GT_control_points, future_GT_control_points
