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

def check_loaded_data(plt, data, index=0):
    agents = np.concatenate([data['obj_trajs'][..., :2], data['obj_trajs_future_state'][..., :2]], axis=-2)
    map = data['map_polylines']

    if len(agents.shape) == 4:
        agents = agents[index]
        map = map[index]
        ego_index = data['track_index_to_predict'][index]
        ego_agent = agents[ego_index]
    else:
        ego_index = data['track_index_to_predict']
        ego_agent = agents[ego_index]

    def draw_line_with_mask(point1, point2, color, line_width=4):
        plt.plot([point1[0], point2[0]], [point1[1], point2[1]], linewidth=line_width, color=color)

    def interpolate_color(t, total_t):
        # Start is green, end is blue
        return (0, 1 - t / total_t, t / total_t)

    def interpolate_color_ego(t, total_t):
        # Start is red, end is blue
        return (1 - t / total_t, 0, t / total_t)

    # Plot the map with mask check
    for lane in map:
        # map_one_hot = lane[0, -20:]
        # if np.argmax(map_one_hot) in [1, 2, 3]:
        #     continue
        for i in range(len(lane) - 1):
            draw_line_with_mask(lane[i, :2], lane[i, 6:8], color='grey', line_width=1)

    # Function to draw trajectories
    def draw_trajectory(trajectory, line_width, ego=False):
        total_t = len(trajectory)
        for t in range(total_t - 1):
            if ego:
                color = interpolate_color_ego(t, total_t)
                if trajectory[t, 0] and trajectory[t + 1, 0]:
                    draw_line_with_mask(trajectory[t], trajectory[t + 1], color=color, line_width=line_width)
            else:
                color = interpolate_color(t, total_t)
                if trajectory[t, 0] and trajectory[t + 1, 0]:
                    draw_line_with_mask(trajectory[t], trajectory[t + 1], color=color, line_width=line_width)

    # Draw trajectories for other agents
    for i in range(agents.shape[0]):
        draw_trajectory(agents[i], line_width=2)
    draw_trajectory(ego_agent, line_width=2, ego=True)

    # Set labels, limits, and other properties
    #vis_range = 100
    # plt.xlim(-vis_range + 30, vis_range + 30)
    # plt.ylim(-vis_range, vis_range)
   # plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')
    plt.axis('equal')
    #plt.tight_layout()

    return plt


def visualize_batch_data(ax, data):
    def decode_obj_trajs(obj_trajs):
        obj_trajs_xy = obj_trajs[..., :2]
        obj_lw = obj_trajs[...,-1, 3:5]
        obj_type_onehot = obj_trajs[...,-1, 6:9]
        obj_type = np.argmax(obj_type_onehot, axis=-1)
        obj_heading_encoding = obj_trajs[...,-1, 33:35]
        return obj_trajs_xy, obj_lw, obj_type, obj_heading_encoding
    def decode_map(map):
        map_xy = map[..., :2]
        map_type = map[...,0, 9:29]
        map_type = np.argmax(map_type, axis=-1)
        return map_xy, map_type

    def plot_objects(obj_xy, obj_lw, obj_heading, obj_mask):
        # 在已有的ax对象上进行绘制
        for i in range(len(obj_lw)):
            if obj_mask[i]:
                # 获取对象的长和宽
                length, width = obj_lw[i]

                # 通过sin和cos计算旋转角度
                sin_angle, cos_angle = obj_heading[i]
                angle = np.arctan2(sin_angle, cos_angle)  # 转换为角度（弧度）

                # 获取对象的中心位置
                x, y = obj_xy[i]

                # 创建旋转矩形对象
                rect = plt.Rectangle((-length / 2, -width / 2), length, width, angle=0,
                                     facecolor='none', edgecolor='grey', linewidth=1)

                # 使用转换矩阵将矩形旋转并平移到对象的中心位置
                t = ax.transData
                # 先将矩形平移到中心位置，然后旋转
                rot = plt.matplotlib.transforms.Affine2D().rotate_around(0, 0, angle).translate(x, y) + t
                rect.set_transform(rot)

                # 添加矩形到现有ax中
                ax.add_patch(rect)
    def draw_trajectory(trajectory, line_width, color=None, ego=False):
        def interpolate_color(start_color, end_color, t, total_t):
            """根据 t 和 total_t 插值计算颜色."""
            return [(1 - t / total_t) * start + (t / total_t) * end for start, end in zip(start_color, end_color)]

        def draw_line_with_mask(point1, point2, color, line_width=4):
            ax.plot([point1[0], point2[0]], [point1[1], point2[1]], linewidth=line_width, color=color,alpha=0.5)
        total_t = len(trajectory)
        for t in range(total_t - 1):
            if color is not None:
                # 使用指定颜色
                if trajectory[t, 0] and trajectory[t + 1, 0]:
                    draw_line_with_mask(trajectory[t], trajectory[t + 1], color=color, line_width=line_width)
            else:
                # 使用默认渐变色方案
                if ego:
                    color = interpolate_color_ego(t, total_t)
                else:
                    color = interpolate_color(t, total_t)
                if trajectory[t, 0] and trajectory[t + 1, 0]:
                    draw_line_with_mask(trajectory[t], trajectory[t + 1], color=color, line_width=line_width)

    obj_trajs = data['obj_trajs']
    map = data['map_polylines']

    obj_trajs_xy, obj_lw, obj_type, obj_heading = decode_obj_trajs(obj_trajs)
    obj_trajs_future_state = data['obj_trajs_future_state'][...,:2]
    all_traj = np.concatenate([obj_trajs_xy, obj_trajs_future_state], axis=-2)

    for i in range(obj_trajs.shape[0]):
        if i == data['track_index_to_predict']:
            ego = True
        else:
            ego = False
        draw_trajectory(all_traj[i], line_width=3,ego=ego)

    map_xy, map_type = decode_map(map)
    obj_mask = data['obj_trajs_mask']
    plot_objects(obj_trajs_xy[:,-1],obj_lw, obj_heading, obj_mask[:,-1])

    for indx, type in enumerate(map_type):
        lane = map_xy[indx]
        if type == 0:
            continue
        if type in [1, 2, 3]:
            # 使用灰色虚线表示中心线
            color = 'grey'
            linestyle = 'dotted'
            linewidth = 1
        else:
            color = 'grey'
            linestyle = '-'
            linewidth = 0.2

        # 绘制线条
        for i in range(len(lane) - 1):
            if lane[i, 0] and lane[i + 1, 0]:
                ax.plot([lane[i, 0], lane[i + 1, 0]], [lane[i, 1], lane[i + 1, 1]],
                        linewidth=linewidth, color=color, linestyle=linestyle)

    # 设置坐标轴比例和范围
    vis_range = 35
    ax.set_aspect('equal')
    ax.axis('off')
    ax.grid(True)
    ax.set_xlim(-vis_range, vis_range)
    ax.set_ylim(-vis_range, vis_range)
    #plt.show()
    return ax

def concatenate_images(images, rows, cols):
    # Determine individual image size
    width, height = images[0].size

    # Create a new image with the total size
    total_width = width * cols
    total_height = height * rows
    new_im = Image.new('RGB', (total_width, total_height))

    # Paste each image into the new image
    for i, image in enumerate(images):
        row = i // cols
        col = i % cols
        new_im.paste(image, (col * width, row * height))

    return new_im


def concatenate_varying(image_list, column_counts):
    if not image_list or not column_counts:
        return None

    # Assume all images have the same size, so we use the first one to calculate ratios
    original_width, original_height = image_list[0].size
    total_height = original_height * column_counts[0]  # Total height is based on the first column

    columns = []  # To store each column of images

    start_idx = 0  # Starting index for slicing image_list

    for count in column_counts:
        # Calculate new height for the current column, maintaining aspect ratio
        new_height = total_height // count
        scale_factor = new_height / original_height
        new_width = int(original_width * scale_factor)

        column_images = []
        for i in range(start_idx, start_idx + count):
            # Resize image proportionally
            resized_image = image_list[i].resize((new_width, new_height), Image.Resampling.LANCZOS)
            column_images.append(resized_image)

        # Update start index for the next batch of images
        start_idx += count

        # Create a column image by vertically stacking the resized images
        column = Image.new('RGB', (new_width, total_height))
        y_offset = 0
        for img in column_images:
            column.paste(img, (0, y_offset))
            y_offset += img.height

        columns.append(column)

    # Calculate the total width for the new image
    total_width = sum(column.width for column in columns)

    # Create the final image to concatenate all column images
    final_image = Image.new('RGB', (total_width, total_height))
    x_offset = 0
    for column in columns:
        final_image.paste(column, (x_offset, 0))
        x_offset += column.width

    return final_image

"""可视化预测结果包括历史轨迹、实际未来轨迹和预测的多个可能轨迹"""
def visualize_prediction(batch, prediction, model_cfg,draw_index=0):
        
    def draw_line_with_mask(point1, point2, color, line_width=1.5,label=None):
        """绘制带掩码的线段"""
        ax.plot([point1[0], point2[0]], [point1[1], point2[1]], linewidth=line_width, color=color,label=label)

    def draw_line_with_point(point1, point2, color, line_width=0.5,label=None):
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

       
        # 添加终点箭头
        if ego and len(valid_points) >= 2:
            start_pt = valid_points[-2]
            end_pt = valid_points[-1]
            # arrow = FancyArrowPatch(posA=start_pt, posB=end_pt,
            #                         arrowstyle='->', color=colors[-1] if color is None else color,
            #                         mutation_scale=6, linewidth=0.5, zorder=5,alpha=0.8)
            
             #将自车轨迹每个轨迹点 以圆点的形式画出
            # for i in range(len(valid_points)):
            #     ax.plot(valid_points[i][0], valid_points[i][1], 'o', color=colors[-1] if color is None else color, zorder=5, markersize=0.2, markerfacecolor=colors[-1] if color is None else color, markeredgecolor=colors[-1] if color is None else color)
            
            # ax.add_patch(arrow)
        # 添加控制点相关函数
    def fit_trajectory_to_control_points(trajectory):
        """将轨迹转换为控制点并生成拟合曲线"""
        # 过滤掉无效点
        valid_points = trajectory[trajectory[:, 0] != 0]
        if len(valid_points) < 2:  # 确保有足够的点
            return None, None
            
        # 应用卡尔曼滤波
        filtered_trajectory = apply_kalman_filter(valid_points)
        
        # 拟合控制点
        degree = 5  # 控制点数量
        control_points = fit_bernstein_curve(filtered_trajectory, degree)
        
        # 生成拟合曲线点
        t = np.linspace(0, 1, 100)
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

    # 在提取控制点的代码段添加条件判断
    if model_cfg.get('unicp', False):
        # 提取控制点数据
        history_control_points = batch['history_control_points'][draw_index].cpu().numpy() 
        future_control_points = batch['future_control_points'][draw_index].cpu().numpy()
        # pred_control_points = prediction['predicted_control_point'][draw_index].detach().cpu().numpy()
    else:
        history_control_points = None
        future_control_points = None
        # pred_control_points = None


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
                
            # draw_trajectory(traj, line_width=0.5, color=None, ego=True)
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
                
            # 自车未来真值轨迹用黑色，使用点表示
            # draw_trajectory(traj, line_width=0.5, color=None, ego=True)
            
        else:
            # 其他车辆使用默认渐变色
            draw_trajectory(traj, line_width=2, color=None, ego=False)

        # 在绘制预测轨迹的部分添加控制点可视化

    # if pred_control_points is not None:
    #     ax.scatter(pred_control_points[:, 0], pred_control_points[:, 1],
    #                 color='orange', marker='*', s=20, zorder=5,
    #                 label='Predicted Control Points')
    #     ax.plot(pred_control_points[:, 0], pred_control_points[:, 1],
    #             color='orange', linewidth=0.2, zorder=6,
    #             label='Future Fitted Trajectory')
    
    # 找出概率最高的轨迹索引
    max_prob_idx = np.argmax(pred_future_prob)

    pred_traj = pred_future_traj[max_prob_idx]
    # 只绘制概率最高的轨迹
    # 使用黄色绘制预测轨迹
    # for i in range(len(pred_traj) - 1):
    #     if i == 0:  # 只在第一段添加图例
    #         draw_line_with_point(pred_traj[i], pred_traj[i + 1], 
    #                             color='yellow', line_width=0.5,
    #                             label='Predicted Trajectory')
    #     else:
    #         draw_line_with_point(pred_traj[i], pred_traj[i + 1], 
    #                         color='yellow', line_width=0.5,
    #                   )


    # #绘制所有概率预测轨迹
    # for idx, traj in enumerate(pred_future_traj):
    #     # 根据概率值指定颜色
    #     color = cm.hot(pred_future_prob[idx])
    #     for i in range(len(traj) - 1):
    #         draw_line_with_mask(traj[i], traj[i + 1], color=color, line_width=0.5)
    
            # 在右上角添加图例
    ax.legend(loc='upper right', 
             fontsize=8,
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
    # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # 确保目录存在
    os.makedirs(save_dir, exist_ok=True)
    # 生成文件名和完整的保存路径
    filename = f'trajectory_plot_{save_commit}.png'
    save_path = os.path.join(save_dir, filename)

    plt.savefig(save_path, dpi=500, bbox_inches='tight')
    # 清理图像以释放内存
    # plt.close()
    return plt,ego_history_traj,ego_future_traj,history_GT_control_points ,future_GT_control_points
