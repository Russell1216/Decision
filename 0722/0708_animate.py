import numpy as np
import matplotlib.pyplot as plt
import heapq
from scipy.interpolate import splprep, splev
import matplotlib.animation as animation
from PIL import Image  # 新增這行

def heuristic(a, b):
    """計算從點a到點b的歐幾里得距離"""
    return np.linalg.norm(np.array(a) - np.array(b))

def astar_bidirectional(grid, start, goal):
    """雙向A*路徑搜尋算法"""

    def reconstruct_path(came_from_start, came_from_goal, meeting_node):
        """重建從起點到終點的路徑"""
        path_start = []
        current = meeting_node
        while current in came_from_start:
            path_start.append(current)
            current = came_from_start[current]
        path_start = path_start[::-1]  # 起點到 meeting_node 的路徑

        path_goal = []
        current = meeting_node
        while current in came_from_goal:
            path_goal.append(current)
            current = came_from_goal[current]

        return path_start + path_goal[1:]  # 連接兩部分路徑

    # 定義8個鄰居方向
    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

    # 初始化起點搜索數據結構
    close_set_start = set()  # 已檢查的節點集合
    came_from_start = {}  # 起點方向的路徑記錄
    gscore_start = {start: 0}  # 起點方向的 g 值（起點到當前節點的實際代價）
    fscore_start = {start: heuristic(start, goal)}  # 起點方向的 f 值（g 值 + 預估值）
    oheap_start = []  # 起點方向的開放集
    heapq.heappush(oheap_start, (fscore_start[start], start))  # 將起點放入開放集

    # 初始化終點搜索數據結構
    close_set_goal = set()  # 已檢查的節點集合
    came_from_goal = {}  # 終點方向的路徑記錄
    gscore_goal = {goal: 0}  # 終點方向的 g 值（終點到當前節點的實際代價）
    fscore_goal = {goal: heuristic(goal, start)}  # 終點方向的 f 值（g 值 + 預估值）
    oheap_goal = []  # 終點方向的開放集
    heapq.heappush(oheap_goal, (fscore_goal[goal], goal))  # 將終點放入開放集

    while oheap_start and oheap_goal:
        # 起點方向搜索
        current_start = heapq.heappop(oheap_start)[1]  # 取出f值最小的節點
        if current_start in close_set_goal:  # 如果當前節點在終點方向的已檢查集合中，則找到路徑
            return reconstruct_path(came_from_start, came_from_goal, current_start)

        close_set_start.add(current_start)  # 將當前節點加入已檢查集合
        for i, j in neighbors:  # 遍歷所有鄰居節點
            neighbor = current_start[0] + i, current_start[1] + j
            tentative_g_score = gscore_start[current_start] + heuristic(current_start, neighbor)
            if 0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]:  # 確保鄰居節點在網格範圍內
                if grid[neighbor[0]][neighbor[1]] == 1:  # 遇到障礙物
                    continue
            else:
                continue

            if neighbor in close_set_start and tentative_g_score >= gscore_start.get(neighbor, float('inf')):
                continue

            if tentative_g_score < gscore_start.get(neighbor, float('inf')) or neighbor not in [i[1] for i in oheap_start]:
                came_from_start[neighbor] = current_start  # 記錄路徑
                gscore_start[neighbor] = tentative_g_score
                fscore_start[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(oheap_start, (fscore_start[neighbor], neighbor))

        # 終點方向搜索
        current_goal = heapq.heappop(oheap_goal)[1]  # 取出f值最小的節點
        if current_goal in close_set_start:  # 如果當前節點在起點方向的已檢查集合中，則找到路徑
            return reconstruct_path(came_from_start, came_from_goal, current_goal)

        close_set_goal.add(current_goal)  # 將當前節點加入已檢查集合
        for i, j in neighbors:  # 遍歷所有鄰居節點
            neighbor = current_goal[0] + i, current_goal[1] + j
            tentative_g_score = gscore_goal[current_goal] + heuristic(current_goal, neighbor)
            if 0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]:  # 確保鄰居節點在網格範圍內
                if grid[neighbor[0]][neighbor[1]] == 1:  # 遇到障礙物
                    continue
            else:
                continue

            if neighbor in close_set_goal and tentative_g_score >= gscore_goal.get(neighbor, float('inf')):
                continue

            if tentative_g_score < gscore_goal.get(neighbor, float('inf')) or neighbor not in [i[1] for i in oheap_goal]:
                came_from_goal[neighbor] = current_goal  # 記錄路徑
                gscore_goal[neighbor] = tentative_g_score
                fscore_goal[neighbor] = tentative_g_score + heuristic(neighbor, start)
                heapq.heappush(oheap_goal, (fscore_goal[neighbor], neighbor))

    return False  # 無法找到路徑

def b_spline_path(points, n_points=100):
    """生成B樣條曲線路徑"""
    points = np.array(points)
    tck, u = splprep([points[:,0], points[:,1]], s=3)  # 生成樣條曲線參數
    u_new = np.linspace(u.min(), u.max(), n_points)  # 生成新的樣條曲線參數
    x_new, y_new = splev(u_new, tck, der=0)  # 計算樣條曲線上的點
    return np.vstack((x_new, y_new)).T  # 返回平滑路徑

def calculate_correction_point(goal, turning_point, distance=5):
    """計算修正點"""
    direction = np.array(turning_point) - np.array(goal)
    direction = direction / np.linalg.norm(direction)  # 單位方向向量
    correction_point = np.array(goal) + direction * distance
    return tuple(np.round(correction_point).astype(int))  # 返回整數座標

def replan_path(grid, current_position, goal):
    """重新規劃路徑，計算轉折點和修正點"""
    virtual_x = goal[0]
    virtual_y = current_position[1]
    turning_point = (virtual_x, virtual_y)
    correction_point = calculate_correction_point(goal, turning_point, distance=30)

    # 確保修正點在網格範圍內並且不是障礙物
    if 0 <= correction_point[1] < grid.shape[1] and grid[correction_point[0], correction_point[1]] == 0:
        path1 = astar_bidirectional(grid, current_position, correction_point)  # 使用雙向A*計算到修正點的路徑
        path2 = astar_bidirectional(grid, correction_point, goal)  # 使用雙向A*計算修正點到目標點的路徑
        if path1 and path2:
            path2.pop(0)  # 移除重複的修正點
            path = path1 + path2  # 合併路徑
        else:
            path = False
    else:
        path = False

    if path:
        path = np.array(path)
        smooth_path = b_spline_path(path, n_points=100)  # 平滑路徑
        return smooth_path
    else:
        return None

# 創建網格地圖並添加障礙物
grid = np.zeros((80, 80))
# grid[45, 20] = 1  # 障礙物
# grid[46, 30] = 1  # 障礙物
# grid[20, 10:35] = 1  # 障礙物
# grid[45:55, 25] = 1  # 障礙物

# 定義起點和終點
start = (10, 10)  # 起點
goal = (46, 46)  # 終點

# 初始路徑規劃
smooth_path = replan_path(grid, start, goal)
if smooth_path is None:
    print("No path found")
else:
    # 繪製路徑
    fig, ax = plt.subplots()
    ax.imshow(grid.T, origin='lower', cmap='Greys')  # 顯示網格
    line, = ax.plot([], [], 'b--')  # 初始化路徑線
    scat_start = ax.scatter(start[0], start[1], color='green', s=40)  # 起點標記
    scat_goal = ax.scatter(goal[0], goal[1], color='red', s=40)  # 終點標記
    scat_current = ax.scatter([], [], color='blue', s=10)  # 當前位置標記

    # 添加起點和終點的標籤
    ax.text(start[0], start[1], 'start', fontsize=12, ha='right', color='green')
    ax.text(goal[0], goal[1], 'goal', fontsize=12, ha='right', color='red')

    def init():
        line.set_data([], [])  # 初始化路徑數據
        scat_current.set_offsets(np.array([[], []]).T)  # 初始化當前位置數據
        return line, scat_current

    def update(num):
        if num < len(smooth_path):
            line.set_data(smooth_path[:num, 0], smooth_path[:num, 1])  # 更新路徑數據
            current_position = smooth_path[num]  # 當前位置
            scat_current.set_offsets(current_position)  # 更新當前位置數據
        return line, scat_current

    ani = animation.FuncAnimation(fig, update, frames=len(smooth_path), init_func=init, interval=100, blit=True)  # 動畫設置

    ani.save('path_animation.gif', writer='pillow', fps=10)  # 保存動畫為GIF檔案
    plt.show()  # 顯示圖形
