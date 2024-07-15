import numpy as np
import matplotlib.pyplot as plt
import heapq
from scipy.interpolate import splprep, splev

#A*算法中的啟發函數，計算兩點 a 和 b 之間的歐幾里得距離（直線距離）
def heuristic(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))
#實現A*算法，用於尋找在grid上start到goal的最短路徑
def astar(grid, start, goal):
    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)] # 定義八個可能的移動方向，包括四個直線和四個對角線。
    close_set = set() # 用於記錄已經評估過的節點。
    came_from = {} # 用於追踪每個節點是從哪個節點來的。
    # 記錄每個節點從起點到該節點的實際成本以及預計成本。
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    oheap = []  # 存放待評估的節點，並按照優先級排序。
    # 把新的元素push進oheap，(排序,內容)
    heapq.heappush(oheap, (fscore[start], start))

    while oheap:
        current = heapq.heappop(oheap)[1]
        # 如果當前節點是目標節點，則構造最短路徑並返回。
        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            return data[::-1]

        close_set.add(current)
        # 遍歷當前節點的所有相鄰節點，計算每個相鄰節點的臨時 g_score。
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            tentative_g_score = gscore[current] + heuristic(current, neighbor)
            # 確保相鄰節點在網格範圍內，並且不是障礙物（即網格值不為1）。
            if 0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]:
                if grid[neighbor[0]][neighbor[1]] == 1:
                    continue
            else:
                continue
            # 如果相鄰節點已在封閉集合中，且新的 g_score 不優於之前的，則忽略該節點。
            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, float('inf')):
                continue
            # 如果找到更優的路徑，則更新 came_from，gscore，fscore，並將相鄰節點加入到開放列表中。
            if tentative_g_score < gscore.get(neighbor, float('inf')) or neighbor not in [i[1] for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))

    return False

# B-spline for path smoothing
def b_spline_path(points, n_points=100):
    points = np.array(points)
    tck, u = splprep([points[:,0], points[:,1]], s=4)
    u_new = np.linspace(u.min(), u.max(), n_points)
    x_new, y_new = splev(u_new, tck, der=0)
    return np.vstack((x_new, y_new)).T

# Example grid map with obstacles, 0表示可通行區域，1表示障礙物
grid = np.zeros((50, 50))

# Adding obstacles
# grid[15:35, 25] = 1  # 垂直障礙物
# grid[45, 20] = 1   # 水平障礙物

# Define start and goal points
start = (5, 5)
goal = (45, 35)

# Check if goal's virtual extension line is vertical to start point
if start[0] == goal[0] or start[1] == goal[1]:
    # Direct A* path without turning point
    path = astar(grid, start, goal)
else:
    # Add virtual extension lines
    virtual_x = goal[0]
    virtual_y = start[1]
    turning_point = (virtual_x, virtual_y)

    # Adjust path to include the turning point
    path1 = astar(grid, start, turning_point)
    path2 = astar(grid, turning_point, goal)  # Remove the duplicate turning point
    path = path1 + path2

# Adding B-spline smoothing
smooth_path = b_spline_path(path, n_points=300)

# Plotting the path
plt.imshow(grid.T, origin='lower', cmap='Greys')
# 顯示虛擬線（如有）
# if start[0] != goal[0] and start[1] != goal[1]:
#     plt.plot([start[0], turning_point[0]], [start[1], turning_point[1]], 'r--')
#     plt.plot([turning_point[0], goal[0]], [turning_point[1], goal[1]], 'r--')
# 不顯示虛擬線，只顯示平滑後的最終路徑
plt.plot(smooth_path[:, 0], smooth_path[:, 1], 'b--')
# 繪製起點和終點
plt.scatter(start[0], start[1], color='green', s=100)
plt.scatter(goal[0], goal[1], color='red', s=100)
plt.show()

