import numpy as np
import matplotlib.pyplot as plt
import heapq
from scipy.interpolate import splprep, splev
import matplotlib.animation as animation

def heuristic(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def astar(grid, start, goal):
    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    oheap = []
    heapq.heappush(oheap, (fscore[start], start))

    while oheap:
        current = heapq.heappop(oheap)[1]
        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            return data[::-1]

        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            tentative_g_score = gscore[current] + heuristic(current, neighbor)
            if 0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]:
                if grid[neighbor[0]][neighbor[1]] == 1:
                    continue
            else:
                continue

            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, float('inf')):
                continue

            if tentative_g_score < gscore.get(neighbor, float('inf')) or neighbor not in [i[1] for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))

    return False

def b_spline_path(points, n_points=100):
    points = np.array(points)
    tck, u = splprep([points[:,0], points[:,1]], s=1)
    u_new = np.linspace(u.min(), u.max(), n_points)
    x_new, y_new = splev(u_new, tck, der=0)
    return np.vstack((x_new, y_new)).T

def replan_path(grid, current_position, goal):
    correction_point = (goal[0], goal[1] -5)  # 修正點位置
    if 0 <= correction_point[1] < grid.shape[1] and grid[correction_point[0], correction_point[1]] == 0:
        path1 = astar(grid, current_position, correction_point)
        path2 = astar(grid, correction_point, goal)
        if path1 and path2:
            path2.pop(0)  # 移除重複的修正點
            path = path1 + path2
        else:
            path = False
    else:
        path = False

    if path:
        path = np.array(path)
        smooth_path = b_spline_path(path, n_points=100)
        return smooth_path
    else:
        return None

# 創建網格地圖並添加障礙物
grid = np.zeros((80, 80))
# grid[45, 20] = 1  # 障礙物
# grid[46, 30] = 1  # 障礙物
# grid[20, 10:25] = 1  # 障礙物
# grid[45:55, 25] = 1  # 障礙物

# 定義起點和終點
start = (10, 10)
goal = (46, 46)

# 初始路徑規劃
smooth_path = replan_path(grid, start, goal)
if smooth_path is None:
    print("No path found")
else:
    # 繪製路徑
    fig, ax = plt.subplots()
    ax.imshow(grid.T, origin='lower', cmap='Greys')
    line, = ax.plot([], [], 'b--')
    scat_start = ax.scatter(start[0], start[1], color='green', s=40)
    scat_goal = ax.scatter(goal[0], goal[1], color='red', s=40)
    scat_current = ax.scatter([], [], color='blue', s=10)

    def init():
        line.set_data([], [])
        scat_current.set_offsets(np.array([[], []]).T)
        return line, scat_current

    def update(num):
        if num < len(smooth_path):
            line.set_data(smooth_path[:num, 0], smooth_path[:num, 1])
            current_position = smooth_path[num]
            scat_current.set_offsets(current_position)
        return line, scat_current

    ani = animation.FuncAnimation(fig, update, frames=len(smooth_path), init_func=init, interval=60, blit=True)
    plt.show()
