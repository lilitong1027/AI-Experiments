import sys
# 修复Windows控制台编码问题，放文件最开头
sys.stdout.reconfigure(encoding="utf-8")
import heapq

# ====================== 公共A*加权搜索函数 ======================
def a_star_weighted(start, goal, grid):
    """
    start: (r,c)
    goal: (r,c)
    grid：地图
    返回：最小代价；无路返回 None
    规则：
    X：障碍物，不可通行
    S/E：起点终点，进入时代价0
    . / A / B / T1~T4：代价1
    数字字符：int(字符)
    """
    rows = len(grid)
    cols = len(grid[0])
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def get_cost(r, c):
        ch = grid[r][c]
        if ch == 'X':
            return None
        if ch in ['S', 'E']:
            return 0
        if ch in ['A', 'B', 'T1', 'T2', 'T3', 'T4', '.']:
            return 1
        try:
            return int(ch)
        except ValueError:
            return None

    def manhattan(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    sr, sc = start
    gr, gc = goal
    if get_cost(sr, sc) is None or get_cost(gr, gc) is None:
        return None

    g_map = dict()
    open_heap = []
    g_map[start] = 0
    heapq.heappush(open_heap, (manhattan(start, goal), start))

    while open_heap:
        f_val, (r, c) = heapq.heappop(open_heap)
        if (r, c) == (gr, gc):
            return g_map[(r, c)]
        for dr, dc in dirs:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                cell_cost = get_cost(nr, nc)
                if cell_cost is None:
                    continue
                new_g = g_map[(r, c)] + cell_cost
                if (nr, nc) not in g_map or new_g < g_map[(nr, nc)]:
                    g_map[(nr, nc)] = new_g
                    f_new = new_g + manhattan((nr, nc), (gr, gc))
                    heapq.heappush(open_heap, (f_new, (nr, nc)))
    return None


# ====================== 测试1：无权迷宫A*寻路（带路径回溯） ======================
def a_star_unwrap_path(start, goal, grid):
    rows = len(grid)
    cols = len(grid[0])
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def manhattan(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    open_q = []
    heapq.heappush(open_q, (manhattan(start, goal), start))
    g = {start: 0}
    parent = {}
    while open_q:
        f, (r, c) = heapq.heappop(open_q)
        if (r, c) == goal:
            path = []
            cur = goal
            while cur is not None:
                path.append(cur)
                cur = parent.get(cur, None)
            return path[::-1]
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != 'X':
                newg = g[(r, c)] + 1
                if (nr, nc) not in g or newg < g[(nr, nc)]:
                    g[(nr, nc)] = newg
                    parent[(nr, nc)] = (r, c)
                    heapq.heappush(open_q, (newg + manhattan((nr, nc), goal), (nr, nc)))
    return []

def test1_maze():
    print("===== Test1: Maze A* =====")
    maze = [
        ['S', '.', '.', 'X', '.'],
        ['.', 'X', '.', '.', '.'],
        ['.', 'X', '.', 'X', '.'],
        ['.', '.', '.', 'X', 'E']
    ]
    s = (0, 0)
    e = (3, 4)
    path = a_star_unwrap_path(s, e, maze)
    print("Path:", path)
    if path:
        print("Total cost:", len(path) - 1)
    print()


# ====================== 测试2：加权地形A* ======================
def test2_weight_map():
    print("===== Test2: Weighted terrain A* =====")
    wgrid = [
        ['S', '1', '1', '8', '1'],
        ['1', '8', '3', '1', '1'],
        ['1', '8', '1', '8', '1'],
        ['1', '1', '1', '8', '1'],
        ['8', '8', '1', '1', 'E']
    ]
    start = (0, 0)
    end = (4, 4)
    cost = a_star_weighted(start, end, wgrid)
    print("Min total cost =", cost)
    print()


# ====================== 测试3：基础博弈树 Minimax / Alpha-Beta ======================
def alpha_beta(node, alpha, beta, is_max):
    if isinstance(node, int):
        return node
    if is_max:
        val = -float('inf')
        for child in node:
            val = max(val, alpha_beta(child, alpha, beta, False))
            alpha = max(alpha, val)
            if alpha >= beta:
                break
        return val
    else:
        val = float('inf')
        for child in node:
            val = min(val, alpha_beta(child, alpha, beta, True))
            beta = min(beta, val)
            if alpha >= beta:
                break
        return val

def test3_game_tree():
    print("===== Test3: Game Tree Alpha-Beta =====")
    max1 = [3, 5]
    max2 = [6, 9]
    max3 = [1, 2]
    min1 = [max1, max2]
    min2 = [max2, max3]
    root = [min1, min2]
    res = alpha_beta(root, -float('inf'), float('inf'), True)
    print("MAX best score =", res)
    print()


# ====================== 题目4：竞争环境下宝藏分配 ======================
print("===== Test4: Treasure Competition =====")
treasure_map = [
    ['A','1','1','X','2','1','T1'],
    ['1','1','X','X','1','X','1'],
    ['1','X','T2','1','1','1','1'],
    ['X','1','X','8','X','1','X'],
    ['1','1','1','T3','1','1','X'],
    ['1','X','1','X','1','1','1'],
    ['T4','1','X','1','X','1','B']
]
treasure_info = {
    "T1": {"pos":(0,6), "value":9},
    "T2": {"pos":(2,2), "value":6},
    "T3": {"pos":(4,4), "value":7},
    "T4": {"pos":(6,0), "value":7},
}
A_start = (0, 0)
B_start = (6, 6)

def treasure_alpha_beta(state_A_pos, state_B_pos, remain_treasures:tuple, scoreA, scoreB, depth, H, alpha, beta, is_max):
    if depth >= H or len(remain_treasures) == 0:
        return scoreA - scoreB, None
    best_move = None
    if is_max:
        val = -float('inf')
        for t_name in remain_treasures:
            t_pos = treasure_info[t_name]["pos"]
            cost = a_star_weighted(state_A_pos, t_pos, treasure_map)
            # 该宝藏不可到达，直接跳过
            if cost is None:
                continue
            gain = treasure_info[t_name]["value"] - cost
            newA = scoreA + gain
            new_remain = tuple(x for x in remain_treasures if x != t_name)
            res, _ = treasure_alpha_beta(t_pos, state_B_pos, new_remain, newA, scoreB, depth+1, H, alpha, beta, False)
            if res > val:
                val = res
                best_move = t_name
            alpha = max(alpha, val)
            if alpha >= beta:
                break
        return val, best_move
    else:
        val = float('inf')
        for t_name in remain_treasures:
            t_pos = treasure_info[t_name]["pos"]
            cost = a_star_weighted(state_B_pos, t_pos, treasure_map)
            # 该宝藏不可到达，直接跳过
            if cost is None:
                continue
            gain = treasure_info[t_name]["value"] - cost
            newB = scoreB + gain
            new_remain = tuple(x for x in remain_treasures if x != t_name)
            res, _ = treasure_alpha_beta(state_A_pos, t_pos, new_remain, scoreA, newB, depth+1, H, alpha, beta, True)
            if res < val:
                val = res
                best_move = t_name
            beta = min(beta, val)
            if alpha >= beta:
                break
        return val, best_move

def run_treasure_exp():
    remain = ("T1","T2","T3","T4")
    val_h2, best_h2 = treasure_alpha_beta(A_start,B_start, remain, 0,0, depth=0, H=2, alpha=-float("inf"), beta=float("inf"), is_max=True)
    print(f"H=2: {{'best_target':'{best_h2}', 'best_value':{val_h2}}}")
    val_h4, best_h4 = treasure_alpha_beta(A_start,B_start, remain,0,0, depth=0, H=4, alpha=-float("inf"), beta=float("inf"), is_max=True)
    print(f"H=4: {{'best_target':'{best_h4}', 'best_value':{val_h4}}}")
    print()


# ====================== main入口 ======================
if __name__ == "__main__":
    test1_maze()
    test2_weight_map()
    test3_game_tree()
    run_treasure_exp()
