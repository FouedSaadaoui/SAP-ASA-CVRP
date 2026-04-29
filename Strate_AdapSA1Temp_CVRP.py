# =============================================================================
# Stratified Acceptance-Probability-based Adaptive Simulated Annealing for
# Capacitated Vehicle Routing Problem (CVRPLIB Benchmark)
# Purpose: Enhanced Simulated Annealing with data-driven adaptive acceptance
# probability using online Weibull mixture modeling of cost increments.
# Code by: Foued Saâdaoui (2026). Designed for Google Colab execution.
# - Clarke-Wright initial solution (fixed index bug)
# - 2-opt intra-route, swap, relocate, Or-opt
# - Reheat mechanism & local search
# =============================================================================
# Adaptive SA with Weibull mixture for CVRP
# Single temperature parameter
# =============================================================================

import numpy as np
import random
import math
import matplotlib.pyplot as plt
from google.colab import files

# Upload file
uploaded = files.upload()
filepath = list(uploaded.keys())[0]
print("Loaded file:", filepath)

# -------------------------------
# Load CVRP instance
# -------------------------------
def load_cvrp_instance(filepath):
    coords, demands = {}, {}
    capacity, depot, section = None, None, None
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if "CAPACITY" in line:
                capacity = int(line.split(":")[1])
            elif "NODE_COORD_SECTION" in line:
                section = "coords"; continue
            elif "DEMAND_SECTION" in line:
                section = "demands"; continue
            elif "DEPOT_SECTION" in line:
                section = "depot"; continue
            elif "EOF" in line:
                break
            elif section == "coords":
                parts = line.split()
                if len(parts) == 3:
                    i, x, y = parts
                    coords[int(i)-1] = (float(x), float(y))
            elif section == "demands":
                i, d = line.split()
                demands[int(i)-1] = int(d)
            elif section == "depot":
                if line != "-1":
                    depot = int(line)-1
    n = len(coords)
    coord_array = np.zeros((n,2))
    demand_array = np.zeros(n, dtype=int)
    for i in range(n):
        coord_array[i] = coords[i]
        demand_array[i] = demands[i]
    return coord_array, demand_array, capacity, depot

def reorder_depot_first(coords, demands, depot):
    if depot == 0: return coords, demands
    idx = list(range(len(coords)))
    idx.remove(depot)
    new_order = [depot] + idx
    return coords[new_order], demands[new_order]

def distance_matrix(coords):
    n = len(coords)
    dist = np.zeros((n,n), dtype=int)
    for i in range(n):
        for j in range(n):
            dist[i][j] = int(round(np.linalg.norm(coords[i]-coords[j])))
    return dist

# -------------------------------
# Clarke-Wright Savings Initial Solution
# -------------------------------
def clarke_wright_initial(demands, capacity, dist):
    n = len(demands)
    customers = list(range(1, n))
    savings = []
    for i in range(1, n):
        for j in range(i+1, n):
            save = dist[0][i] + dist[0][j] - dist[i][j]
            savings.append((save, i, j))
    savings.sort(reverse=True)

    routes = []
    route_index = {}
    first = {}
    last = {}
    load = {}

    for c in customers:
        idx = len(routes)
        routes.append([c])
        route_index[c] = idx
        first[idx] = c
        last[idx] = c
        load[idx] = demands[c]

    def merge(i, j):
        ri = route_index[i]
        rj = route_index[j]
        if ri == rj: return False
        if load[ri] + load[rj] > capacity: return False
        routes[ri] = routes[ri] + routes[rj]
        load[ri] += load[rj]
        last[ri] = last[rj]
        for c in routes[rj]:
            route_index[c] = ri
        routes[rj] = []
        load[rj] = 0
        return True

    for _, i, j in savings:
        if last[route_index[i]] == i and first[route_index[j]] == j:
            merge(i, j)
        elif last[route_index[j]] == j and first[route_index[i]] == i:
            merge(j, i)

    final_routes = [r for r in routes if r]
    return final_routes

# -------------------------------
# Cost functions
# -------------------------------
def route_cost(route, dist):
    if not route: return 0
    cost = dist[0][route[0]]
    for k in range(len(route)-1):
        cost += dist[route[k]][route[k+1]]
    cost += dist[route[-1]][0]
    return cost

def total_cost(routes, dist):
    return sum(route_cost(r, dist) for r in routes)

# -------------------------------
# Intra-route 2-opt
# -------------------------------
def two_opt(route, dist):
    improved = True
    best = route[:]
    best_cost = route_cost(route, dist)
    while improved:
        improved = False
        for i in range(1, len(best)-1):
            for j in range(i+1, len(best)):
                if j - i == 1: continue
                new_route = best[:i] + best[i:j+1][::-1] + best[j+1:]
                new_cost = route_cost(new_route, dist)
                if new_cost < best_cost:
                    best = new_route
                    best_cost = new_cost
                    improved = True
        if not improved:
            break
    return best

def local_search(routes, dist):
    return [two_opt(r, dist) for r in routes]

# -------------------------------
# Neighbourhood moves
# -------------------------------
def swap_move(routes, demands, capacity):
    if len(routes) < 2: return routes
    new = [r[:] for r in routes]
    r1, r2 = random.sample(range(len(new)), 2)
    if not new[r1] or not new[r2]: return routes
    i = random.randint(0, len(new[r1])-1)
    j = random.randint(0, len(new[r2])-1)
    c1, c2 = new[r1][i], new[r2][j]
    load1 = sum(demands[c] for c in new[r1])
    load2 = sum(demands[c] for c in new[r2])
    if load1 - demands[c1] + demands[c2] <= capacity and load2 - demands[c2] + demands[c1] <= capacity:
        new[r1][i], new[r2][j] = c2, c1
        return new
    return routes

def relocate_move(routes, demands, capacity):
    if len(routes) < 2: return routes
    new = [r[:] for r in routes]
    r1, r2 = random.sample(range(len(new)), 2)
    if not new[r1]: return routes
    i = random.randint(0, len(new[r1])-1)
    c = new[r1].pop(i)
    load2 = sum(demands[cx] for cx in new[r2])
    if load2 + demands[c] <= capacity:
        pos = random.randint(0, len(new[r2]))
        new[r2].insert(pos, c)
        new = [r for r in new if r]
        return new
    else:
        new[r1].insert(i, c)
        return routes

def or_opt_move(routes, demands, capacity):
    """Relocate a segment of up to 3 consecutive customers"""
    if len(routes) < 2: return routes
    new = [r[:] for r in routes]
    r1, r2 = random.sample(range(len(new)), 2)
    if len(new[r1]) < 2: return routes
    length = random.randint(1, min(3, len(new[r1])))
    start = random.randint(0, len(new[r1]) - length)
    seg = new[r1][start:start+length]
    new[r1] = new[r1][:start] + new[r1][start+length:]
    seg_load = sum(demands[c] for c in seg)
    load2 = sum(demands[c] for c in new[r2])
    if load2 + seg_load <= capacity:
        pos = random.randint(0, len(new[r2]))
        new[r2][pos:pos] = seg
        new = [r for r in new if r]
        return new
    else:
        new[r1][start:start] = seg
        return routes

# -------------------------------
# Fallback initial solution
# -------------------------------
def initial_solution_greedy(demands, capacity):
    customers = list(range(1, len(demands)))
    random.shuffle(customers)
    routes, route, load = [], [], 0
    for c in customers:
        if load + demands[c] <= capacity:
            route.append(c)
            load += demands[c]
        else:
            routes.append(route)
            route = [c]
            load = demands[c]
    if route:
        routes.append(route)
    return [r for r in routes if r]

# -------------------------------
# Enhanced Weibull estimators (rolling window, capping)
# -------------------------------
def estimate_weibull_from_window(window):
    """Estimate Weibull k, lam using method of moments with capped deltas."""
    if len(window) < 10:
        return 1.0, 1.0
    # Cap deltas at 3 * mean to reduce outlier influence
    capped = np.array(window)
    mean_w = np.mean(capped)
    capped = np.minimum(capped, 3 * mean_w)
    mean_capped = np.mean(capped)
    var_capped = np.var(capped)
    if mean_capped < 1e-9 or var_capped < 1e-9:
        return 1.0, 1.0
    cv = math.sqrt(var_capped) / mean_capped
    # Heuristic: k = cv^{-1.086} (common approximation)
    if cv < 0.1:
        k = 10.0
    elif cv > 2.0:
        k = 0.5
    else:
        k = cv ** (-1.086)
    k = min(max(k, 0.3), 5.0)
    lam = mean_capped / math.gamma(1 + 1/k)
    lam = max(lam, 1e-6)
    return k, lam

# -------------------------------
# Hybrid SA with enhanced Weibull
# -------------------------------
def hybrid_sa(dist, demands, capacity,
              T0=5000, alpha=0.999, Tmin=1e-6,
              max_iter=100000, rho=0.02,
              window_size=500, min_pos_deltas=200,
              update_freq=100, min_prob=0.001,
              reheats=3, seed=42):
    random.seed(seed); np.random.seed(seed)
    
    # Initial solution
    routes = clarke_wright_initial(demands, capacity, dist)
    if not routes or any(not r for r in routes):
        print("Clarke-Wright failed, using greedy fallback")
        routes = initial_solution_greedy(demands, capacity)
    routes = local_search(routes, dist)
    current_cost = total_cost(routes, dist)
    best_routes = [r[:] for r in routes]
    best_cost = current_cost
    
    T = T0
    # Rolling window for positive deltas
    delta_window = []
    k, lam = 1.0, 1.0
    weibull_active = False   # becomes True after enough positive deltas
    last_update = 0
    no_improve = 0
    reheat_count = 0
    
    for it in range(max_iter):
        # Choose move
        r = random.random()
        if r < 0.4:
            cand = swap_move(routes, demands, capacity)
        elif r < 0.7:
            cand = relocate_move(routes, demands, capacity)
        else:
            cand = or_opt_move(routes, demands, capacity)
        
        if cand is routes:
            continue
        
        cand = local_search(cand, dist)
        cand_cost = total_cost(cand, dist)
        delta = cand_cost - current_cost
        
        # Collect positive deltas (capped for storage)
        if delta > 0:
            # Cap at 5*mean to avoid extreme values (but we'll cap again in estimation)
            delta_window.append(delta)
            if len(delta_window) > window_size:
                delta_window.pop(0)
        
        # Determine acceptance probability
        if delta <= 0:
            prob = 1.0
        else:
            # Activate Weibull only after enough data
            if len(delta_window) >= min_pos_deltas:
                weibull_active = True
                # Periodic re-estimation
                if (it - last_update) >= update_freq:
                    k, lam = estimate_weibull_from_window(delta_window)
                    last_update = it
                # Proper T scaling: (delta / (lam * T))**k
                exponent = (delta / (lam * T)) ** k
                prob = math.exp(-exponent)
                prob = max(prob, min_prob)  # minimum acceptance
            else:
                # Classical SA fallback while collecting data
                prob = math.exp(-delta / T) if T > 0 else 0.0
        
        # Accept/reject
        if random.random() < prob:
            routes = cand
            current_cost = cand_cost
            if current_cost < best_cost:
                best_routes = [r[:] for r in routes]
                best_cost = current_cost
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
        
        # Reheat if stuck
        if no_improve > 10000 and reheat_count < reheats:
            T = T0 * (0.5 ** (reheat_count+1))
            no_improve = 0
            reheat_count += 1
            print(f"Reheating to T={T:.2f} (reheat {reheat_count})")
            # Reset Weibull data on reheat
            delta_window = []
            weibull_active = False
            last_update = 0
            k, lam = 1.0, 1.0
            continue
        
        T *= alpha
        if T < Tmin:
            break
        
        if it % 5000 == 0:
            phase = "Classical" if not weibull_active else "Weibull"
            print(f"Iter {it:7d} | Best: {best_cost:8.2f} | T: {T:6.2f} | Phase: {phase} | Window: {len(delta_window)}")
    
    # Final local search
    best_routes = local_search(best_routes, dist)
    best_cost = total_cost(best_routes, dist)
    return best_routes, best_cost

# -------------------------------
# Plot solution
# -------------------------------
def plot_solution(coords, routes):
    plt.figure(figsize=(10,8))
    for route in routes:
        path = [0] + route + [0]
        xs = [coords[i][0] for i in path]
        ys = [coords[i][1] for i in path]
        plt.plot(xs, ys, marker='o', linewidth=1, markersize=5)
    plt.scatter(coords[0][0], coords[0][1], s=200, c='red', marker='s', label='Depot')
    plt.title("Hybrid SA with Enhanced Weibull Adaptive Acceptance")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# -------------------------------
# Run
# -------------------------------
coords, demands, capacity, depot = load_cvrp_instance(filepath)
coords, demands = reorder_depot_first(coords, demands, depot)
print(f"Customers: {len(coords)-1}, Capacity: {capacity}")
dist = distance_matrix(coords)

best_routes, best_cost = hybrid_sa(dist, demands, capacity,
                                   T0=5000, alpha=0.999, max_iter=100000,
                                   reheats=3)

print("\n" + "="*60)
print(f"BEST COST: {best_cost:.2f}")
print(f"Number of routes: {len(best_routes)}")
for i, route in enumerate(best_routes):
    load = sum(demands[c] for c in route)
    print(f"Route {i+1}: {route} (load: {load})")

plot_solution(coords, best_routes)