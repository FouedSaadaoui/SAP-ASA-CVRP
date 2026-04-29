# =============================================================================
# Stratified Acceptance-Probability-based Adaptive Simulated Annealing
# for Capacitated Vehicle Routing Problem (CVRPLIB Benchmark), 
# Purpose: Enhanced Simulated Annealing with data-driven adaptive acceptance
# probability using online exponential modeling of cost increments.
# Code by: Foued Saâdaoui (2026). Designed for Google Colab execution.
# =============================================================================

# ================================
# 1. IMPORTS
# ================================
import numpy as np
import random
import math
import matplotlib.pyplot as plt


# ================================
# 2. UPLOAD FILE (COLAB)
# ================================
from google.colab import files

uploaded = files.upload()
filepath = list(uploaded.keys())[0]

print("Loaded file:", filepath)


# ================================
# 3. LOAD CVRPLIB INSTANCE
# ================================
def load_cvrp_instance(filepath):
    coords = {}
    demands = {}
    capacity = None
    depot = None
    section = None

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if "CAPACITY" in line:
                capacity = int(line.split(":")[1])

            elif "NODE_COORD_SECTION" in line:
                section = "coords"
                continue

            elif "DEMAND_SECTION" in line:
                section = "demands"
                continue

            elif "DEPOT_SECTION" in line:
                section = "depot"
                continue

            elif "EOF" in line:
                break

            elif section == "coords":
                i, x, y = line.split()
                coords[int(i) - 1] = (float(x), float(y))

            elif section == "demands":
                i, d = line.split()
                demands[int(i) - 1] = int(d)

            elif section == "depot":
                if line != "-1":
                    depot = int(line) - 1

    n = len(coords)
    coord_array = np.zeros((n, 2))
    demand_array = np.zeros(n, dtype=int)

    for i in range(n):
        coord_array[i] = coords[i]
        demand_array[i] = demands[i]

    return coord_array, demand_array, capacity, depot


# ================================
# 4. REORDER DEPOT FIRST
# ================================
def reorder_depot_first(coords, demands, depot):
    if depot == 0:
        return coords, demands

    idx = list(range(len(coords)))
    idx.remove(depot)
    new_order = [depot] + idx

    return coords[new_order], demands[new_order]


# ================================
# 5. DISTANCE MATRIX (EUC_2D)
# ================================
def distance_matrix(coords):
    n = len(coords)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i][j] = round(np.linalg.norm(coords[i] - coords[j]))
    return dist


# ================================
# 6. INITIAL SOLUTION
# ================================
def initial_solution(demands, capacity):
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

    return routes


# ================================
# 7. COST FUNCTION
# ================================
def route_cost(route, dist):
    cost = 0
    prev = 0
    for c in route:
        cost += dist[prev][c]
        prev = c
    cost += dist[prev][0]
    return cost


def total_cost(routes, dist):
    return sum(route_cost(r, dist) for r in routes)


# ================================
# 8. NEIGHBORHOODS
# ================================
def swap_move(routes):
    new_routes = [r[:] for r in routes]
    r1, r2 = random.sample(range(len(new_routes)), 2)

    if new_routes[r1] and new_routes[r2]:
        i = random.randint(0, len(new_routes[r1]) - 1)
        j = random.randint(0, len(new_routes[r2]) - 1)
        new_routes[r1][i], new_routes[r2][j] = new_routes[r2][j], new_routes[r1][i]

    return new_routes


def relocate_move(routes, demands, capacity):
    new_routes = [r[:] for r in routes]
    r1, r2 = random.sample(range(len(new_routes)), 2)

    if not new_routes[r1]:
        return new_routes

    i = random.randint(0, len(new_routes[r1]) - 1)
    c = new_routes[r1].pop(i)

    if sum(demands[x] for x in new_routes[r2]) + demands[c] <= capacity:
        new_routes[r2].append(c)
    else:
        new_routes[r1].append(c)

    return new_routes


# ================================
# 9. ADAPTIVE SA 
# ================================
def adaptive_sa(dist, demands, capacity,
                T0=1000, alpha_T=0.995,
                Tmin=1e-4, max_iter=10000,
                rho=0.05, alpha=0.7, t0=500):

    current = initial_solution(demands, capacity)
    current_cost = total_cost(current, dist)

    best = current
    best_cost = current_cost

    T = T0
    avg_delta = 1.0

    for it in range(max_iter):

        # neighbor
        if random.random() < 0.5:
            candidate = swap_move(current)
        else:
            candidate = relocate_move(current, demands, capacity)

        candidate_cost = total_cost(candidate, dist)
        delta = candidate_cost - current_cost

        # -----------------------------
        # acceptance rule
        # -----------------------------
        if it < t0:
            prob = math.exp(-delta / T) if delta > 0 else 1.0
        else:
            if delta > 0:
                avg_delta = (1 - rho) * avg_delta + rho * delta

            lambda_t = 1.0 / max(avg_delta, 1e-6)
            lambda_t = min(lambda_t, 1e3)

            beta_t = (lambda_t ** alpha) / T
            prob = math.exp(-beta_t * delta) if delta > 0 else 1.0

        # accept/reject
        if random.random() < prob:
            current = candidate
            current_cost = candidate_cost

            if current_cost < best_cost:
                best = current
                best_cost = current_cost

        # cooling
        T *= alpha_T
        if T < Tmin:
            break

        if it % 500 == 0:
            print(f"Iter {it} | Best: {best_cost:.2f} | T: {T:.4f}")

    return best, best_cost


# ================================
# 10. VISUALIZATION
# ================================
def plot_solution(coords, routes):
    plt.figure(figsize=(8, 6))

    for route in routes:
        path = [0] + route + [0]
        xs = [coords[i][0] for i in path]
        ys = [coords[i][1] for i in path]
        plt.plot(xs, ys, marker='o')

    plt.scatter(coords[0][0], coords[0][1], s=120, label="Depot")
    plt.title("Adaptive SA for CVRP (CVRPLIB)")
    plt.legend()
    plt.show()


# ================================
# 11. RUN
# ================================
coords, demands, capacity, depot = load_cvrp_instance(filepath)
coords, demands = reorder_depot_first(coords, demands, depot)

dist = distance_matrix(coords)

best_routes, best_cost = adaptive_sa(dist, demands, capacity)

print("\nBEST COST:", best_cost)
print("ROUTES:", best_routes)

plot_solution(coords, best_routes)