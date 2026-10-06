import random

# ==========================================
# 1. Same Network Topology (Adjacency Matrix)
# ==========================================
INF = float('inf')

# Graph representing link costs/latencies between Nodes 0 to 5
network_graph = [
    [0, 4, 2, INF, INF, INF], # Node 0
    [4, 0, 1, 5, INF, INF], # Node 1
    [2, 1, 0, 8, 10, INF], # Node 2
    [INF, 5, 8, 0, 2, 6], # Node 3
    [INF, INF, 10, 2, 0, 3], # Node 4
    [INF, INF, INF, 6, 3, 0] # Node 5
]

NUM_NODES = len(network_graph)
SOURCE_NODE = 0
DEST_NODE = 5


# ==========================================
# 2. Path Generation & Cost Helpers
# ==========================================

def generate_random_path(src, dest):
    """Generates a valid path from src to dest using random walk."""
    path = [src]
    current = src
    visited = {src}
    
    while current != dest:
        neighbors = [
            i for i in range(NUM_NODES) 
            if network_graph[current][i] != INF and i not in visited
        ]
        if not neighbors:
            return generate_random_path(src, dest)
            
        current = random.choice(neighbors)
        path.append(current)
        visited.add(current)
        
    return path


def calculate_path_cost(path):
    """Calculates cumulative latency/cost along the route."""
    cost = 0
    for i in range(len(path) - 1):
        c = network_graph[path[i]][path[i+1]]
        if c == INF:
            return float('inf')
        cost += c
    return cost


def calculate_fitness(path):
    """Inverse of path cost."""
    cost = calculate_path_cost(path)
    return 1.0 / cost if cost > 0 else 0


# ==========================================
# 3. Discrete Velocity & Position Operators
# ==========================================

def splice_paths(target_path, guide_path):
    """
    Cognitive/Social Operator: Modifies 'target_path' by splicing sub-routes 
    from 'guide_path' at a shared intermediate node.
    """
    common_nodes = list(set(target_path[1:-1]).intersection(set(guide_path[1:-1])))
    if not common_nodes:
        return target_path[:]
    
    pivot = random.choice(common_nodes)
    idx_target = target_path.index(pivot)
    idx_guide = guide_path.index(pivot)
    
    new_path = target_path[:idx_target] + guide_path[idx_guide:]
    
    # Check for loop-free path validity
    if len(new_path) == len(set(new_path)) and calculate_path_cost(new_path) != INF:
        return new_path
    return target_path[:]


def perturb_path(path):
    """Inertia Operator: Reroutes a sub-path from a random intermediate node."""
    if len(path) <= 2:
        return path[:]
        
    mutation_idx = random.randint(1, len(path) - 2)
    mutation_node = path[mutation_idx]
    
    new_subpath = generate_random_path(mutation_node, DEST_NODE)
    mutated_path = path[:mutation_idx] + new_subpath
    
    if len(mutated_path) == len(set(mutated_path)) and calculate_path_cost(mutated_path) != INF:
        return mutated_path
    return path[:]


# ==========================================
# 4. PSO Main Loop
# ==========================================

class Particle:
    def __init__(self, src, dest):
        self.position = generate_random_path(src, dest) # Current route
        self.pbest_position = self.position[:] # Personal best route
        self.pbest_cost = calculate_path_cost(self.position)


def run_pso_routing(num_particles=20, max_iterations=30, w=0.3, c1=0.4, c2=0.5):
    # Initialize swarm
    swarm = [Particle(SOURCE_NODE, DEST_NODE) for _ in range(num_particles)]
    
    # Initialize Global Best
    gbest_particle = min(swarm, key=lambda p: p.pbest_cost)
    gbest_position = gbest_particle.pbest_position[:]
    gbest_cost = gbest_particle.pbest_cost
    
    print(f"Running Discrete PSO for Route Optimization (Node {SOURCE_NODE} -> Node {DEST_NODE})...\n")

    for it in range(1, max_iterations + 1):
        for particle in swarm:
            
            # --- Velocity / Position Update ---
            current_path = particle.position[:]
            
            # 1. Inertia Component (w): Random structural exploration
            if random.random() < w:
                current_path = perturb_path(current_path)
                
            # 2. Cognitive Component (c1): Move toward Personal Best (Pbest)
            if random.random() < c1:
                current_path = splice_paths(current_path, particle.pbest_position)
                
            # 3. Social Component (c2): Move toward Global Best (Gbest)
            if random.random() < c2:
                current_path = splice_paths(current_path, gbest_position)
            
            # Update current position
            particle.position = current_path
            current_cost = calculate_path_cost(particle.position)
            
            # --- Update Personal Best (Pbest) ---
            if current_cost < particle.pbest_cost:
                particle.pbest_cost = current_cost
                particle.pbest_position = particle.position[:]
                
                # --- Update Global Best (Gbest) ---
                if current_cost < gbest_cost:
                    gbest_cost = current_cost
                    gbest_position = particle.position[:]
                    
        if it == 1 or it % 10 == 0:
            print(f"Iteration {it:02d} | Best Cost: {gbest_cost} | Best Route: {gbest_position}")

    return gbest_position, gbest_cost


if __name__ == "__main__":
    optimal_route, optimal_cost = run_pso_routing()
    print("\n" + "="*50)
    print(f"PSO OPTIMAL PATH FOUND: {' -> '.join(map(str, optimal_route))}")
    print(f"TOTAL LATENCY/COST: {optimal_cost}")
    print("="*50)

