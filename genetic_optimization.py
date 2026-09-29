import random

# ==========================================
# 1. NETWORK TOPOLOGY (Adjacency Matrix)
# ==========================================
# INF means no direct link exists. Numbers represent latency/cost between routers.
INF = float('inf')

network = [
    # Node 0, 1,  2,  3,  4,  5
    [0, 4, 2, INF, INF, INF],  # Router 0 (SOURCE)
    [4, 0, 1, 5, INF, INF],    # Router 1
    [2, 1, 0, 8, 10, INF],     # Router 2
    [INF, 5, 8, 0, 2, 6],      # Router 3
    [INF, INF, 10, 2, 0, 3],   # Router 4
    [INF, INF, INF, 6, 3, 0]   # Router 5 (DESTINATION)
]

SOURCE = 0
DESTINATION = 5
POPULATION_SIZE = 10
GENERATIONS = 20
MUTATION_RATE = 0.2


# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def get_neighbors(node):
    """Finds all directly connected routers to a given node."""
    neighbors = []
    for neighbor_id, weight in enumerate(network[node]):
        if weight > 0 and weight != INF:
            neighbors.append(neighbor_id)
    return neighbors


def generate_random_path(src, dest):
    """Generates a valid path from source to destination using random walks."""
    path = [src]
    current = src
    visited = {src}

    while current != dest:
        neighbors = [n for n in get_neighbors(current) if n not in visited]
        if not neighbors:
            # Dead end reached, restart path discovery
            return generate_random_path(src, dest)
        
        next_node = random.choice(neighbors)
        path.append(next_node)
        visited.add(next_node)
        current = next_node

    return path


def calculate_cost(path):
    """Calculates total latency/cost along a given path."""
    total_cost = 0
    for i in range(len(path) - 1):
        from_node = path[i]
        to_node = path[i + 1]
        total_cost += network[from_node][to_node]
    return total_cost


def calculate_fitness(path):
    """Fitness is inverse of path cost. Shorter paths = Higher fitness."""
    cost = calculate_cost(path)
    return 1.0 / cost if cost > 0 else 0


# ==========================================
# 3. GENETIC OPERATORS
# ==========================================

def select_parent(population, fitnesses):
    """Selects a parent path using Roulette Wheel Selection."""
    total_fitness = sum(fitnesses)
    pick = random.uniform(0, total_fitness)
    current = 0
    
    for i, fitness in enumerate(fitnesses):
        current += fitness
        if current >= pick:
            return population[i]
    return population[-1]


def crossover(parent1, parent2):
    """Combines two paths at a shared middle node."""
    # Find common nodes excluding source and destination
    common_nodes = set(parent1[1:-1]).intersection(set(parent2[1:-1]))
    
    if not common_nodes:
        # If no middle node is shared, return parent copies
        return parent1[:], parent2[:]
    
    cut_node = random.choice(list(common_nodes))
    idx1 = parent1.index(cut_node)
    idx2 = parent2.index(cut_node)
    
    child1 = parent1[:idx1] + parent2[idx2:]
    child2 = parent2[:idx2] + parent1[idx1:]
    
    # Check for valid paths without loops/duplicates
    if len(child1) == len(set(child1)) and len(child2) == len(set(child2)):
        return child1, child2
    return parent1[:], parent2[:]


def mutate(path):
    """Randomly modifies a segment of the path if mutation occurs."""
    if random.random() > MUTATION_RATE or len(path) <= 3:
        return path
    
    # Pick a random intermediate node to break and rebuild
    split_index = random.randint(1, len(path) - 2)
    sub_path = path[:split_index]
    
    # Generate new suffix from split node to destination
    suffix = generate_random_path(path[split_index], DESTINATION)
    new_path = sub_path + suffix[1:]
    
    # Return new path only if it contains no repeating nodes (no loops)
    if len(new_path) == len(set(new_path)):
        return new_path
    return path


# ==========================================
# 4. MAIN GENETIC ALGORITHM EXECUTION
# ==========================================

def run_genetic_algorithm():
    # Step 1: Initialize population
    population = [generate_random_path(SOURCE, DESTINATION) for _ in range(POPULATION_SIZE)]

    print("=== INITIAL POPULATION ===")
    for p in population:
        print(f"Path: {p} | Cost: {calculate_cost(p)}")

    # Step 2: Evolve across generations
    for gen in range(1, GENERATIONS + 1):
        fitnesses = [calculate_fitness(p) for p in population]
        new_population = []

        # Elitism: Retain best path from previous generation
        best_idx = fitnesses.index(max(fitnesses))
        new_population.append(population[best_idx])

        # Fill remaining slots using Crossover & Mutation
        while len(new_population) < POPULATION_SIZE:
            p1 = select_parent(population, fitnesses)
            p2 = select_parent(population, fitnesses)
            c1, c2 = crossover(p1, p2)
            new_population.append(mutate(c1))
            if len(new_population) < POPULATION_SIZE:
                new_population.append(mutate(c2))

        population = new_population

    # Step 3: Find and display optimal path
    final_fitnesses = [calculate_fitness(p) for p in population]
    best_index = final_fitnesses.index(max(final_fitnesses))
    optimal_path = population[best_index]
    optimal_cost = calculate_cost(optimal_path)

    print("\n=== OPTIMAL ROUTE FOUND ===")
    print(f"Best Path : {' -> '.join(map(str, optimal_path))}")
    print(f"Total Cost: {optimal_cost}")


if __name__ == "__main__":
    run_genetic_algorithm()
