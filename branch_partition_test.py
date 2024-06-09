# @title branch partition test
def partition_into_branches(graph, start):
    def dfs(node, visited):
        stack = [(node, [node])]
        branches = []

        while stack:
            current, path = stack.pop()
            visited.add(current)

            # Check if the current node is a terminal node (no unvisited neighbors)
            neighbors = [n for n in graph[current] if n not in visited]
            if not neighbors:  # Terminal node
                if len(path) > 1:
                    branches.append(path)
            else:
                for neighbor in neighbors:
                    stack.append((neighbor, path + [neighbor]))

        return branches

    visited = set()
    all_branches = []

    # Start the partition from the initial start node
    all_branches.extend(dfs(start, visited))

    # Check for any unvisited nodes and use them as new branch points
    for node in graph:
        if node not in visited:
            all_branches.extend(dfs(node, visited))

    return all_branches

# Example usage
graph = {
    0: [1, 2],
    1: [0, 3, 4],
    2: [0, 5],
    3: [1],
    4: [1],
    5: [2]
}

start_node = 1
branches = partition_into_branches(graph, start_node)
for i, branch in enumerate(branches):
    print(f"Branch {i+1}: {branch}")
