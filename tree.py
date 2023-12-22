from typing import Callable, Set, List

class Tree:
    def __init__(self, value):
        self.value = value
        self.children:List['Tree'] = []

    def add(self, node: "Tree"):
        self.children.append(node)

    def remove(self, node):
        self.children.remove(node)

    def __contains__(self, node):
        return node in self.children

    def __iter__(self):
        return iter(self.children)

    def __len__(self):
        return len(self.children)

    def __repr__(self):
        return f"Tree({self.value})"

    def __str__(self):
        return f"Tree({self.value})"

    def count_all(self):
        count = 1  # start with one to count self
        for child in self.children:
            count += child.count_all()  # recursively count all children
        return count

    def depth_first_traversal(self, visit_fn:Callable, aggregate):
        aggregate = visit_fn(self, aggregate)
        # then visit all children
        for child in self.children:
            aggregate = child.depth_first_traversal(visit_fn, aggregate)
        return aggregate

    def breadth_first_traversal(self, visit_fn:Callable, aggregate):
        queue:List['Tree'] = [self]  # start with the self
        while queue:  # while there are nodes to process
            current_node = queue.pop(0)  # remove the first node
            aggregate = visit_fn(current_node, aggregate)  # visit the node
            queue.extend(current_node.children)  # add its children to the queue
        return aggregate


# Example of usage
if __name__ == "__main__":
    # Create nodes
    root = Tree(1)
    child1 = Tree(2)
    child2 = Tree(3)
    child2_1 = Tree(4)

    # Build the tree
    root.add(child1)
    root.add(child2)
    child2.add(child2_1)

    # Example function to apply to each node
    def print_and_sum(node, aggregate):
        print(node.value)
        return node.value + aggregate

    # Execute traversals
    print("Depth-First Traversal:")
    sum_dft = root.depth_first_traversal(print_and_sum, 0)
    print("Sum of node values:", sum_dft)

    print("\nBreadth-First Traversal:")
    sum_bft = root.breadth_first_traversal(print_and_sum, 0)
    print("Sum of node values:", sum_bft)

    # Node count
    print("\nNode count in subtree rooted at root:", root.count_all())
    print("Node count in subtree rooted at child1:", child1.count_all())
