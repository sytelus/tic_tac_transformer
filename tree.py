from collections import deque
from typing import TypeVar, Generic, Callable, Set, List, Any, Optional, Iterator

# Define a type variable
TNodeValue = TypeVar('TNodeValue')

class Tree(Generic[TNodeValue]):
    def __init__(self, value: TNodeValue):
        self.value: TNodeValue = value
        self.children: List['Tree'] = []
        self.parent: Optional['Tree'] = None

    def add(self, node: 'Tree') -> 'Tree':
        self.children.append(node)
        node.parent = self
        return node

    def remove(self, node: 'Tree') -> 'Tree':
        self.children.remove(node)
        node.parent = None
        return node

    def __contains__(self, node: 'Tree') -> bool:
        return node in self.children

    def __iter__(self)->Iterator['Tree']:
        return iter(self.children)

    def __len__(self) -> int:
        return len(self.children)

    def __repr__(self) -> str:
        return f"Tree({self.value})"

    def __str__(self) -> str:
        return f"Tree({self.value})"

    def count_all(self, only_leaves=False) -> int:
        count = 1 if len(self.children)==0 else 0  # start with one to count self
        for child in self.children:
            count += child.count_all(only_leaves=only_leaves)  # recursively count all children
        return count

    def is_leaf(self) -> bool:
        return len(self.children)==0

    def is_root(self) -> bool:
        return self.parent is None

    def visit_leafs(self, visit_fn: Callable[[Any, Any], Any], aggregate: Any) -> Any:
        if len(self.children)==0:
            aggregate = visit_fn(self, aggregate)
        else:
            for child in self.children:
                aggregate = child.visit_leafs(visit_fn, aggregate)
        return aggregate

    def depth_first_traversal(self, visit_fn: Callable[[Any, Any], Any], aggregate: Any) -> Any:
        aggregate = visit_fn(self, aggregate)
        # then visit all children
        for child in self.children:
            aggregate = child.depth_first_traversal(visit_fn, aggregate)
        return aggregate

    def breadth_first_traversal(self, visit_fn: Callable[[Any, Any], Any], aggregate: Any) -> Any:
        queue:deque['Tree'] = deque([self])  # Using deque for efficient pops from the front
        while queue:
            current_node = queue.popleft()  # O(1) time complexity
            aggregate = visit_fn(current_node, aggregate)
            queue.extend(current_node.children)  # Extending to the right side, which is efficient
        return aggregate

    def pretty_print(self, prefix: str = "", is_last: bool = True) -> None:
        print(prefix + ("└── " if is_last else "├── ") + str(self.value))
        prefix += "    " if is_last else "│   "
        for i, child in enumerate(self.children):
            is_last_child = i == (len(self.children) - 1)
            child.pretty_print(prefix, is_last_child)

    def ancestors(self) -> List['Tree']:
        ancestors = []
        node = self
        while node is not None:
            ancestors.append(node)
            node = node.parent
        return ancestors

    def descendants(self) -> Set['Tree']:
        descendants = set()
        for child in self.children:
            descendants.add(child)
            descendants.update(child.descendants())
        return descendants

    def save(self: 'Tree', file_path: str) -> None:
        with open(file_path, 'w', encoding='utf-8') as file:
            def save_node(node: Tree, depth: int) -> None:
                file.write('  ' * depth + str(node.value) + '\n')
                for child in node.children:
                    save_node(child, depth + 1)
            save_node(self, 0)

    @staticmethod
    def load(file_path: str) -> Optional['Tree']:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        node_stack: List[Tree] = []
        root: Optional[Tree] = None
        last_depth = -1
        for line in lines:
            depth = line.count('  ')
            node_value = line.strip()
            node = Tree(node_value)
            if depth == 0:
                root = node
            else:
                while depth <= last_depth:
                    node_stack.pop()
                    last_depth -= 1
                node_stack[-1].add(node)
            node_stack.append(node)
            last_depth = depth
        return root

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

    root.pretty_print()
