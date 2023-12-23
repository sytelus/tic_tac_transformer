from collections import deque
from typing import TypeVar, Generic, Callable, Set, List, Any, Optional, Iterator, Dict

# Define a type variable
TNodeKey = TypeVar('TNodeKey')
TNodeValue = TypeVar('TNodeValue')

class Tree(Generic[TNodeValue, TNodeKey]):

    _id:int = 0

    @staticmethod
    def _next_id(node:'TNodeValue')->int:
        Tree._id += 1
        return Tree._id

    def __init__(self, value: TNodeValue, node_key_fn: Callable[[TNodeValue], TNodeKey] = _next_id):
        self.node_key_fn = node_key_fn
        self.key:TNodeKey = node_key_fn(value)
        self.value: TNodeValue = value

        self.children: Dict[TNodeKey, 'Tree'] = {}

        self.parent: Optional['Tree'] = None

    def add(self, node: 'Tree') -> 'Tree':
        assert node.key not in self.children, f"Node with key {node.key} already exists in tree"
        self.children[node.key] = node
        node.parent = self
        return node

    def remove(self, node: 'Tree') -> 'Tree':
        del self.children[node.key]
        node.parent = None
        return node

    def __contains__(self, node: 'Tree') -> bool:
        return node in self.children

    def __iter__(self)->Iterator['Tree']:
        return iter(self.children.values())

    def __len__(self) -> int:
        return len(self.children)

    def __repr__(self) -> str:
        return f"Tree({self.value})"

    def __str__(self) -> str:
        return f"Tree({self.value})"

    def count_all(self, only_leaves=False) -> int:
        count = 1 if len(self.children)==0 else 0  # start with one to count self
        for child in self.children.values():
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
            for child in self.children.values():
                aggregate = child.visit_leafs(visit_fn, aggregate)
        return aggregate

    def depth_first_traversal(self, visit_fn: Callable[[Any, Any], Any], aggregate: Any) -> Any:
        aggregate = visit_fn(self, aggregate)
        # then visit all children
        for child in self.children.values():
            aggregate = child.depth_first_traversal(visit_fn, aggregate)
        return aggregate

    def breadth_first_traversal(self, visit_fn: Callable[[Any, Any], Any], aggregate: Any) -> Any:
        queue:deque['Tree'] = deque([self])  # Using deque for efficient pops from the front
        while queue:
            current_node = queue.popleft()  # O(1) time complexity
            aggregate = visit_fn(current_node, aggregate)
            queue.extend(current_node.children.values())  # Extending to the right side, which is efficient
        return aggregate

    def pretty_print(self, prefix: str = "", is_last: bool = True) -> None:
        print(prefix + ("└── " if is_last else "├── ") + str(self.value))
        prefix += "    " if is_last else "│   "
        for i, child in enumerate(self.children.values()):
            is_last_child = i == (len(self.children) - 1)
            child.pretty_print(prefix, is_last_child)

    def ancestors(self) -> Iterator['Tree']:
        node = self
        while node is not None:
            yield node
            node = node.parent

    def descendants(self) -> Iterator['Tree']:
        for child in self.children.values():
            yield child
            for descendant in child.descendants():
                yield descendant

    def all_leaves(self) -> Iterator['Tree']:
        if self.is_leaf():
            yield self
        else:
            for child in self.children.values():
                for leaf in child.all_leaves():
                    yield leaf

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
