import math
import sys
from collections import deque

from utils import *


class Problem:
    """The abstract class for a formal problem. You should subclass
    this and implement the methods actions and result, and possibly
    __init__, goal_test, and path_cost. Then you will create instances
    of your subclass and solve them with the various search functions."""

    def __init__(self, initial, goal=None):
        """The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal. Your subclass's constructor can add
        other arguments."""
        self.initial = initial
        self.goal = goal

    def actions(self, state):
        """Return the actions that can be executed in the given
        state. The result would typically be a list, but if there are
        many actions, consider yielding them one at a time in an
        iterator, rather than building them all at once."""
        raise NotImplementedError

    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        raise NotImplementedError

    def goal_test(self, state):
        """Return True if the state is a goal. The default method compares the
        state to self.goal or checks for state in self.goal if it is a
        list, as specified in the constructor. Override this method if
        checking against a single self.goal is not enough."""
        if isinstance(self.goal, list):
            return is_in(state, self.goal)
        else:
            return state == self.goal

    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2. If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        return c + 1

    def value(self, state):
        """For optimization problems, each state has a value. Hill Climbing
        and related algorithms try to maximize this value."""
        value = 9
        value = value - sum(s != g for (s, g) in zip(state, self.goal))
        return value
        # raise NotImplementedError


# ______________________________________________________________________________


class Node:
    """A node in a search tree. Contains a pointer to the parent (the node
    that this is a successor of) and to the actual state for this node. Note
    that if a state is arrived at by two paths, then there are two nodes with
    the same state. Also includes the action that got us to this state, and
    the total path_cost (also known as g) to reach the node. Other functions
    may add an f and h value; see best_first_graph_search and astar_search for
    an explanation of how the f and h values are handled. You will not need to
    subclass this class."""

    def __init__(self, state, parent=None, action=None, path_cost=0):
        """Create a search tree Node, derived from a parent by an action."""
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        return "<Node {}>".format(self.state)

    def __lt__(self, node):
        return self.state < node.state

    def expand(self, problem):
        """List the nodes reachable in one step from this node."""
        return [self.child_node(problem, action)
                for action in problem.actions(self.state)]

    def child_node(self, problem, action):
        """[Figure 3.10]"""
        next_state = problem.result(self.state, action)
        next_node = Node(next_state, self, action, problem.path_cost(self.path_cost, self.state, action, next_state))
        return next_node

    def solution(self):
        """Return the sequence of actions to go from the root to this node."""
        return [node.action for node in self.path()[1:]]

    def path(self):
        """Return a list of nodes forming the path from the root to this node."""
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    # We want for a queue of nodes in breadth_first_graph_search or
    # astar_search to have no duplicated states, so we treat nodes
    # with the same state as equal. [Problem: this may not be what you
    # want in other contexts.]

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        # We use the hash value of the state
        # stored in the node instead of the node
        # object itself to quickly search a node
        # with the same state in a Hash Table
        return hash(self.state)


# ______________________________________________________________________________

class EightPuzzle(Problem):
    """ The problem of sliding tiles numbered from 1 to 8 on a 3x3 board, where one of the
    squares is a blank. A state is represented as a tuple of length 9, where  element at
    index i represents the tile number  at index i (0 if it's an empty square) """

    def __init__(self, initial, goal=(1, 2, 3, 4, 5, 6, 7, 8, 0)):
        """ Define goal state and initialize a problem """
        super().__init__(initial, goal)

    def find_blank_square(self, state):
        """Return the index of the blank square in a given state"""

        return state.index(0)

    def actions(self, state):
        """ Return the actions that can be executed in the given state.
        The result would be a list, since there are only four possible actions
        in any given state of the environment """

        possible_actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        index_blank_square = self.find_blank_square(state)

        if index_blank_square % 3 == 0:
            possible_actions.remove('LEFT')
        if index_blank_square < 3:
            possible_actions.remove('UP')
        if index_blank_square % 3 == 2:
            possible_actions.remove('RIGHT')
        if index_blank_square > 5:
            possible_actions.remove('DOWN')

        return possible_actions

    def result(self, state, action):
        """ Given state and action, return a new state that is the result of the action.
        Action is assumed to be a valid action in the state """

        # blank is the index of the blank square
        blank = self.find_blank_square(state)
        new_state = list(state)

        delta = {'UP': -3, 'DOWN': 3, 'LEFT': -1, 'RIGHT': 1}
        neighbor = blank + delta[action]
        new_state[blank], new_state[neighbor] = new_state[neighbor], new_state[blank]

        return tuple(new_state)

    def goal_test(self, state):
        """ Given a state, return True if state is a goal state or False, otherwise """

        return state == self.goal

    def check_solvability(self, state):
        """ Checks if the given state is solvable """

        inversion = 0
        for i in range(len(state)):
            for j in range(i + 1, len(state)):
                if (state[i] > state[j]) and state[i] != 0 and state[j] != 0:
                    inversion += 1

        return inversion % 2 == 0

    def h(self, node):
        """ Return the heuristic value for a given state. Default heuristic function used is
        h(n) = number of misplaced tiles """

        return sum(s != g for (s, g) in zip(node.state, self.goal))

class SixteenPuzzle(Problem):
    """ The problem of sliding tiles numbered from 1 to 15 on a 4x4 board, where one of the
    squares is a blank. A state is represented as a tuple of length 16, where  element at
    index i represents the tile number  at index i (0 if it's an empty square) """

    def __init__(self, initial, goal=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0)):
        """ Define goal state and initialize a problem """
        super().__init__(initial, goal)

    def find_blank_square(self, state):
        """Return the index of the blank square in a given state"""

        return state.index(0)

    def actions(self, state):
        """ Return the actions that can be executed in the given state.
        The result would be a list, since there are only four possible actions
        in any given state of the environment """

        possible_actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        index_blank_square = self.find_blank_square(state)

        if index_blank_square % 4 == 0:
            possible_actions.remove('LEFT')
        if index_blank_square < 4:
            possible_actions.remove('UP')
        if index_blank_square % 4 == 3:
            possible_actions.remove('RIGHT')
        if index_blank_square > 11:
            possible_actions.remove('DOWN')

        return possible_actions

    def result(self, state, action):
        """ Given state and action, return a new state that is the result of the action.
        Action is assumed to be a valid action in the state """

        # blank is the index of the blank square
        blank = self.find_blank_square(state)
        new_state = list(state)

        delta = {'UP': -4, 'DOWN': 4, 'LEFT': -1, 'RIGHT': 1}
        neighbor = blank + delta[action]
        new_state[blank], new_state[neighbor] = new_state[neighbor], new_state[blank]

        return tuple(new_state)

    def goal_test(self, state):
        """ Given a state, return True if state is a goal state or False, otherwise """

        return state == self.goal

    def check_solvability(self, state):
        """ Checks if the given state is solvable """

        inversion = 0
        for i in range(len(state)):
            for j in range(i + 1, len(state)):
                if (state[i] > state[j]) and state[i] != 0 and state[j] != 0:
                    inversion += 1

        return inversion % 2 == 0

    def h(self, node):
        """ Return the heuristic value for a given state. Default heuristic function used is
        h(n) = number of misplaced tiles """

        return sum(s != g for (s, g) in zip(node.state, self.goal))

def a_star(problem, h=None, display=None):
    """Search the nodes with the lowest f scores first.
            You specify the function f(node) that you want to minimize; for example,
            if f is a heuristic estimate to the goal, then we have greedy best
            first search; if f is node.depth then we have breadth-first search.
            There is a subtlety: the line "f = memoize(f, 'f')" means that the f
            values will be cached on the nodes as they are computed. So after doing
            a best first search you can examine the f values of the path returned."""
    f = h
    node = Node(problem.initial)
    frontier = PriorityQueue('min', f)
    frontier.append(node)
    explored = set()
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            if display:
                print(len(explored), "paths have been expanded and", len(frontier), "paths remain in the frontier")
            return node
        explored.add(node.state)
        if problem.goal_test(node.state):
            return node.state
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                frontier.append(child)
            elif child in frontier:
                if f(child) < frontier[child]:
                    del frontier[child]
                    frontier.append(child)
    return None


def manhattan(self, node):
    x1, y1 = 1, 1
    array2d = [list() for f in range(9)]
    result = 0
    for i in range(9):
        array2d[i].append([x1, y1])
        x1 += 1
        if x1 % 3 == 1:
            x1 //= 3
            y1 += 1

    for j in range(9):
        m, n = 0, 0
        if node.state[j] == 0:
            continue

        n = array2d[node.state[j] - 1][0][0] - array2d[self.goal[j] - 1][0][0]
        m = array2d[node.state[j] - 1][0][1] - array2d[self.goal[j] - 1][0][1]
        result += abs(n) + abs(m)

    return result


def gaschnig(self, node):
    result = 0
    puzz = node.state
    puzz = list(puzz)
    while True:
        i = 0

        for b in range(9):
            # print(self.initial[b])
            if puzz[b] == 0:
                i = b
                break

        if not puzz[i] == self.goal[i]:
            for j in range(9):
                if puzz[j] == self.goal[i]:
                    temp = puzz[j]
                    puzz[j] = puzz[i]
                    puzz[i] = temp
                    result += 1
                    break
        else:
            for z in range(9):
                if not puzz[z] == self.goal[z]:
                    temp = puzz[z]
                    puzz[z] = puzz[i]
                    puzz[i] = temp
                    result += 1
                    break
        p = 0
        for x in range(9):
            if puzz[x] == self.goal[x]:
                p += 1
        if p == 9:
            # display(puzz)
            break

    # print(result)
    return result


def make_rand_8puzzle():
    while True:
        seq = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        random.shuffle(seq)

        puzz = EightPuzzle(tuple(seq), )

        if puzz.check_solvability(seq) is True:
            break

    print(seq)
    # print("successful!")
    return puzz

def user_input():
    # 1 2 3 4 5 6 7 0 8
    # 1 0 2 3 4 5 6 7 8
    initial_state = tuple(int(x) for x in input("enter the values: ").split())
    l = len(initial_state)
    if not l == 0:
        puzz = EightPuzzle(initial_state)
        return puzz, initial_state
    else:
        puzz = make_rand_8puzzle()

    return puzz

def hill_climbing(problem):
    # simple hill climbing
    expanded_states = 0
    current = Node(problem.initial)
    while True:
        neighbors = current.expand(problem)
        if not neighbors:
            break
        neighbor = argmax_random_tie(neighbors, key=lambda node: problem.value(node.state))
        expanded_states = expanded_states + 1
        if problem.value(neighbor.state) <= problem.value(current.state):
            break
        current = neighbor
    print("Expanded: ", expanded_states)
    return current.state

def hill_climbing_random_restart(problem):
    # 1 2 3 4 5 0 6 7 8
    # random restart hill climbing
    expanded_states = 0
    limit = 10000

    current = Node(problem.initial)
    first_time = True
    while limit >= 0:
        limit = limit - 1

        if not first_time:
            state = current.state
            l = list(state)
            random.shuffle(l)
            current.state = tuple(l)
        else:
            first_time = False

        state, expanded_states_tmp = hill_climbing_random_restart_helper(problem, current)
        expanded_states = expanded_states + expanded_states_tmp
        if problem.goal_test(state):
            break

    print("Expanded: ", expanded_states)
    return state

def hill_climbing_random_restart_helper(problem, current):
    # helper of random restart hill climbing
    expanded_states = 0
    while (True):
        neighbors = current.expand(problem)
        if not neighbors:
            break
        neighbor = argmax_random_tie(neighbors, key=lambda node: problem.value(node.state))
        expanded_states = expanded_states + 1
        if problem.value(neighbor.state) <= problem.value(current.state):
            break
        current = neighbor
    return current.state, expanded_states

def hill_climbing_simulated_annealing(problem):
    # not complete
    # 1 2 3 4 5 0 6 7 8
    # simulated annealing hill climbing
    # https://medium.com/swlh/how-to-implement-simulated-annealing-algorithm-in-python-ab196c2f56a0
    expanded_states = 0
    initial_temp = 90
    final_temp = .1
    alpha = 0.01
    current_temp = initial_temp
    current = Node(problem.initial)

    while current_temp > final_temp:
        neighbors = current.expand(problem)
        if not neighbors:
            break
        index = random.randint(0, len(neighbors) - 1)
        neighbor = neighbors[index]
        expanded_states = expanded_states + 1
        delta_cost = problem.value(neighbor.state) - problem.value(current.state)
        if delta_cost > 0:
            current = neighbor
            if problem.goal_test(current.state):
                break
        elif random.uniform(0, 1) <= math.exp(delta_cost / current_temp):
            current = neighbor
        current_temp -= alpha

    print("Expanded: ", expanded_states)
    return current

def local_beam(problem):
    # https://en.wikipedia.org/wiki/Beam_search
    # not complete, not optimal
    # 1 2 3 4 5 0 6 7 8
    # local beam
    k = 10
    expanded_states = 0
    current = Node(problem.initial)
    k_list = []
    for i in range(k):
        state = current.state
        l = list(state)
        random.shuffle(l)
        current.state = tuple(l)
        k_list.append(current)

    while True:
        for random_state in k_list:
            neighbors = random_state.expand(problem)
            if not neighbors:
                break
            neighbor = argmax_random_tie(neighbors, key=lambda node: problem.value(node.state))
            expanded_states = expanded_states + 1
            if problem.goal_test(neighbor.state):
                current = neighbor
                break
            k_list.pop(0)
            k_list.append(neighbor)

    print("Expanded: ", expanded_states)
    return current


puzzle, state = user_input()
# print(puzzle.find_blank_square(state))
# print(hill_climbing(puzzle))
# print(hill_climbing_random_restart(puzzle))
# print(hill_climbing_simulated_annealing(puzzle))
print(local_beam(puzzle))

