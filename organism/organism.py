import copy
import numpy as np

PAD = -1

EAT = 'EAT'
MOVE = 'MOVE'
GROW = 'GROW'
CIRCULATE = 'CIRCULATE'

def build_rows(rings):
    rows = []
    for ring in np.arange(rings):
        flank = np.arange(ring+1, rings)
        pad = np.full(ring // 2, PAD)
        row = np.concatenate([
            np.full(ring // 2, PAD),
            np.flip(flank, 0),
            np.full(ring+1, ring),
            flank,
            np.full(int(round(ring / 2.0)), PAD)])
        rows.append(row)

    return np.concatenate([
        np.flip(rows[1:], 0),
        rows])

def ring_length(ring):
    if ring == 0:
        return 1
    else:
        return ring * 6

def add_adjacency(adjacencies, source, destination):
    if not source in adjacencies:
        adjacencies[source] = []
    adjacencies[source].append(destination)

def symmetric_adjacency(adjacencies, source, destination):
    add_adjacency(adjacencies, source, destination)
    add_adjacency(adjacencies, destination, source)

def build_rings(rings):
    adjacencies = {}

    for ring_index, ring in enumerate(rings):
        length = ring_length(ring_index)
        for step in np.arange(length):
            space = (ring, step)

            if ring_index > 0:
                ratio = (step / float(ring_index)) * (ring_index - 1)
                inner = (rings[ring_index-1], int(np.ceil(ratio)))
                symmetric_adjacency(adjacencies, space, inner)

                corner = step % ring_index == 0
                if not corner:
                    other = (rings[ring_index-1], int(np.floor(ratio)))
                    symmetric_adjacency(adjacencies, space, other)

                symmetric_adjacency(adjacencies, space, (ring, (step+1) % length))

    return adjacencies
            

class OrganismBoard(object):
    def __init__(self, rings):
        self.rings = rings
        self.rows = build_rows(len(rings))
        self.adjacencies = build_rings(rings)
        self.spaces = {
            space: {
                'element': None,
                'food': 0}
            for space in self.adjacencies.keys()}

    def place_element(self, space, player, element):
        self.spaces[space]['element'] = {
            'player': player,
            'type': element}

    def add_food(self, space, amount):
        self.spaces[space]['food'] += amount

    def initialize_food(self, levels):
        for space, state in self.spaces.items():
            state['food'] = levels[space[0]]

    def push_food(self, from_space, to_space):
        food = self.spaces[from_space]['food']
        if self.spaces[to_space]['element'] is None:
            self.spaces[to_space]['food'] += food
        self.spaces[from_space]['food'] = 0

    def space_player(self, space):
        if self.spaces[space]['element']:
            return self.spaces[space]['element']['player']

    def adjacent_elements_of_type(self, space, player, element_type):
        return [
            adjacent_space
            for adjacent_space in self.adjacencies[space]
            if self.spaces[adjacent_space]['element'] and self.spaces[adjacent_space]['element']['player'] == player and self.spaces[adjacent_space]['element']['type'] == element_type]

    def elements(self, spaces):
        return [
            self.spaces[space]['element']
            for space in spaces]

    def elements_of(self, spaces, element_type):
        return {
            space: self.spaces[space]['element']
            for space in spaces
            if self.spaces[space]['element']['type'] == element_type}

    def find_organisms(self):
        organisms = {}
        index = 1
        for space, state in self.spaces.items():
            if state.get('element'):
                player = state['element']['player']
                if organisms.get(player):
                    found = []
                    for adjacent_space in self.adjacencies[space]:
                        looking = organisms[player].get(adjacent_space)
                        if looking:
                            organisms[player][space] = looking
                            if len(found) > 0:
                                for organism_space in organisms[player].keys():
                                    if organisms[player][organism_space] == looking:
                                        organisms[player][organism_space] = organisms[player][found[0]]
                            found.append(adjacent_space)
                    if not organisms[player].get(space):
                        organisms[player][space] = index
                        index += 1
                else:
                    organisms[player] = {}
                    organisms[player][space] = index
                    index += 1

        invert = {}
        for player, elements in organisms.items():
            if not invert.get(player):
                invert[player] = {}
            for space, index in elements.items():
                if not invert[player].get(index):
                    invert[player][index] = []
                invert[player][index].append(space)

        return invert



class OrganismTree(object):
    def __init__(self, board, organisms, player, organism_index):
        self.board = board
        self.organisms = organisms
        self.player = player
        self.organism_index = organism_index
        self.action = None
        self.action_type = None
        self.sequence = []
        self.choices = []
        self.organism = self.organisms[self.player][self.organism_index]

    def clone(self):
        return copy.deepcopy(self)

    def action_code(self):
        return [self.action] + self.choices
        
    def complete_choice(self):
        self.choices.append(self.sequence)
        self.sequence = []

    def walk_order(self):
        if len(self.choices) < self.total:
            return self.organism_tree_action()
        else:
            return self.action_code()

    def walk_eat_from(self, adjacent):
        self.sequence.append(adjacent)
        self.complete_choice()

        return self.walk_order()

    def walk_eat_to(self, space):
        board = self.board
        if board.spaces[space]['food'] < 3:
            adjacent_food = [
                adjacent_space
                for adjacent_space in board.adjacencies[space]
                if board.spaces[adjacent_space]['food'] > 0 and board.spaces[adjacent_space]['element'] is None]
            if len(adjacent_food) > 0:
                self.sequence.append(space)
                return [
                    self.clone().walk_eat_from(adjacent_space)
                    for adjacent_space in adjacent_food]

    def walk_eat(self):
        eat_elements = self.board.elements_of(self.organism, EAT)
        return [
            self.clone().walk_eat_to(space)
            for space in eat_elements.keys()]

    def walk_move_from(self, space):
        board = self.board
        self.sequence.append(space)

    def walk_move(self):
        move_elements = self.board.elements_of(self.organism, MOVE)
        adjacent_spaces = [
            self.board.adjacencies[move_space]
            for move_space in move_elements.keys()]
        flat_spaces = frozenset([item for sublist in adjacent_spaces for item in sublist] + list(move_elements.keys()))
        elements_with_food = [
            space
            for space in flat_spaces
            if self.board.space_player(space) == self.player and self.board.spaces[space]['food'] > 0]

        return [
            self.clone().walk_move_from(space)
            for space in elements_with_food]

    def walk_grow(self):
        self.complete_choice()
        return self.action_code()

    def walk_circulate(self):
        self.complete_choice()
        return self.action_code()

    def walk_switch(self, action_type):
        self.action_type = action_type
        self.sequence.append(action_type)

        if action_type == EAT:
            return self.walk_eat()
        elif action_type == MOVE:
            return self.walk_move()
        elif action_type == GROW:
            return self.walk_grow()
        elif action_type == CIRCULATE:
            return self.walk_circulate()

    def walk_action(self, action):
        self.action = action

        elements = self.board.elements_of(self.organism, action)
        self.total = len(elements)

        return [
            self.clone().walk_switch(action_type)
            for action_type in [self.action, CIRCULATE]]

    def walk(self):
        return [
            self.clone().walk_action(action)
            for action in [EAT, GROW, MOVE]]


# def organism_tree_order(state):
#     if len(state.choices) < state.total:
#         return organism_tree_action(state)
#     else:
#         return state.action_code()

# def organism_tree_eat_from(state, adjacent):
#     state.sequence.append(adjacent)
#     state.complete_choice()

#     return organism_tree_order(state)

# def organism_tree_eat_to(state, space):
#     board = state.board
#     if board.spaces[space]['food'] < 3:
#         adjacent_food = [
#             adjacent_space
#             for adjacent_space in board.adjacencies[space]
#             if board.spaces[adjacent_space]['food'] > 0 and board.spaces[adjacent_space]['element'] is None]
#         if len(adjacent_food) > 0:
#             state.sequence.append(space)
#             return [
#                 organism_tree_eat_from(copy.deepcopy(state), adjacent_space)
#                 for adjacent_space in adjacent_food]

# def organism_tree_eat(state):
#     eat_elements = state.board.elements_of(state.organism, EAT)
#     return [
#         organism_tree_eat_to(copy.deepcopy(state), space)
#         for space in eat_elements.keys()]

# def organism_tree_move(state):
#     move_elements = state.board.elements_of(state.organism, MOVE)



#     state.complete_choice()
#     return state.action_code()

# def organism_tree_grow(state):
#     state.complete_choice()
#     return state.action_code()

# def organism_tree_circulate(state):
#     state.complete_choice()
#     return state.action_code()

# def organism_tree_switch(state, action_type):
#     state.action_type = action_type
#     state.sequence.append(action_type)

#     if action_type == EAT:
#         return organism_tree_eat(state)
#     elif action_type == MOVE:
#         return organism_tree_move(state)
#     elif action_type == GROW:
#         return organism_tree_grow(state)
#     elif action_type == CIRCULATE:
#         return organism_tree_circulate(state)

# def organism_tree_action(state):
#     return [
#         organism_tree_switch(copy.deepcopy(state), action_type)
#         for action_type in [state.action, CIRCULATE]]

# def organism_tree(board, organisms, player, organism_index):
#     return [
#         organism_tree_action(
#             TurnState(board, organisms, player, organism_index, action))
#         for action in [EAT, GROW, MOVE]]
        




        


class OrganismAction(object):
    def apply_action(self, board):
        pass

class CirculateAction(OrganismAction):
    def __init__(self, from_space, to_space):
        self.from_space = from_space
        self.to_space = to_space
        
    def apply_action(self, board):
        assert board.spaces[self.from_space]['food'] > 0
        assert board.spaces[self.to_space]['food'] < 3
        
        board.spaces[self.from_space]['food'] -= 1
        board.spaces[self.to_space]['food'] += 1

class EatAction(OrganismAction):
    def __init__(self, from_space, to_space):
        self.from_space = from_space
        self.to_space = to_space

    def apply_action(self, board):
        assert board.spaces[self.from_space]['element'] == None
        assert board.spaces[self.from_space]['food'] > 0
        assert board.spaces[self.to_space]['element']['type'] == EAT
        assert board.spaces[self.to_space]['food'] < 3

        board.spaces[self.from_space]['food'] -= 1
        board.spaces[self.to_space]['food'] += 1

class MoveAction(OrganismAction):
    def __init__(self, from_space, to_space, push_food_space):
        self.from_space = from_space
        self.to_space = to_space
        self.push_food_space = push_food_space

    def apply_action(self, board):
        player = board.space_player(self.from_space)
        food = board.spaces[self.to_space]['food']
        adjacent_move_elements = board.adjacent_elements_of_type(self.from_space, player, MOVE)
        if board.spaces[self.from_space]['element']['type'] == MOVE:
            adjacent_move_elements.append(self.from_space)

        assert board.spaces[self.from_space]['food'] > 0
        assert len(adjacent_move_elements) > 0
        assert board.spaces[self.to_space]['element'] is None

        board.push_food(self.to_space, self.push_food_space)
        board.spaces[self.to_space] = board.spaces[self.from_space]
        board.spaces[self.from_space] = {
            'element': None,
            'food': 0}

class GrowAction(OrganismAction):
    def __init__(self, consume_food, element_type, birth_space, push_food_space):
        self.consume_food = consume_food
        self.element_type = element_type
        self.birth_space = birth_space
        self.push_food_space = push_food_space

    def apply_action(self, board):
        player = board.space_player(list(self.consume_food.keys())[0])
        adjacent_grow_elements = board.adjacent_elements_of_type(self.birth_space, player, GROW)

        for space, consume in self.consume_food.items():
            assert board.spaces[space]['food'] >= consume
            assert board.spaces[space]['element']['type'] == GROW

        assert board.spaces[self.birth_space]['element'] is None
        assert len(adjacent_grow_elements) > 0

        board.push_food(self.birth_space, self.push_food_space)
        for space, consume in self.consume_food.items():
            board.spaces[space]['food'] -= consume
        board.spaces[self.birth_space]['element'] = {
            'player': player,
            'type': self.element_type}

organism_actions = {
    EAT: EatAction,
    MOVE: MoveAction,
    GROW: GrowAction,
    CIRCULATE: CirculateAction}


class OrganismTurn(object):
    def __init__(self, board, organisms, player, index):
        self.board = board
        self.organisms = organisms
        self.player = player
        self.index = index
        self.action_type = None
        self.number = -1
        self.choices = []

    def choose_action_type(self, action_type):
        self.action_type = action_type
        organism_spaces = self.organisms[self.player][self.index]
        matching_elements = [
            space
            for space in organism_spaces
            if self.board.spaces[space]['element']['type'] == action_type]
        self.number = len(matching_elements)

    def choose_action(self, action):
        self.choices.append(action)

    def take_action(self, action_data):
        choice = action_data[0]
        make_action = organism_actions[choice]
        action = make_action(*action_data[1:])
        self.choose_action(action)

    def take_turn(self, turn_data):
        action_type = turn_data[0]
        choices = turn_data[1:]
        self.choose_action_type(action_type)
        for choice in choices:
            self.take_action(choice)

    def apply_actions(self, board):
        for choice in self.choices:
            choice.apply_action(board)


def player_keys(organisms, player):
    return list(organisms[player].keys())

def test_organism():
    board = OrganismBoard([
        'red',
        'orange',
        'green',
        'blue',
        'purple'])

    print(board.rows)
    print(board.adjacencies)

    test_multiple_organisms = False

    board.initialize_food({
        'purple': 1,
        'blue': 1,
        'green': 2,
        'orange': 3,
        'red': 5})

    board.place_element(('purple', 1), 'Aorwa', EAT)
    board.place_element(('purple', 2), 'Aorwa', MOVE)
    board.place_element(('purple', 3), 'Aorwa', GROW)

    board.place_element(('purple', 13), 'Maxoz', EAT)
    board.place_element(('purple', 14), 'Maxoz', MOVE)
    board.place_element(('purple', 15), 'Maxoz', GROW)

    if test_multiple_organisms:
        board.place_element(('orange', 1), 'Maxoz', EAT)
        board.place_element(('orange', 2), 'Maxoz', MOVE)
        board.place_element(('orange', 3), 'Maxoz', GROW)

    print(board.spaces)

    organisms = board.find_organisms()
    print(organisms)

    turn = OrganismTurn(board, organisms, 'Aorwa', player_keys(organisms, 'Aorwa')[0])
    turn.take_turn([MOVE, [MOVE, ('purple', 3), ('blue', 2), ('blue', 1)]])
    turn.apply_actions(board)

    print(board.spaces[('blue', 1)])
    print(board.spaces[('blue', 2)])

    organisms = board.find_organisms()
    print(organisms)

    turn = OrganismTurn(board, organisms, 'Maxoz', player_keys(organisms, 'Maxoz')[0])
    turn.take_turn([EAT, [EAT, ('blue', 9), ('purple', 13)]])
    turn.apply_actions(board)

    print(board.spaces[('blue', 9)])
    print(board.spaces[('purple', 13)])

    organisms = board.find_organisms()
    print(organisms)

    turn = OrganismTurn(board, organisms, 'Aorwa', player_keys(organisms, 'Aorwa')[0])
    turn.take_turn([GROW, [GROW, {('blue', 2): 1}, GROW, ('blue', 1), ('blue', 0)]])
    turn.apply_actions(board)

    print(board.spaces[('blue', 0)])
    print(board.spaces[('blue', 1)])
    print(board.spaces[('blue', 2)])

    organisms = board.find_organisms()
    print(organisms)

    turn = OrganismTurn(board, organisms, 'Maxoz', player_keys(organisms, 'Maxoz')[0])
    turn.take_turn([EAT, [CIRCULATE, ('purple', 13), ('purple', 14)]])
    turn.apply_actions(board)

    print(board.spaces[('purple', 13)])
    print(board.spaces[('purple', 14)])

    organisms = board.find_organisms()
    print(organisms)

    turn = OrganismTurn(board, organisms, 'Aorwa', player_keys(organisms, 'Aorwa')[0])
    turn.take_turn([GROW, [CIRCULATE, ('purple', 2), ('blue', 2)], [CIRCULATE, ('purple', 1), ('blue', 1)]])
    turn.apply_actions(board)

    print(board.spaces[('blue', 1)])
    print(board.spaces[('blue', 2)])

    organisms = board.find_organisms()
    print(organisms)

    # tree = organism_tree(board, organisms, 'Aorwa', player_keys(organisms, 'Aorwa')[0])
    tree = OrganismTree(board, organisms, 'Aorwa', player_keys(organisms, 'Aorwa')[0])
    print(tree.walk())




if __name__ == '__main__':
    test_organism()

