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
            

class OrganismState(object):
    def __init__(self, ):
        pass

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
    def __init__(self, player, index):
        self.player = player
        self.index = index
        self.action_type = None
        self.number = -1
        self.choices = []

    def choose_action_type(self, board, organisms, action_type):
        self.action_type = action_type
        organism_spaces = organisms[self.player][self.index]
        matching_elements = [
            space
            for space in organism_spaces
            if board.spaces[space]['element']['type'] == action_type]
        self.number = len(matching_elements)

    def choose_action(self, action):
        self.choices.append(action)

    def read_action(self, action_data):
        choice = action_data[0]
        make_action = organism_actions[choice]
        action = make_action(*action_data[1:])
        self.choose_action(action)

    def read_turn(self, board, organisms, turn_data):
        action_type = turn_data[0]
        choices = turn_data[1:]
        self.choose_action_type(board, organisms, action_type)
        for choice in choices:
            self.read_action(choice)

    def apply_actions(self, board):
        for choice in self.choices:
            choice.apply_action(board)


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

    turn = OrganismTurn('Aorwa', list(organisms['Aorwa'].keys())[0])
    turn.read_turn(board, organisms, [MOVE, [MOVE, ('purple', 3), ('blue', 2), ('blue', 1)]])
    turn.apply_actions(board)

    print(board.spaces[('blue', 1)])
    print(board.spaces[('blue', 2)])

    organisms = board.find_organisms()
    print(organisms)

    turn = OrganismTurn('Maxoz', list(organisms['Maxoz'].keys())[0])
    turn.read_turn(board, organisms, [EAT, [EAT, ('blue', 9), ('purple', 13)]])
    turn.apply_actions(board)

    print(board.spaces[('blue', 9)])
    print(board.spaces[('purple', 13)])

    organisms = board.find_organisms()
    print(organisms)

    turn = OrganismTurn('Aorwa', list(organisms['Aorwa'].keys())[0])
    turn.read_turn(board, organisms, [GROW, [GROW, {('blue', 2): 1}, GROW, ('blue', 1), ('blue', 0)]])
    turn.apply_actions(board)

    print(board.spaces[('blue', 0)])
    print(board.spaces[('blue', 1)])
    print(board.spaces[('blue', 2)])

    organisms = board.find_organisms()
    print(organisms)

    turn = OrganismTurn('Maxoz', list(organisms['Maxoz'].keys())[0])
    turn.read_turn(board, organisms, [EAT, [CIRCULATE, ('purple', 13), ('purple', 14)]])
    turn.apply_actions(board)

    print(board.spaces[('purple', 13)])
    print(board.spaces[('purple', 14)])

    organisms = board.find_organisms()
    print(organisms)



if __name__ == '__main__':
    test_organism()

