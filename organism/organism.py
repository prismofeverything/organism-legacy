import copy
import numpy as np
import collections

PAD = -1

EAT = 'EAT'
MOVE = 'MOVE'
GROW = 'GROW'
CIRCULATE = 'CIRCULATE'

def flatten(l):
    if isinstance(l, collections.Sequence) and not isinstance(l, (str, bytes, tuple)):
        return [a for i in l for a in flatten(i)]
    else:
        return [l]

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

    def move_food(self, from_space, to_space):
        self.spaces[from_space]['food'] -= 1
        self.spaces[to_space]['food'] += 1

    def move_element(self, from_space, to_space):
        self.spaces[to_space]['element'] = self.spaces[from_space]['element']
        self.spaces[to_space]['food'] = self.spaces[from_space]['food']
        self.spaces[from_space]['element'] = None
        self.spaces[from_space]['food'] = 0

    def space_player(self, space):
        if self.spaces[space]['element']:
            return self.spaces[space]['element']['player']

    def other_players_adjacent(self, space, player):
        other_players = []
        for adjacent_space in self.adjacencies[space]:
            adjacent_player = self.space_player(adjacent_space)
            if adjacent_player and adjacent_player != player:
                other_players.append(adjacent_player)
        return other_players

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



class expanding_tuple(object):
    def __init__(self, tup=None):
        self.tup = tup or ()

    def append(self, element):
        expanding = list(self.tup)
        expanding.append(element)
        self.tup = tuple(expanding)

    def tuple_for(self):
        return self.tup

    def reset(self):
        self.tup = ()


def grow_consume_options(availability, total):
    options = []
    for space, food in availability.items():
        for i in range(food):
            options.append(space)

    

class OrganismTree(object):
    def __init__(self, board, organisms, player, organism_index):
        self.board = board
        self.organisms = organisms
        self.player = player
        self.organism_index = organism_index
        self.action = None
        self.action_type = None
        self.sequence = expanding_tuple()
        self.choices = []
        self.food_limit = 0
        self.organism = self.organisms[self.player][self.organism_index]

    def clone(self):
        return copy.deepcopy(self)

    def action_code(self):
        return tuple([self.action] + self.choices)
        
    def complete_choice(self):
        self.choices.append(self.sequence.tuple_for())
        self.food_limit = 0
        self.sequence.reset()

    def walk_order(self):
        self.complete_choice()
        if len(self.choices) < self.total:
            return self.walk_choice()
        else:
            return self.action_code()

    def walk_eat_from(self, space, food_space):
        self.board.move_food(space, food_space)
        self.sequence.append(food_space)

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
                    self.clone().walk_eat_from(space, adjacent_space)
                    for adjacent_space in adjacent_food]

    def walk_eat(self):
        eat_elements = self.board.elements_of(self.organism, EAT)
        return [
            self.clone().walk_eat_to(space)
            for space in eat_elements.keys()]

    def walk_move_food(self, from_space, to_space, open_space):
        self.board.spaces[open_space]['food'] += self.board.spaces[to_space]['food']
        self.board.spaces[to_space]['food'] = 0
        self.board.move_element(from_space, to_space)
        self.sequence.append(open_space)

        return self.walk_order()

    def walk_move_to(self, from_space, to_space):
        self.sequence.append(to_space)

        open_spaces = [
            open_space
            for open_space in self.board.adjacencies[to_space]
            if open_space != from_space and not self.board.spaces[open_space]['element']]

        if self.board.spaces[to_space]['food'] > 0 and len(open_spaces) > 0:
            return [
                self.clone().walk_move_food(from_space, to_space, open_space)
                for open_space in open_spaces]
        else:
            self.board.move_element(from_space, to_space)
            return self.walk_order()

    def walk_move_from(self, space):
        board = self.board
        if self.board.space_player(space) == self.player and self.board.spaces[space]['food'] > 0:
            empty_spaces = [
                empty_space
                for empty_space in self.board.adjacencies[space]
                if not self.board.spaces[empty_space]['element']]

            if len(empty_spaces) > 0:
                self.sequence.append(space)
                return [
                    self.clone().walk_move_to(space, empty_space)
                    for empty_space in empty_spaces]

    def walk_move(self):
        move_elements = self.board.elements_of(self.organism, MOVE)
        adjacent_spaces = [
            self.board.adjacencies[move_space]
            for move_space in move_elements.keys()]
        flat_spaces = frozenset([item for sublist in adjacent_spaces for item in sublist] + list(move_elements.keys()))

        return [
            self.clone().walk_move_from(space)
            for space in flat_spaces]

    def walk_grow_food(self, into_space, open_space):
        self.board.spaces[open_space]['food'] += self.board.spaces[into_space]['food']
        self.board.spaces[into_space]['food'] = 0
        self.board.place_element(into_space, self.player, self.sequence[1])
        self.sequence.append(open_space)

        return self.walk_order()

    def walk_grow_into(self, into_space):
        self.sequence.append(into_space)

        open_spaces = [
            open_space
            for open_space in self.board.adjacencies[into_space]
            if not self.board.spaces[open_space]['element']]

        if self.board.spaces[into_space]['food'] > 0 and len(open_spaces) > 0:
            return [
                self.clone().walk_grow_food(into_space, open_space)
                for open_space in open_spaces]
        else:
            self.board.spaces[into_space]['food'] = 0
            self.board.place_element(into_space, self.player, self.sequence[1])
            return self.walk_order()

    def walk_grow_consume(self, consume):
        for consume_space, consume_food in consume.items():
            self.board.spaces[consume_space]['food'] -= consume_food
        self.sequence.append(consume)

        grow_elements = self.board.elements_of(self.organism, GROW)
        adjacent_spaces = [
            self.board.adjacencies[grow_space]
            for grow_space in grow_elements.keys()]
        flat_spaces = frozenset([item for sublist in adjacent_spaces for item in sublist])

        open_spaces = [
            adjacent
            for adjacent in flat_spaces
            if not self.board.spaces[adjacent]['element'] and not self.board.other_players_adjacent(adjacent, self.player)]
        
        if open_spaces:
            return [
                self.clone().walk_grow_into(open_space)
                for open_space in open_spaces]

    def walk_grow_element(self, element_type):
        elements = self.board.elements_of(self.organism, element_type)
        if self.food_limit >= len(elements):
            self.sequence.append(element_type)
            return [
                self.clone().walk_grow_consume(consume)
                for consume in grow_consume_options(self.food_availability, len(elements))]

    def walk_grow(self):
        self.complete_choice()
        return self.action_code()

        grow_elements = self.board.elements_of(self.organism, GROW)
        self.food_availability = {
            grow: self.board.spaces[grow]['food']
            for grow in grow_elements.keys()
            if self.board.spaces[grow]['food'] > 0}
        self.food_limit = np.sum(self.food_availability.values())

        return [
            self.clone().walk_grow_element(element_type)
            for element_type in [EAT, MOVE, GROW]]

    def walk_circulate_to(self, from_space, to_space):
        self.board.move_food(from_space, to_space)
        self.sequence.append(to_space)

        return self.walk_order()
        
    def walk_circulate_from(self, space):
        board = self.board
        if board.spaces[space]['food'] > 0:
            adjacent_spaces = [
                adjacent_space
                for adjacent_space in board.adjacencies[space]
                if board.space_player(adjacent_space) == self.player and board.spaces[adjacent_space]['food'] < 3]

            if len(adjacent_spaces) > 0:
                self.sequence.append(space)
                return [
                    self.clone().walk_circulate_to(space, adjacent_space)
                    for adjacent_space in adjacent_spaces]

    def walk_circulate(self):
        elements = self.board.elements(self.organism)
        return [
            self.clone().walk_circulate_from(space)
            for space in self.organism]

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

    def walk_choice(self):
        return [
            self.clone().walk_switch(action_type)
            for action_type in [self.action, CIRCULATE]]

    def walk_action(self, action):
        self.action = action

        elements = self.board.elements_of(self.organism, action)
        self.total = len(elements)

        return self.walk_choice()

    def walk(self):
        path = [
            self.clone().walk_action(action)
            for action in [EAT, GROW, MOVE]]

        return flatten(path)
        return [
            step
            for step in flatten(path)
            if step]
        


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
    def __init__(self, element_type, consume_food, birth_space, push_food_space):
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
    turn.take_turn([GROW, [GROW, GROW, {('blue', 2): 1}, ('blue', 1), ('blue', 0)]])
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
    walk = tree.walk()
    print(walk)
    print(len(walk))




if __name__ == '__main__':
    test_organism()

