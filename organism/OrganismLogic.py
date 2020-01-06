import copy
from math import factorial
import numpy as np
import matplotlib.pyplot as plt
import pprint
import collections
from itertools import combinations

pp = pprint.PrettyPrinter(indent=4)

PAD = -1

EAT = 'EAT'
MOVE = 'MOVE'
GROW = 'GROW'
CIRCULATE = 'CIRCULATE'

element_heterarchy = {
    EAT: GROW,
    GROW: MOVE,
    MOVE: EAT
}

def items(d):
    return list(d.items())

def flatten(l):
    if isinstance(l, collections.Sequence) and not isinstance(l, (str, bytes, tuple)):
        return [a for i in l for a in flatten(i)]
    else:
        return [l]

def choose(n, k):
    return factorial(n) / factorial(k) / factorial(n - k)

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
                inner = (rings[ring_index-1], int(np.ceil(ratio)) % ring_length(ring_index-1))
                symmetric_adjacency(adjacencies, space, inner)

                corner = step % ring_index == 0
                if not corner:
                    other = (rings[ring_index-1], int(np.floor(ratio)))
                    symmetric_adjacency(adjacencies, space, other)

                symmetric_adjacency(adjacencies, space, (ring, (step+1) % length))

    return adjacencies

def topological_sort(edges):
    """
    Returns a sorted list of directed edges, in the topological order of the
    source nodes.
    """
    nodes = []
    for edge in edges:
        nodes.extend(list(edge))
    nodes = list(set(nodes))

    # Calculate in-degrees for each node
    in_degrees = {node: 0 for node in nodes}

    for edge in edges:
        in_degrees[edge[1]] += 1

    # Initialize node queue and sorted list
    sorted_nodes = []
    node_queue = []

    for node in nodes:
        if in_degrees[node] == 0:
            node_queue.append(node)

    # Do sorting of nodes
    while len(node_queue) > 0:
        next_node = node_queue.pop(0)

        for edge in edges:
            if edge[0] == next_node:
                in_degrees[edge[1]] -= 1

                if in_degrees[edge[1]] == 0:
                    node_queue.append(edge[1])

        sorted_nodes.append(next_node)

    # Get sorted edge list
    sorted_edges = []

    for node in sorted_nodes:
        for edge in edges:
            if edge[0] == node:
                sorted_edges.append(edge)

    return sorted_edges


class OrganismBoard(object):
    def __init__(self, rings, n_tokens_to_win=7, n_organisms_to_win=3):
        self.rings = rings
        self.n_rings = len(rings)
        self.rows = build_rows(len(rings))
        self.adjacencies = build_rings(rings)
        self.spaces = {
            space: {
                'element': None,
                'food': 0}
            for space in self.adjacencies.keys()}

        # Current player and the indexes of organisms that have not been moved
        # yet
        self.current_player = None
        self.organisms_to_move = []

        self.element_order = {
            EAT: 0, MOVE: 1, GROW: 2
        }
        self.space_to_axial_coords = {}
        self.axial_coords_to_space = {}

        for space in self.adjacencies.keys():
            axial_coords = self._build_axial_coords(space)
            self.space_to_axial_coords[space] = axial_coords
            self.axial_coords_to_space[axial_coords] = space

        # Initialize win conditions
        self.n_tokens_to_win = n_tokens_to_win
        self.n_organisms_to_win = n_organisms_to_win

        # Initialize counts of tokens for each player
        self.tokens = {}

    def place_element(self, space, player, element):
        self.spaces[space]['element'] = {
            'player': player,
            'type': element}

        if player not in self.tokens:
            self.tokens[player] = 0

    def add_food(self, space, amount):
        self.spaces[space]['food'] += amount

    def initialize_food(self, levels):
        for space, state in items(self.spaces):
            state['food'] = levels[space[0]]

    def set_current_player(self, player):
        """
        Set the given player to be the current player of the game.
        """
        self.current_player = player
        self.organisms_to_move = [x for x in self.find_organisms()[self.current_player].values()]

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

    def obstructing_elements(self, space, player, element_type):
        return [
            adjacent_space
            for adjacent_space in self.adjacencies[space]
            if self.spaces[adjacent_space]['element'] and self.spaces[adjacent_space]['element']['player'] != player and self.spaces[adjacent_space]['element']['type'] == element_type]

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
        for space, state in items(self.spaces):
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
        for player, elements in items(organisms):
            if not invert.get(player):
                invert[player] = {}
            for space, index in items(elements):
                if not invert[player].get(index):
                    invert[player][index] = []
                invert[player][index].append(space)

        return invert

    def find_players(self):
        """
        Return list of all player names.
        """
        return self.tokens.keys()

    def find_winners(self):
        """
        Returns a list of players that have won the game. Returns empty list
        if the game has not ended yet.
        """
        winners = []
        all_players = self.find_players()

        # Determine if game has ended
        game_ended = False

        # One or more players have enough organisms or tokens
        organisms = self.find_organisms()

        if self.n_organisms_to_win > 0:
            n_organisms = {}
            for player in all_players:
                n_organisms[player] = len(organisms.get(player, []))
                if n_organisms[player] >= self.n_organisms_to_win:
                    game_ended = True

        if self.n_tokens_to_win > 0:
            for player, n_tokens in self.tokens.items():
                if n_tokens >= self.n_tokens_to_win:
                    game_ended = True

        # One or more players eliminated from board
        players_with_organisms = list(organisms.keys())

        if len(players_with_organisms) < 2:
            game_ended = True

        # If game ended, determine winner and add to list of winners
        if game_ended:
            # Find players with organisms equal to or more than the number
            # to win
            if self.n_organisms_to_win > 0:
                for player, n_player_organisms in n_organisms.items():
                    if n_player_organisms >= self.n_organisms_to_win:
                        winners.append(player)

            # Find players with tokens equal to or more than the number to win
            if len(winners) == 0:
                if self.n_tokens_to_win > 0:
                    for player, n_tokens in self.tokens.items():
                        if n_tokens >= self.n_tokens_to_win:
                            winners.append(player)

            # Find eliminated players
            if len(winners) == 0:
                if len(players_with_organisms) == 1:
                    winners.append(players_with_organisms[0])

                # If no organisms exist on the board, current player is the
                # winner (Opponent committed suicide)
                elif len(players_with_organisms) == 0:
                    winners.append(self.current_player)

        else:
            # Check if current player does not have any valid moves
            player_organisms = organisms[self.current_player]
            valid_moves = []

            for index, organism in player_organisms.items():
                organism_is_movable = False

                for movable_organism in self.organisms_to_move:
                    if set(organism) == set(movable_organism):
                        organism_is_movable = True
                        break

                if organism_is_movable:
                    tree = OrganismTree(self, organisms, self.current_player, index)
                    walk = tree.walk()
                    valid_moves.extend(walk)

            # If current player does not have any valid moves the opponent wins
            if len(valid_moves) == 0:
                for player in all_players:
                    if player != self.current_player:
                        winners.append(player)

        return list(set(winners))

    def resolve_conflicts(self):
        """
        Resolves all conflicts between adjacent elements on the board.
        """
        conflicts = []  # List of (victim_space, attacking_space) tuples

        for space, state in items(self.spaces):
            if state['element'] is not None:
                player = state['element']['player']
                element_type = state['element']['type']

                for adjacent_space in self.adjacencies[space]:
                    adjacent_state = self.spaces[adjacent_space]

                    if adjacent_state['element'] is not None:
                        adjacent_player = adjacent_state['element']['player']
                        adjacent_element_type = adjacent_state['element']['type']

                        if adjacent_player != player and adjacent_element_type == element_heterarchy[element_type]:
                            conflicts.append((adjacent_space, space))
                            self.tokens[player] += 1

        # Topologically sort the list of conflicts
        sorted_conflicts = topological_sort(conflicts)

        # Dissolve elements for each conflict
        for conflict in sorted_conflicts:
            self._dissolve_element(*conflict)

    def _dissolve_element(self, victim_space, attacking_space):
        """
        Dissolves the element inside victim_space. Food that existed in this
        space is claimed by the element in attacking_space.
        """
        # Add food to attacking space
        self.spaces[attacking_space]['food'] += self.spaces[victim_space]['food']

        # Remove element and food
        self.spaces[victim_space]['element'] = None
        self.spaces[victim_space]['food'] = 0

    def check_organism_integrity(self):
        """
        Checks integrity of each organism, and removes any organisms that lack
        any of the three elements.
        """
        full_set = {'EAT', 'GROW', 'MOVE'}
        organisms = self.find_organisms()

        for player, player_organisms in organisms.items():
            for organism in player_organisms.values():
                element_types = [self.spaces[space]['element']['type'] for space in organism]

                if set(element_types) != full_set:
                    for space in organism:
                        self._remove_element(space)

                    n_removed_elements = len(organism)

                    # Add tokens for all other players
                    for p in self.find_players():
                        if p != player:
                            self.tokens[p] += n_removed_elements

    def _remove_element(self, space):
        """
        Removes element in given space, leaving one food behind.
        """
        self.spaces[space]['element'] = None
        self.spaces[space]['food'] += 1

    def draw(self, filename='test.png', ax=None):
        """
        Draw the state of the board using matplotlib, output as file.
        """
        n_rings = len(self.rings)

        if ax is None:
            ax_given = False
            fig, ax = plt.subplots(figsize=(5, 5))
        else:
            ax_given = True

        ax.set_axis_off()
        ax.set_xlim([-n_rings, n_rings])
        ax.set_ylim([-n_rings, n_rings])

        # Get coordinates of each board position
        coords = {}

        for ring_index, ring in enumerate(self.rings):
            if ring_index == 0:
                c = (0, 0)
                coords[(ring, 0)] = c
            else:
                for step in range(6*ring_index):
                    edge_index = step // ring_index
                    position_index = step % ring_index

                    c_x = (ring_index - position_index)*np.cos(edge_index*np.pi/3) + position_index*np.cos((edge_index + 1)*np.pi/3)
                    c_y = (ring_index - position_index)*np.sin(edge_index*np.pi/3) + position_index*np.sin((edge_index + 1)*np.pi/3)

                    c = (c_x, c_y)
                    coords[(ring, step)] = c

        # Designate color for each player
        player_colors = ['#6D1300', '#001A6D', '#006D12']
        players = self.find_players()

        player2color = {}
        for i, player in enumerate(players):
            player2color[player] = player_colors[i % 3]

        # Get locations of organisms that can be moved
        movable_spaces = []
        for spaces in self.organisms_to_move:
            movable_spaces.extend(spaces)

        # Add highlights for movable positions
        for space in movable_spaces:
            # Get coordinates for space
            c = coords[space]

            background = plt.Circle(c, 0.58, fill=True, fc='#ccff00')
            ax.add_artist(background)

        # Draw board state for each position
        for space in self.spaces.keys():
            # Get coordinates for space
            c = coords[space]

            # Add background for each board position
            background = plt.Circle(c, 0.45, fill=True, fc='#dddddd')
            ax.add_artist(background)

            if self.spaces[space]['element'] is not None:
                # Add colored circles for each element type
                type = self.spaces[space]['element']['type']

                if type == EAT:
                    element = plt.Circle(c, 0.45, fill=True, fc='#000000')
                elif type == MOVE:
                    element = plt.Circle(c, 0.45, fill=True, fc='#800080')
                else:
                    element = plt.Circle(c, 0.45, fill=True, fc='#228B22')

                ax.add_artist(element)

                # Add empty circles for each player name
                player = self.spaces[space]['element']['player']
                player_circle = plt.Circle(c, 0.45, fill=None, ec=player2color[player], lw=3)
                ax.add_artist(player_circle)

            # Add food
            n_food = self.spaces[space]['food']

            if n_food == 0:
                continue
            elif n_food == 1:
                ax.plot(c[0], c[1], marker='D', color='r', ms=4)
            elif n_food == 2:
                ax.plot([c[0] - 0.15, c[0] + 0.15],
                        [c[1], c[1]],
                        marker='D', color='r', ms=4, ls='')
            elif n_food == 3:
                ax.plot([c[0] - 0.15, c[0] + 0.15, c[0]],
                        [c[1] + 0.08, c[1] + 0.08, c[1] - 0.14],
                        marker='D', color='r', ms=4, ls='')
            elif n_food == 4:
                ax.plot([c[0] - 0.13, c[0] - 0.13, c[0] + 0.13, c[0] + 0.13],
                        [c[1] - 0.13, c[1] + 0.13, c[1] - 0.13, c[1] + 0.13],
                        marker='D', color='r', ms=4, ls='')
            else:
                ax.plot(c[0] - 0.2, c[1], marker='D', color='r', ms=4)
                ax.text(c[0] - 0.05, c[1] - 0.05, 'x' + str(n_food), color='r', verticalalignment='center')

        # Add current player
        ax.text(0, n_rings - 0.5,
            'Current player: %s' % (self.current_player, ),
            horizontalalignment='center')

        # Add number of tokens each player has won
        token_str = ''
        for player, n_tokens in self.tokens.items():
            token_str += '%s: %2d\n' % (player, n_tokens)

        ax.text(0, -n_rings - 0.5, token_str, horizontalalignment='center')

        if not ax_given:
            # Save figure
            plt.savefig(filename)
            plt.close()

    def _build_axial_coords(self, space):
        """
        For a given space, identify the x and y coordinates of the position of
        the space after the hexagonal lattice is converted into an axial square
        lattice.
        """
        ring, space_index = space
        ring_index = self.rings.index(ring)

        c = self.n_rings - 1  # Center point is (c, c)

        if ring_index == 0:
            return c, c
        else:
            if space_index <= ring_index:
                return c - space_index, c + ring_index
            elif space_index <= 2*ring_index:
                return c - ring_index, c + 2*ring_index - space_index
            elif space_index <= 3*ring_index:
                return c - 3*ring_index + space_index, c - space_index + 2*ring_index
            elif space_index <= 4*ring_index:
                return c + space_index - 3*ring_index, c - ring_index
            elif space_index <= 5*ring_index:
                return c + ring_index, c - 5*ring_index + space_index
            else:
                return c + 6*ring_index - space_index, c + space_index - 5*ring_index

    def get_array_form(self, first_player):
        """
        Gets the "array form" of the board that will be fed into the neural
        network in the perspective of the player given.
        Arguments:
            first_player: name of the first player
        """
        array_form = np.zeros((11, 2*self.n_rings - 1, 2*self.n_rings - 1))

        # Get name of opponent (assumes there are just two players)
        opponent = self._find_opponent(first_player)

        # Loop through each space, add array elements in appropriate places
        for space, specs in self.spaces.items():
            coords = self.space_to_axial_coords[space]

            # Add food to 7th layer
            array_form[6, coords[0], coords[1]] = specs['food']

            if specs['element'] is not None:
                player = specs['element']['player']
                element_type = specs['element']['type']

                array_form[3*(player == opponent) + self.element_order[element_type], coords[0], coords[1]] = 1

        # Add layers for remaining tokens to win game
        array_form[7, :, :] = self.n_tokens_to_win - self.tokens[first_player]
        array_form[8, :, :] = self.n_tokens_to_win - self.tokens[opponent]

        # Add layers to indicate moveable organisms and current player
        for spaces in self.organisms_to_move:
            for space in spaces:
                coords = self.space_to_axial_coords[space]
                array_form[9, coords[0], coords[1]] = 1

        array_form[10, :, :] = 2*((first_player == self.current_player) - 0.5)

        return array_form

    def fill_from_array(self, array, player_list):
        """
        Fills the game board using data in array.
        """
        assert len(player_list) == 2

        elements_array = array[:6, :, :]

        # Fill in elements and food
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                space = self.axial_coords_to_space.get((i, j), None)

                if space is None:
                    continue

                k = np.where(elements_array[:, i, j])[0]

                if len(k) == 1:
                    k = k[0]

                    if k == 0:
                        self.place_element(space, player_list[0], EAT)
                    elif k == 1:
                        self.place_element(space, player_list[0], MOVE)
                    elif k == 2:
                        self.place_element(space, player_list[0], GROW)
                    elif k == 3:
                        self.place_element(space, player_list[1], EAT)
                    elif k == 4:
                        self.place_element(space, player_list[1], MOVE)
                    elif k == 5:
                        self.place_element(space, player_list[1], GROW)

                self.spaces[space]['food'] = int(array[6, i, j])

        # Get current number of tokens for each player
        self.tokens[player_list[0]] = self.n_tokens_to_win - int(array[7, 0, 0])
        self.tokens[player_list[1]] = self.n_tokens_to_win - int(array[8, 0, 0])

        # Get active organisms and current player
        if array[10, 0, 0] == 1:
            self.current_player = player_list[0]
        else:
            self.current_player = player_list[1]

        org_dict = self.find_organisms().get(self.current_player, {})
        player_organisms = [x for x in org_dict.values()]
        organism_indexes = []

        for i, j in zip(np.where(array[9, :, :])[0], np.where(array[9, :, :])[1]):
            space = self.axial_coords_to_space[(i, j)]

            for index, organism in enumerate(player_organisms):
                if space in organism:
                    organism_indexes.append(index)

        organism_indexes = list(set(organism_indexes))

        self.organisms_to_move = [player_organisms[i] for i in organism_indexes]

    def update_turn_data(self, player, organism_index):
        """
        Updates the self.current_player and self.organisms_to_move variables
        based on the move made in current turn.
        """
        # Get old index of given organism
        organism_spaces = self.find_organisms()[player][organism_index]
        old_index = None

        for index, spaces in enumerate(self.organisms_to_move):
            if set(spaces) == set(organism_spaces):
                old_index = index
                break

        # Assert move is valid
        assert player == self.current_player
        assert old_index is not None

        self.organisms_to_move.pop(old_index)

        # Switch player if current player has moved all organisms
        if len(self.organisms_to_move) == 0:
            self.set_current_player(self._find_opponent(player))

    def _find_opponent(self, player):
        """
        Finds the name of the opponent of given player. Assumes there are only
        two players.
        """
        all_players = list(self.find_players())
        all_players.remove(player)
        assert len(all_players) == 1

        return all_players[0]

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


def histogram(events):
    result = {}
    for event in events:
        if not result.get(event):
            result[event] = 0
        result[event] += 1
    return tuple(sorted(result.items()))

def grow_consume_options(availability, total):
    options = []
    for space, food in items(availability):
        for i in range(food):
            options.append(space)
    
    return frozenset([
        histogram(consume)
        for consume in combinations(options, total)])


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
        self.board.move_food(food_space, space)
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
        self.organism.remove(from_space)
        self.organism.append(to_space)
        self.sequence.append(open_space)

        return self.walk_order()

    def walk_move_to(self, from_space, to_space):
        self.sequence.append(to_space)

        open_spaces = [
            open_space
            for open_space in self.board.adjacencies[to_space]
            if open_space != from_space and self.board.spaces[open_space]['element'] is None]

        if self.board.spaces[to_space]['food'] > 0 and len(open_spaces) > 0:
            return [
                self.clone().walk_move_food(from_space, to_space, open_space)
                for open_space in open_spaces]
        else:
            self.board.move_element(from_space, to_space)
            self.organism.remove(from_space)
            self.organism.append(to_space)

            return self.walk_order()

    def walk_move_from(self, space):
        board = self.board
        player = self.board.space_player(space)
        element_type = self.board.spaces[space]['element']['type']

        if self.board.spaces[space]['food'] > 0:
            empty_spaces = [
                empty_space
                for empty_space in self.board.adjacencies[space]
                if self.board.spaces[empty_space]['element'] is None and len(self.board.obstructing_elements(empty_space, player, element_type)) == 0]

            if len(empty_spaces) > 0:
                self.sequence.append(space)
                return [
                    self.clone().walk_move_to(space, empty_space)
                    for empty_space in empty_spaces]

    def walk_move(self):
        try:
            move_elements = self.board.elements_of(self.organism, MOVE)
        except TypeError:
            player_organisms = self.board.find_organisms()[self.player]
            for organism in player_organisms.values():
                break_loop = False
                for pos in self.organism:
                    if pos in organism:
                        self.organism = organism
                        break_loop = True
                        break

                if break_loop:
                    break

            move_elements = self.board.elements_of(self.organism, MOVE)

        adjacent_spaces = [
            self.board.adjacencies[move_space]
            for move_space in move_elements.keys()]
        flat_spaces = frozenset([
            item
            for sublist in adjacent_spaces
            for item in sublist
            if self.board.space_player(item) == self.player] + list(move_elements.keys()))

        return [
            self.clone().walk_move_from(space)
            for space in flat_spaces]

    # def walk_grow_food(self, into_space, open_space):
    #     self.board.spaces[open_space]['food'] += self.board.spaces[into_space]['food']
    #     self.board.spaces[into_space]['food'] = 0
    #     self.board.place_element(into_space, self.player, self.sequence.tup[1])
    #     self.sequence.append(open_space)

    #     return self.walk_order()

    def walk_grow_into(self, into_space):
        self.board.place_element(into_space, self.player, self.sequence.tup[1])
        self.board.spaces[into_space]['food'] = 0
        self.organism.append(into_space)
        self.sequence.append(into_space)

        return self.walk_order()

        # self.sequence.append(into_space)
        # open_spaces = [
        #     open_space
        #     for open_space in self.board.adjacencies[into_space]
        #     if not self.board.spaces[open_space]['element']]

        # if self.board.spaces[into_space]['food'] > 0 and len(open_spaces) > 0:
        #     return [
        #         self.clone().walk_grow_food(into_space, open_space)
        #         for open_space in open_spaces]
        # else:
        #     self.board.spaces[into_space]['food'] = 0
        #     self.board.place_element(into_space, self.player, self.sequence.tup[1])
        #     return self.walk_order()

    def walk_grow_consume(self, consume):
        for consume_space, consume_food in consume:
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
        grow_elements = self.board.elements_of(self.organism, GROW)
        self.food_availability = {
            grow: self.board.spaces[grow]['food']
            for grow in grow_elements.keys()
            if self.board.spaces[grow]['food'] > 0}
        self.food_limit = sum(self.food_availability.values())

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

        return sorted(frozenset([
            step
            for step in flatten(path)
            if step]))
        


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
    def __init__(self, eat_space, food_space):
        self.eat_space = eat_space
        self.food_space = food_space

    def apply_action(self, board):
        assert board.spaces[self.eat_space]['element']['type'] == EAT
        assert board.spaces[self.eat_space]['food'] < 3
        assert board.spaces[self.food_space]['element'] == None
        assert board.spaces[self.food_space]['food'] > 0

        board.spaces[self.food_space]['food'] -= 1
        board.spaces[self.eat_space]['food'] += 1

class MoveAction(OrganismAction):
    def __init__(self, from_space, to_space, push_food_space=None):
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

        if self.push_food_space is not None:
            board.push_food(self.to_space, self.push_food_space)
        board.spaces[self.to_space] = board.spaces[self.from_space]
        board.spaces[self.from_space] = {
            'element': None,
            'food': 0}

class GrowAction(OrganismAction):
    def __init__(self, element_type, consume_food, birth_space):
        self.consume_food = consume_food
        self.element_type = element_type
        self.birth_space = birth_space

    def apply_action(self, board):
        player = board.space_player(self.consume_food[0][0])
        adjacent_grow_elements = board.adjacent_elements_of_type(self.birth_space, player, GROW)

        for space, consume in self.consume_food:
            assert board.spaces[space]['food'] >= consume
            assert board.spaces[space]['element']['type'] == GROW

        assert board.spaces[self.birth_space]['element'] is None
        assert len(adjacent_grow_elements) > 0

        board.spaces[self.birth_space]['food'] = 0
        for space, consume in self.consume_food:
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
        board.update_turn_data(self.player, self.index)

        for choice in self.choices:
            choice.apply_action(board)

        board.resolve_conflicts()
        board.check_organism_integrity()


def player_keys(organisms, player):
    return list(organisms[player].keys())


def test_proximity():
    board = OrganismBoard([
        'red',
        'orange',
        'green'])

    board.initialize_food({
        'green': 1,
        'orange': 2,
        'red': 3})

    board.place_element(('green', 1), 'Balam', EAT)
    board.place_element(('green', 2), 'Balam', MOVE)
    board.place_element(('green', 3), 'Balam', GROW)

    board.place_element(('green', 7), 'Omdor', EAT)
    board.place_element(('green', 8), 'Omdor', MOVE)
    board.place_element(('green', 9), 'Omdor', GROW)

    board.set_current_player('Balam')
    board.draw(filename='small_board.png')

    organisms = board.find_organisms()
    print(organisms)

    turn = OrganismTurn(board, organisms, 'Balam', player_keys(organisms, 'Balam')[0])
    turn.take_turn([MOVE, [MOVE, ('green', 2), ('orange', 1), ('orange', 0)]])
    # turn.take_turn([GROW, [GROW, MOVE, ((('green', 3), 1),), ('orange', 1)]])
    turn.apply_actions(board)
    organisms = board.find_organisms()

    board.draw(filename='balam1.png')

    turn = OrganismTurn(board, organisms, 'Omdor', player_keys(organisms, 'Omdor')[0])
    turn.take_turn([MOVE, [MOVE, ('green', 7), ('orange', 4), ('orange', 3)]])
    turn.apply_actions(board)
    organisms = board.find_organisms()

    board.draw(filename='omdor1.png')

    turn = OrganismTurn(board, organisms, 'Balam', player_keys(organisms, 'Balam')[0])
    turn.take_turn([MOVE, [MOVE, ('green', 1), ('orange', 0), ('orange', 5)]])
    turn.apply_actions(board)
    organisms = board.find_organisms()

    board.draw(filename='balam2.png')

    tree = OrganismTree(board, organisms, 'Balam', player_keys(organisms, 'Balam')[0])
    walk = tree.walk()
    print(walk)

    moves = [
        move
        for action in walk if action[0] == 'MOVE'
        for move in action[1:] if move[0] == 'MOVE']

    print(moves)
    print(board.obstructing_elements(('red', 0), 'Balam', EAT))

    for move in moves:
        assert not (move[1] == ('orange', 0) and move[2] == ('red', 0))

    turn = OrganismTurn(board, organisms, 'Omdor', player_keys(organisms, 'Omdor')[0])
    turn.take_turn([GROW, [GROW, EAT, ((('green', 9), 1),), ('green', 10)]])
    turn.apply_actions(board)
    organisms = board.find_organisms()

    board.draw(filename='omdor2.png')

    tree = OrganismTree(board, organisms, 'Omdor', player_keys(organisms, 'Omdor')[0])
    walk = tree.walk()
    print(walk)

    eats = [
        action
        for action in walk if action[0] == EAT and action[1][0] == EAT and action[2][0] == EAT]

    for eat in eats:
        print(eat)
        food_spaces = [
            action[2] for action in eat[1:]]
        print(food_spaces)
        assert not (food_spaces[0] == ('green', 11) and food_spaces[0] == food_spaces[1])


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

    board.set_current_player('Aorwa')
    board.draw(filename='initial_board.png')

    organisms = board.find_organisms()
    print(organisms)

    turn = OrganismTurn(board, organisms, 'Aorwa', player_keys(organisms, 'Aorwa')[0])
    turn.take_turn([MOVE, [MOVE, ('purple', 3), ('blue', 2), ('blue', 1)]])
    turn.apply_actions(board)

    print(board.spaces[('blue', 1)])
    print(board.spaces[('blue', 2)])

    board.draw(filename='aorwa1.png')

    organisms = board.find_organisms()
    print(organisms)

    tree = OrganismTree(board, organisms, 'Aorwa', player_keys(organisms, 'Aorwa')[0])
    walk = tree.walk()
    pp.pprint(walk)
    print(len(walk))

    turn = OrganismTurn(board, organisms, 'Maxoz', player_keys(organisms, 'Maxoz')[0])
    turn.take_turn([EAT, [EAT, ('purple', 13), ('blue', 9)]])
    turn.apply_actions(board)

    print(board.spaces[('blue', 9)])
    print(board.spaces[('purple', 13)])

    board.draw(filename='maxoz1.png')

    organisms = board.find_organisms()
    print(organisms)

    turn = OrganismTurn(board, organisms, 'Aorwa', player_keys(organisms, 'Aorwa')[0])
    turn.take_turn([GROW, [GROW, GROW, ((('blue', 2), 1),), ('blue', 1)]])
    turn.apply_actions(board)

    print(board.spaces[('blue', 0)])
    print(board.spaces[('blue', 1)])
    print(board.spaces[('blue', 2)])

    board.draw(filename='aorwa2.png')

    organisms = board.find_organisms()
    print(organisms)

    turn = OrganismTurn(board, organisms, 'Maxoz', player_keys(organisms, 'Maxoz')[0])
    turn.take_turn([EAT, [CIRCULATE, ('purple', 13), ('purple', 14)]])
    turn.apply_actions(board)

    print(board.spaces[('purple', 13)])
    print(board.spaces[('purple', 14)])

    board.draw(filename='maxoz2.png')

    organisms = board.find_organisms()
    print(organisms)

    turn = OrganismTurn(board, organisms, 'Aorwa', player_keys(organisms, 'Aorwa')[0])
    turn.take_turn([GROW, [CIRCULATE, ('purple', 2), ('blue', 2)], [CIRCULATE, ('purple', 1), ('blue', 1)]])
    turn.apply_actions(board)

    print(board.spaces[('blue', 1)])
    print(board.spaces[('blue', 2)])

    board.draw(filename='aorwa3.png')

    organisms = board.find_organisms()
    print(organisms)

    tree = OrganismTree(board, organisms, 'Aorwa', player_keys(organisms, 'Aorwa')[0])
    walk = tree.walk()
    pp.pprint(walk)
    print(len(walk))


if __name__ == '__main__':
    test_organism()
    test_proximity()
