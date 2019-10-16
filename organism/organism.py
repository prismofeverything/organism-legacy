import numpy as np

PAD = -1

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
        for space, state in self.spaces.iteritems():
            state['food'] = levels[space[0]]

    def find_organisms(self):
        organisms = {}
        index = 1
        for space, state in self.spaces.iteritems():
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
        for player, elements in organisms.iteritems():
            if not invert.get(player):
                invert[player] = {}
            for space, index in elements.iteritems():
                if not invert[player].get(index):
                    invert[player][index] = []
                invert[player][index].append(space)

        return invert


class OrganismAction(object):
    def __init__(self):
        self.action = 5


def test_organism():
    board = OrganismBoard([
        'red',
        'orange',
        'green',
        'blue',
        'purple'])

    print(board.rows)
    print(board.adjacencies)

    board.initialize_food({
        'purple': 1,
        'blue': 1,
        'green': 2,
        'orange': 3,
        'red': 5})

    board.place_element(('purple', 0), 'Aorwa', 'EAT')
    board.place_element(('purple', 1), 'Aorwa', 'MOVE')
    board.place_element(('purple', 2), 'Aorwa', 'GROW')

    board.place_element(('purple', 10), 'Maxoz', 'EAT')
    board.place_element(('purple', 11), 'Maxoz', 'MOVE')
    board.place_element(('purple', 12), 'Maxoz', 'GROW')

    print(board.spaces)

    organisms = board.find_organisms()

    print(organisms)


if __name__ == '__main__':
    test_organism()

