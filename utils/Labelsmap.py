import string


class LabelTool:
    def __init__(self):
        voc = '<' + string.printable[:-6] + '>'
        self.voc = voc
        char2id = {'PADDING': 0}
        id2char = {0: 'PADDING'}
        for i, c in enumerate(self.voc):
            char2id[c] = i+1
            id2char[i+1] = c
        self.char2id = char2id
        self.id2char = id2char