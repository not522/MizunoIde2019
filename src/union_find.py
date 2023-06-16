class UnionFind(object):
    def __init__(self, n):
        self.i2g = [i for i in range(n)]
        self.g2i = [[i] for i in range(n)]

    def unite(self, a, b):
        x = self.i2g[a]
        y = self.i2g[b]
        if x == y:
            return
        if len(self.g2i[x]) < len(self.g2i[y]):
            a, b = b, a
            x, y = y, x
        for i in self.g2i[y]:
            self.i2g[i] = x
        self.g2i[x] += self.g2i[y]
        self.g2i[y] = []

    def get_groups(self):
        res = []
        for group in self.g2i:
            if len(group) > 0:
                res.append(group)
        return res
