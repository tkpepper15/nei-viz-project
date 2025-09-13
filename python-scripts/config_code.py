class ConfigId:
    def __init__(self, ra, rb, rsh, ca, cb, grid_point):
        self.ra = ra
        self.rb = rb
        self.rsh = rsh
        self.ca = ca
        self.cb = cb
        self.grid_point = grid_point

c1 = ConfigId("01", "01", "01", "01","01", "01",)

def config_id_generator(ra, rb, rsh, ca, cb, grid_point):
    grid_point = int(grid_point)
    print(c1.ra, c1.rb, c1.rsh, c1.ca, c1.cb, c1.grid_point)
    print(grid_point)

config_id_generator(c1.ra, c1.rb, c1.ca, c1.cb, c1.grid_point)


