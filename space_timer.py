class SpaceTimer:

    def __init__(self, time_gap):
        self.time_gap = time_gap
        self.t1 = 0

    def start(self, time_stamp):
        self.t1 = time_stamp

    def is_cooling_down(self, time_now):
        t_gap = time_now - self.t1
        print ("t_gap: ", t_gap)
        if t_gap >= self.time_gap:
            return True
        else:
            return False


