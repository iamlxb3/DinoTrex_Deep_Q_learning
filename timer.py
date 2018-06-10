class Timer:

    def __init__(self, space_time_gap):
        self.space_time_gap = space_time_gap
        self.space_t = 0

    def is_space_cooling_down(self, time_now):
        t_gap = time_now - self.space_t
        print ("t_gap: ", t_gap)
        if t_gap >= self.space_time_gap:
            return True
        else:
            return False


