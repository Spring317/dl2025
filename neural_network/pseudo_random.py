class PseudoRandom:
    def __init__(self, seed=None):
        # Use system time if no seed is provided
        if seed is None:
            seed = int(sum(ord(c) for c in str(__import__('time').time())))
        
        self.state = seed
        self.a = 48271
        self.c = 0
        self.m = 2**31 - 1 
    
    def _next_int(self):
        """Generate the next integer in the sequence"""
        self.state = (self.a * self.state + self.c) % self.m
        return self.state
    
    def random(self):
        """Return a random float in the range [0.0, 1.0)"""
        return self._next_int() / self.m
    
    def randint(self, a, b):
        """Return a random integer in the range [a, b]"""
        return a + int(self.random() * (b - a + 1))
    
    def choice(self, seq):
        """Return a random element from the sequence"""
        return seq[self.randint(0, len(seq) - 1)]
    
    def shuffle(self, seq):
        """Shuffle the sequence in place"""
        for i in range(len(seq) - 1, 0, -1):
            j = self.randint(0, i)
            seq[i], seq[j] = seq[j], seq[i]
        return seq
    
    def randrange(self, start, stop=None, step=1):
        """Return a random integer from range(start, stop, step)"""
        if stop is None:
            stop = start
            start = 0
        width = stop - start
        if step == 1:
            return start + int(self.random() * width)
        else:
            num_steps = width // step
            return start + (self.randint(0, num_steps - 1) * step)
        
