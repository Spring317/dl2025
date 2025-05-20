import math
import typing
from typing import Optional, TypeVar, List, Sequence, Any, Union

T = TypeVar('T')  # Generic type for sequences

class PseudoRandom:
    """
    A pseudorandom number generator implementing a linear congruential generator.
    Provides deterministic random number generation based on a seed value.
    """
    
    def __init__(self, seed: Optional[int] = None) -> None:
        """
        Initialize the pseudorandom number generator.
        
        Args:
            seed: Integer seed for reproducible sequences. Uses system time if None.
        """
        # Use system time if no seed is provided
        if seed is None:
            seed = int(sum(ord(c) for c in str(__import__('time').time())))
        
        self.state: int = seed
        self.a: int = 48271
        self.c: int = 0
        self.m: int = 2**31 - 1 
    
    def _next_int(self) -> int:
        """
        Generate the next integer in the sequence.
        
        Returns:
            int: The next pseudorandom integer.
        """
        self.state = (self.a * self.state + self.c) % self.m
        return self.state
    
    def random(self) -> float:
        """
        Return a random float in the range [0.0, 1.0).
        
        Returns:
            float: A pseudorandom float uniformly distributed between 0 and 1.
        """
        return self._next_int() / self.m
    
    def randint(self, a: int, b: int) -> int:
        """
        Return a random integer in the range [a, b].
        
        Args:
            a: Lower bound (inclusive)
            b: Upper bound (inclusive)
            
        Returns:
            int: A pseudorandom integer between a and b (inclusive).
        """
        return a + int(self.random() * (b - a + 1))
    
    def choice(self, seq: Sequence[T]) -> T:
        """
        Return a random element from the sequence.
        
        Args:
            seq: A sequence to choose from
            
        Returns:
            T: A randomly selected element from the sequence
        """
        return seq[self.randint(0, len(seq) - 1)]
    
    def shuffle(self, seq: List[T]) -> List[T]:
        """
        Shuffle the sequence in place using Fisher-Yates algorithm.
        
        Args:
            seq: A list to be shuffled
            
        Returns:
            List[T]: The shuffled list (modified in place)
        """
        for i in range(len(seq) - 1, 0, -1):
            j = self.randint(0, i)
            seq[i], seq[j] = seq[j], seq[i]
        return seq
    
    def randrange(self, start: int, stop: Optional[int] = None, step: int = 1) -> int:
        """
        Return a random integer from range(start, stop, step).
        
        Args:
            start: Start of range or stop if stop is None
            stop: End of range (exclusive)
            step: Step size between values
            
        Returns:
            int: A randomly selected integer from the specified range
        """
        if stop is None:
            stop = start
            start = 0
        width = stop - start
        if step == 1:
            return start + int(self.random() * width)
        else:
            num_steps = width // step
            return start + (self.randint(0, num_steps - 1) * step)
    
    def gauss(self, mu: float = 0, sigma: float = 1) -> float:
        """
        Return a random float from a Gaussian distribution.
        
        Uses the Box-Muller transform to generate normal distribution values.
        
        Args:
            mu: Mean (center) of the distribution
            sigma: Standard deviation (width) of the distribution
            
        Returns:
            float: A pseudorandom float from the specified Gaussian distribution
        """
        u1 = self.random()
        u2 = self.random()
        z0 = (-(2 * math.log(u1))**0.5) * math.cos(2 * math.pi * u2)
        return mu + z0 * sigma