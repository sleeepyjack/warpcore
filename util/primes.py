import numpy as np
import sys

def primesfrom2to(n):
    """ Input n>=6, Returns a array of primes, 2 <= p < n """
    sieve = np.ones(n//3 + (n%6==2), dtype=np.bool)
    for i in range(1,int(n**0.5)//3+1):
        if sieve[i]:
            k=3*i+1|1
            sieve[       k*k//3     ::2*k] = False
            sieve[k*(k-2*(i&1)+4)//3::2*k] = False
    return np.r_[2,3,((3*np.nonzero(sieve)[0][1:]+1)|1)]

assert(len(sys.argv) == 3)

# generate list of primes up to sys.argv[1]
primes = primesfrom2to(int(sys.argv[1]))
print(len(primes))

filtered_primes = []
j = 1
for i in range(len(primes)-1):
    offset = j * int(sys.argv[2])
    if (primes[i] < offset) and (primes[i+1] >= offset):
        j += 1
        filtered_primes.append(primes[i+1])
filtered_primes = np.array(filtered_primes, dtype=np.int64)
print(len(filtered_primes))

np.savetxt("primes.txt", filtered_primes, delimiter=',\n', fmt='%1u')