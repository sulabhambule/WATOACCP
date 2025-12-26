public class Main {
  static final long MOD = 998244353L;
  static final long INF = (long) 1e18;
  static class Combinatorics {
    final int MOD;
    long[] fact, invFact;
    public Combinatorics(int maxN, int mod) {
      this.MOD = mod;
      fact = new long[maxN + 1];
      invFact = new long[maxN + 1];
      precompute(maxN);
    }
    void precompute(int maxN) {
      fact[0] = 1;
      for (int i = 1; i <= maxN; i++) {
        fact[i] = (i * fact[i - 1]) % MOD;
      }
      invFact[maxN] = modPow(fact[maxN], MOD - 2); // Fermats little theorem
      for (int i = maxN - 1; i >= 0; i--) {
        invFact[i] = (invFact[i + 1] * (i + 1)) % MOD;
      }
    }
    // NCK : no of ways to choose the k elements 
    // from n distinct element wihout caring order.
    long nCk(int n, int k) {
      if (k > n || k < 0)
        return 0;
      return (((fact[n] * invFact[k]) % MOD) * invFact[n - k]) % MOD;
    }
    // NPK : no. of ways to arrange k elements out of n, 
    // where order matters
    long nPk(int n, int k) {
      if (k > n || k < 0)
        return 0;
      return (fact[n] * invFact[n - k]) % MOD;
    }

    long factorial(int n) {
      return fact[n];
    }
    // stars and bars fomula C (n + k - 1, n) --> no. of ways to distribute n
    // identical stars into k bins
    long starsAndBars(int n_stars, int k_bins) {
      if (n_stars == 0)
        return 1;
      if (k_bins == 0)
        return 0;
      return nCk(n_stars + k_bins - 1, n_stars);
    }
    long modPow(long a, long b) {
      long res = 1;
      while (b > 0) {
        if ((b & 1) == 1)
          res = (res * a) % MOD;
        b >>= 1;
        a = (a * a) % MOD;
      }
      return res;
    }
  }
}