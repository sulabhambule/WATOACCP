
import java.util.*;

public class Utility {
  public static void main(String[] args) {   }
  // If we want to case in which we want small l value and large r value such that
  // we do -L and +R the sort the arr on the basis of li + ri

  // --> We are asked to count the number of non-decreasing sequences of length
  // 2𝑚 where each element is between 1 and n it is same as
  // stars and bars where there are 2m identical object and n boxes so the formula
  // for this is 2m + n - 1 C (n - 1 or 2m)

  // Swapping adjacent elements in a distinct array is basically trying to equate
  // two permutations using adjacent swaps. When is it possible? --> if the parity
  // of inversion in both arrays are same.

  // GCD contains the minimum powers of primes
  // LCM contains the maximum powers of primes

  /*
   * The formula (x+k)+(y+k)=(x+k)⊕(y+k) is equivalent to (x+k)&(y+k)=0, where &
   * denotes the bitwise AND operation.
   * It can be shown that such an non-negative integer k does not exist when x=y.
   * When x≠y, one can show that k=2n−max(x,y) is a possible answer, where 2n is a
   * power of 2 that is sufficiently large.
   * 
   * Important tip : if we do the Xor and we allso take the Xor of the twp numebr
   * then the bit parity never changes
   * means : 1 ^ 1 -> 0 and 1 & 1 = 0. the bit remains the same at that bit.
   */

  // if ax + by = c then Let g = gcd(a, b) then there exists integers x, y such
  // that ax + by = g. Therefore c % g == 0, for the above conditions.

  // we need to find the value of x and y then the formula is (g = gcd(a, b))
  // (only one solution)
  // x => (c / g) * (a / g) ^ -1 * (mod b / g)
  // y => (c - ax) / g.

  /*
   * we have greed and we need to calcullate the sum of some x * y grid, 0 and 1
   * are there in the grid
   * 
   */

  public static int lowerBound(List<Integer> list, int val) {
    int pos = Collections.binarySearch(list, val);
    return (pos >= 0) ? pos : -pos - 1; // First index >= val
  }

  public static int upperBound(List<Integer> list, int val) {
    int pos = Collections.binarySearch(list, val);
    return (pos >= 0) ? pos + 1 : -pos - 1; // First index > val
  }

  public static int floorIndex(List<Integer> list, int val) {
    int pos = Collections.binarySearch(list, val);
    return (pos >= 0) ? pos : -pos - 2; // Last index <= val
  }

  public static int lowerThanIndex(List<Integer> list, int val) {
    int pos = Collections.binarySearch(list, val);
    return (pos >= 0) ? pos - 1 : -pos - 2; // Last index < val
  }

  {
    int[][] prefix = new int[n + 2][m + 2];
    for (int i = 1; i <= n; i++) {
      for (int j = 1; j <= m; j++) {
        int g = (s[i - 1][j - 1] == 1) ? 1 : 0;
        prefix[i][j] = prefix[i - 1][j] + prefix[i][j - 1] - prefix[i - 1][j - 1] + g;
      }
    }
    int totalG = 0;
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < m; j++) {
        totalG += (s[i][j] == 1) ? 1 : 0;
      }
    }
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < m; j++) {
        // checking for the sum of 2k * 2k grid.

        int r1 = Math.max(0, i - k + 1); // top row
        int r2 = Math.min(n, i + k); // bottom row (exclusive)
        int c1 = Math.max(0, j - k + 1); // left col
        int c2 = Math.min(m, j + k); // right col (exclusive)

        // Number of 1s in the rec. (r1, c1) to (r2-1, c2-1)
        int count = prefix[r2][c2] - prefix[r2][c1] - prefix[r1][c2] + prefix[r1][c1];
      }
    }
  }

  // for each possible value x in the array, the minimum prefix length k such that
  // in every prefix of length ≥ k, the value x appears at least once
  {
    int n = 100000;
    int[] a = new int[n + 1];
    int[] gap = new int[n + 1], last = new int[n + 1], ans = new int[n + 1];
    Arrays.fill(ans, -1);

    for (int i = 1; i <= n; i++) {
      int x = a[i];
      gap[x] = Math.max(gap[x], i - last[x]);
      last[x] = i;
    }

    // now we will calculate for each number from 1 to n, what will be the min value
    // of prefix. If lets say for num = 1 min prefix is 3 then for all prfix > 3
    // have ans is 1 because we need to deal with the minimum value that in all the
    // subarray of the lenghr k from 1 to n.

    for (int x = 1; x <= n; x++) {
      gap[x] = Math.max(gap[x], n - last[x] + 1);
      // so for x the max gap is gap[x].
      for (int j = gap[x]; j <= n && ans[j] == -1; j++) {
        ans[j] = x;
      }
    }
  }

  // hashing function for the string.
  long computeHash(String s) {
    final int p = 31;
    final int m = (int) 1e9 + 9;
    long hashValue = 0;
    long pPower = 1;

    for (int i = 0; i < s.length(); i++) {
      char c = s.charAt(i);
      hashValue = (hashValue + (c - 'a' + 1) * pPower) % m;
      pPower = (pPower * p) % m;
    }

    return hashValue;
  }

  public static int countUniqueSubstrings(String s) {
    int n = s.length();
    final int p = 31;
    final int m = (int) 1e9 + 9;

    long[] pPow = new long[n];
    pPow[0] = 1;
    for (int i = 1; i < n; i++) {
      pPow[i] = (pPow[i - 1] * p) % m;
    }

    // Compute prefix hashes
    long[] h = new long[n + 1];
    for (int i = 0; i < n; i++) {
      h[i + 1] = (h[i] + (s.charAt(i) - 'a' + 1) * pPow[i]) % m;
    }

    int count = 0;
    for (int len = 1; len <= n; len++) {
      Set<Long> hashSet = new HashSet<>();
      for (int i = 0; i <= n - len; i++) {
        long curHash = (h[i + len] - h[i] + m) % m;
        curHash = (curHash * pPow[n - i - 1]) % m;
        hashSet.add(curHash);
      }
      count += hashSet.size();
    }

    return count;
  }

  static class Pair {
    int first, second;

    Pair(int first, int second) {
      this.first = first;
      this.second = second;
    }

    @Override
    public boolean equals(Object obj) {
      if (obj == this)
        return true;
      if (!(obj instanceof Pair))
        return false;
      Pair pair = (Pair) obj;
      return pair.first == this.first && pair.second == this.second;
    }

    @Override
    public int hashCode() {
      return Objects.hash(first, second);
    }
  }

  // Funciton that Returns minimum swaps required to sort 
  // arr[] in ascending order
  static int minSwaps(int[] arr) {
    int n = arr.length;
    int[][] paired = new int[n][2];
    for (int i = 0; i < n; i++) {
      paired[i][0] = arr[i];
      paired[i][1] = i;
    }
    Arrays.sort(paired, (a, b) -> Integer.compare(a[0], b[0]));
    boolean[] visited = new boolean[n];
    int swaps = 0;
    for (int i = 0; i < n; i++) {
      if (visited[i] || paired[i][1] == i)
        continue;
      int cycleSize = 0;
      int j = i;
      while (!visited[j]) {
        visited[j] = true;
        j = paired[j][1];
        cycleSize++;
      }
      if (cycleSize > 1)
        swaps += (cycleSize - 1);
    }
    return swaps;
  }
  private static long maxSubarraySum(long[] a, int left, int right) {
    long curr = 0, maxSum = 0;
    for (int i = left; i <= right; i++) {
      curr += a[i];
      maxSum = Math.max(maxSum, curr);
      if (curr < 0) {
        curr = 0;
      }
    }
    return maxSum;
  }
  private static long minSubarraySum(long[] a, int left, int right) {
    long curr = 0, maxSum = 0;
    for (int i = left; i <= right; i++) {
      curr -= a[i];
      maxSum = Math.max(maxSum, curr);
      if (curr < 0) {
        curr = 0;
      }
    }
    return -maxSum;
  }
  private static int lowerBound(long[] a, int start, int end, long val) {
    int lo = start, hi = end, res = end + 1;
    while (lo <= hi) {
      int mid = lo + (hi - lo) / 2;
      if (a[mid] >= val) {
        res = mid;
        hi = mid - 1;
      } else {
        lo = mid + 1;
      }
    }
    return res;
  }
  static long nCr_(int n, int k) {
    if (k > n)
      return 0;
    long numerator = fact[n];
    long denominator = (fact[k] * fact[n - k]) % MOD;
    return (numerator * modInverse(denominator, MOD)) % MOD;
  }
  public static long nCr(int n, int r) {
    if (r > n)
      return 0;
    if (r == 0 || r == n)
      return 1;
    r = Math.min(r, n - r);
    long result = 1;
    for (int i = 0; i < r; i++) {
      result = (result * (n - i)) % MOD;
      result = (modDiv(result, (i + 1), MOD));
    }
    return result;
  }
  static long modInverse(long a, long mod) {
    return modPow(a, mod - 2, mod);
  }
  static long modDiv(long x, long y, long mod) {
    // x * y^(MOD-2) % MOD
    return (x * modPow(y, mod - 2, mod)) % mod;
  }
  static long modPow(long base, long exp, long mod) {
    long result = 1;
    base = base % mod;
    while (exp > 0) {
      if ((exp & 1) == 1) {
        result = (result * base) % mod;
      }
      base = (base * base) % mod;
      exp >>= 1;
    }
    return result;
  }
  static long modMul(long a, long b, long mod) {
    long result = 0;
    a %= mod;
    b %= mod;
    while (b > 0) {
      if ((b & 1) == 1) {
        result = (result + a) % mod;
      }
      a = (a << 1) % mod; // a = (a * 2) % mod
      b >>= 1; // b = b / 2
    }
    return result;
  }
  static long binpow(long a, long b) {
    long res = 1;
    while (b > 0) {
      if ((b & 1) == 1)
        res = res * a;
      a = a * a;
      b >>= 1;
    }
    return res;
  }
  static void derangement() {
    int k = 4;
    int[] derangements = new int[k + 1];
    derangements[0] = 1; // D(0) =
    if (k > 0)
      derangements[1] = 0; // D(1) =
    for (int i = 2; i <= k; i++) {
      derangements[i] = (i - 1) * (derangements[i - 1] + derangements[i - 2]);
    }
  }
  private static void SPF() {
    int N = 100;
    int[] spf = new int[N + 1];
    for (int i = 1; i <= N; i++) {
      spf[i] = i;
    }
    for (int i = 2; i * i <= N; i++) {
      if (spf[i] == i) {
        // this is the prime?
        for (int j = i * i; j <= N; j += i) {
          if (spf[j] == j) {
            // this number is not touched ever.
            spf[j] = i;
          }
        }
      }
    }
  }
  private static void addAllPrimFact(int x, HashMap<Integer, Integer> map) {
    int i = 2;
    while (i * i <= x) {
      while (x % i == 0) {
        map.put(i, map.getOrDefault(i, 0) + 1);
        x /= i;
      }
      i++;
    }
    if (x > 1) {
      map.put(x, map.getOrDefault(x, 0) + 1);
    }
  }

  static boolean[] isPrime;
  static ArrayList<Integer> primes;
  public static void sieve(int n) {
    isPrime = new boolean[n + 1];
    primes = new ArrayList<>();
    Arrays.fill(isPrime, true);
    isPrime[0] = false;
    isPrime[1] = false;
    for (int i = 2; i * i <= n; i++) {
      if (isPrime[i]) {
        for (int j = i * i; j <= n; j += i) {
          isPrime[j] = false;
        }
      }
    }
    for (int i = 2; i <= n; i++) {
      if (isPrime[i]) {
        primes.add(i);
      }
    }
  }

  // Find primes in range
  public static List<Boolean> segmentedSieve(long L, long R) {
    long lim = (long) Math.sqrt(R);
    boolean[] mark = new boolean[(int) (lim + 1)];
    List<Long> primes = new ArrayList<>();
    for (long i = 2; i <= lim; i++) {
      if (!mark[(int) i]) {
        primes.add(i);
        for (long j = i * i; j <= lim; j += i) {
          mark[(int) j] = true;
        }
      }
    }
    List<Boolean> isPrime = new ArrayList<>();
    for (int i = 0; i <= R - L; i++) {
      isPrime.add(true);
    }
    for (long prime : primes) {
      long start = Math.max(prime * prime, (L + prime - 1) / prime * prime);
      for (long j = start; j <= R; j += prime) {
        isPrime.set((int) (j - L), false);
      }
    }
    if (L == 1) {
      isPrime.set(0, false);
    }
    return isPrime;
  }
  public static int countPrimes(int n) {
    final int S = 10000;
    int nsqrt = (int) Math.sqrt(n);
    List<Integer> primes = new ArrayList<>();
    boolean[] isPrime = new boolean[nsqrt + 1];
    Arrays.fill(isPrime, true);
    for (int i = 2; i <= nsqrt; i++) {
      if (isPrime[i]) {
        primes.add(i);
        for (int j = i * i; j <= nsqrt; j += i) {
          isPrime[j] = false;
        }
      }
    }
    int result = 0;
    boolean[] block = new boolean[S];
    for (int k = 0; k * S <= n; k++) {
      Arrays.fill(block, true);
      int start = k * S;
      for (int p : primes) {
        int startIdx = Math.max((start + p - 1) / p, p);
        int j = startIdx * p - start;
        for (; j < S; j += p) {
          block[j] = false;
        }
      }
      if (k == 0) {
        block[0] = block[1] = false;
      }
      for (int i = 0; i < S && start + i <= n; i++) {
        if (block[i]) {
          result++;
        }
      }
    }
    return result;
  }
  // to check in arr[i] the j- th bit set or not.
  // if((arr[i]&(1<<j))!=0) {
  // count++; this means the jth bit is set.increase count
  // }
  // int bit = (num >> i) & 1;

  int flipBit(int n, int j) {
    return n ^ (1 << j);
  } // note: if we add 2^(x-1) to num then num will not divisibe by that x again.

  // mex calculate for the arr of permutation
  // long mex = (n * (n + 1) / 2) - sum;

  private static int computeXOR(int n) {
    if (n % 4 == 0)
      return n;
    if (n % 4 == 1)
      return 1;
    if (n % 4 == 2)
      return n + 1;
    return 0;
  }

  public static int findMSB(long n) {
    int msb = 0;
    while (n > 1) {
      n >>= 1;
      msb++;
    }
    return 1 << msb;
  }
  public static long gcd(long a, long b) {
    if (a == 0)
      return b;
    return gcd(b % a, a);
  }
  public static void factor(long n) {
    long count = 0;
    for (int i = 1; i * i <= n; i++) {
      if (n % i == 0) {
        // i -> is the one factor
        count++;
        if (i != n / i) {
          // n / i -> is the other factor
          count++;
        }
      }
    }
  }
  private static int getPrime(int n) {
    while (n % 2 == 0)
      return 2;
    for (int i = 3; i <= Math.sqrt(n); i += 2) {
      while (n % i == 0)
        return i;
    }
    if (n > 2)
      return n;
    return n;
  }
  public static long MahantaDist(long x1, long y1, long x2, long y2) {
    return Math.abs(x1 - x2) + Math.abs(y1 - y2);
  }

  public static long numberOfDivisors(long num) {
    long total = 1;
    for (long i = 2; i * i <= num; i++) {
      if (num % i == 0) {
        int e = 0;
        while (num % i == 0) {
          e++;
          num /= i;
        }
        total *= (e + 1);
      }
    }
    if (num > 1) {
      total *= 2;
    }
    return total;
  }
  public static long sumOfDivisors(long num) {
    long total = 1;
    for (long i = 2; i * i <= num; i++) {
      if (num % i == 0) {
        int e = 0;
        while (num % i == 0) {
          e++;
          num /= i;
        }
        long sum = 0, pow = 1;
        while (e-- >= 0) {
          sum += pow;
          pow *= i;
        }
        total *= sum;
      }
    }
    if (num > 1) {
      total *= (1 + num);
    }
    return total;
  }
  public static long lcm(long a, long b) {
    return Math.abs(a * b) / gcd(a, b);
  }
  static long nCk(int n, int k) {
    if (k > n || n < 0 || k < 0)
      return 0;
    return (((fact[n] * factInverse[k]) % mod) 
    * factInverse[n - k]) % mod;
  }
  static long combination(long n, long r, long[] fact, long[] ifact) {
    if (r > n || r < 0)
      return 0;
    return ((fact[(int) n] * ifact[(int) r])
     % MOD * ifact[(int) (n - r)] % MOD) % MOD;
  }

  // This is used when we use Pair inside the map
  Map<Pair, Integer> map = new HashMap<>();

  static class Pair {
    long first, second;

    Pair(long first, long second) {
      this.first = first;
      this.second = second;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o)
        return true;
      if (o == null || getClass() != o.getClass())
        return false;
      Pair pair = (Pair) o;
      return first == pair.first && second == pair.second;
    }

    @Override
    public int hashCode() {
      return (int) (31 * first + second);
    }
  }

  // Method to generate the next lexicographical permutation
  public static boolean nextPermutation(char[] array) {
    int n = array.length;
    int i = n - 2;
    while (i >= 0 && array[i] >= array[i + 1]) {
      i--;
    }
    if (i < 0) {
      return false;
    }
    int j = n - 1;
    while (array[j] <= array[i]) {
      j--;
    }
    swap2(array, i, j);
    reverse2(array, i + 1, n - 1);
    return true;
  }
  private static long calculateDigitSum(int n) {
    // to calculate the sum (1 + 2 + .... )
    // (each digit it replaced by there sum of the digit).

    long sum = 0;
    int factor = 1;
    int leftOver = 0;

    // Process each digit position
    while (n > 0) {
      int digit = n % 10;
      int higher = n / 10;
      sum += higher * factor * 45; // Sum of all digits from 0 to 9 is 45
      sum += digit * (digit - 1) / 2 * factor; // Sum of digits within the current group
      sum += digit * leftOver; // Adjust for digits already processed
      leftOver += digit * factor; // Update leftover for next digit position
      factor *= 10;
      n /= 10;
    }
    return sum;
  }

  /*---------------------------------------------------------------------------------------- */

  // TREES
  private static void dfs(int node, List<List<Integer>> edges, int parent, int[] subtreeSize) {
    // subtreeSize[x] = 1 + sum(subtreeSize[child])
    subtreeSize[node] = 1;
    for (int neighbour : edges.get(node)) {
      if (neighbour != parent) {
        dfs(neighbour, edges, node, subtreeSize);
        // subtreeSize of neighbour child is added.
        subtreeSize[node] += subtreeSize[neighbour];
      }
    }
    // once we move out of the dfs call, the subtreeSize of node is correctly
    // populated
  }

  private static void dfs2(int node, List<List<Integer>> edges, int parent, int[] level) {
    if (parent == -1) {
      level[node] = 1;
    } else {
      level[node] = level[parent] + 1;
    }
    for (int neighbour : edges.get(parent))
      if (neighbour != parent)
        dfs(neighbour, edges, node, level);
  }

  /*--------------------------------------------------------------------------------------------- */
  // GRAPH
  static class Pair implements Comparable<Pair> {
    int node, weight;

    Pair(int node, int weight) {
      this.node = node;
      this.weight = weight;
    }

    public int compareTo(Pair other) {
      return this.weight - other.weight;
    }
  }

  static int[] dijkstra(List<List<Pair>> graph, int src, int n) {
    PriorityQueue<Pair> pq = new PriorityQueue<>();
    int[] dist = new int[n];
    Arrays.fill(dist, Integer.MAX_VALUE);
    dist[src] = 0;
    pq.add(new Pair(src, 0));

    while (!pq.isEmpty()) {
      Pair p = pq.poll();
      int u = p.node;
      if (p.weight > dist[u])
        continue;

      for (Pair neighbor : graph.get(u)) {
        int v = neighbor.node;
        int weight = neighbor.weight;
        if (dist[u] + weight < dist[v]) {
          dist[v] = dist[u] + weight;
          pq.add(new Pair(v, dist[v]));
        }
      }
    }
    return dist;
  }

  public static int[] bellmanFord(int n, int[][] edges, int src) {
    int[] dist = new int[n + 1];
    Arrays.fill(dist, (int) 1e9);
    dist[src] = 0;

    // Relax all edges (n - 1) times
    for (int i = 1; i <= n - 1; i++) {
      boolean any = false;
      for (int[] edge : edges) {
        int u = edge[0];
        int v = edge[1];
        int wt = edge[2];
        if (dist[u] != (int) 1e9 && dist[u] + wt < dist[v]) {
          dist[v] = dist[u] + wt;
          any = true;
        }
      }
      if (!any)
        break;
      if (i == n - 1) {
        return new int[] {};
      }
    }
    return dist;
  }
  // static final int INF = 1_000_000_000;
  static void floydWarshall(int[][] dist, int n) {
    for (int k = 0; k < n; k++) {
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
          if (dist[i][k] < INF && dist[k][j] < INF)
            dist[i][j] = Math.min(dist[i][j], dist[i][k] + dist[k][j]);
        }
      }
    }

    for (int i = 0; i < n; i++) {
      if (dist[i][i] < 0) {
        // negative cycle
      }
    }
  }

  // TOPOSORT and all that stuff
  // toposort + cycle detection
  public static boolean dfs(int node, int[] used, List<List<Integer>> adj, List<Integer> ans) {
    used[node] = 1; // in recurtion stack
    for (int adjNode : adj.get(node)) {
      if (used[adjNode] == 1) {
        return false; // detected a cycle
      } else if (used[adjNode] == 0) {
        // not visited
        if (!dfs(adjNode, used, adj, ans)) {
          return false;
        }
      }
    }
    used[node] = 2; // visited but out of stack
    ans.add(node);
    return true;
  }
  // DFS cycle detection (Recomended)
  public static boolean dfsCycleDG(int node, List<List<Integer>> adj,
     boolean[] visited, boolean[] onStack) {
    visited[node] = true;
    onStack[node] = true;
    for (int neighbor : adj.get(node)) {
      if (!visited[neighbor]) {
        if (dfsCycleDG(neighbor, adj, visited, onStack))
          return true;
      } else if (onStack[neighbor]) {
        return true; // Cycle detected
      }
    }
    onStack[node] = false;
    return false;
  }
  // BFS Cycle Detection (Kahn’s Algorithm)
  public static boolean hasCycle(int n, List<List<Integer>> adj) {
    int[] inDegree = new int[n];
    for (int u = 0; u < n; u++) {
      for (int v : adj.get(u))
        inDegree[v]++;
    }
    Queue<Integer> q = new LinkedList<>();
    for (int i = 0; i < n; i++) {
      if (inDegree[i] == 0)
        q.add(i);
    }
    int count = 0;
    while (!q.isEmpty()) {
      int u = q.poll();
      count++;
      for (int v : adj.get(u)) {
        if (--inDegree[v] == 0)
          q.add(v);
      }
    }
    return count != n; // If count < n, there is a cycle
  }
  // DFS-Based Topological Sort
  public static List<Integer> topoSortDfs(int n, List<List<Integer>> adj) {
    boolean[] visited = new boolean[n];
    List<Integer> topo = new ArrayList<>();

    for (int i = 0; i < n; i++) {
      if (!visited[i])
        dfsTopo(i, adj, visited, topo);
    }

    Collections.reverse(topo);
    return topo;
  }

  public static void dfsTopo(int node, List<List<Integer>> adj, boolean[] visited, List<Integer> topo) {
    visited[node] = true;
    for (int neighbor : adj.get(node)) {
      if (!visited[neighbor]) {
        dfsTopo(neighbor, adj, visited, topo);
      }
    }
    topo.add(node);
  }

  public static List<Integer> topoSortBFS(int n, List<List<Integer>> adj) {
    int[] inDegree = new int[n];
    for (int u = 0; u < n; u++) {
      for (int v : adj.get(u)) {
        inDegree[v]++;
      }
    }
    Queue<Integer> q = new LinkedList<>();
    for (int i = 0; i < n; i++) {
      if (inDegree[i] == 0)
        q.add(i);
    }
    List<Integer> topo = new ArrayList<>();
    while (!q.isEmpty()) {
      int u = q.poll();
      topo.add(u);
      for (int v : adj.get(u)) {
        if (--inDegree[v] == 0)
          q.add(v);
      }
    }
    return topo.size() == n ? topo : new ArrayList<>();
    // Return empty if cycle exists
  }
}

// MST using DSU (Krushkal ALgorythm)

public static void main(String[] args) {
  int n;// Nodes
  int m; // Edges
  Edge[] edges = new Edge[m];

  for (int i = 0; i < m; i++) {
    int u = in.nextInt();
    int v = in.nextInt();
    int w = in.nextInt();
    edges[i] = new Edge(u, v, w);
  }

  Arrays.sort(edges); // Sort edges by weight
  DSU dsu = new DSU(n);
  long mstWeight = 0;
  ArrayList<Edge> mstEdges = new ArrayList<>();

  for (Edge e : edges) {
    if (dsu.union(e.u, e.v)) { // If u and v are in different sets
      mstWeight += e.w;
      mstEdges.add(e);
    }
  }
}

static class Edge implements Comparable<Edge> {
  int u, v, w;

  Edge(int u, int v, int w) {
    this.u = u;
    this.v = v;
    this.w = w;
  }

  public int compareTo(Edge o) {
    return Integer.compare(this.w, o.w);
  }
}

// MST using PriorityQueue Prims Algorythm
static long primsMST(int n, List<List<int[]>> adj) {
  boolean[] visited = new boolean[n + 1];
  PriorityQueue<int[]> pq = new PriorityQueue<>((x, y) -> (x[1] - y[1]));
  pq.add(new int[] { 1, 0 }); // Start from node 1
  long mstWeight = 0;
  while (!pq.isEmpty()) {
    int[] curr = pq.poll();
    int u = curr[0], w = curr[1];
    if (visited[u])
      continue;
    visited[u] = true;
    mstWeight += w;
    for (int[] v : adj.get(u)) {
      if (!visited[v[0]]) {
        pq.add(new int[] { v[0], v[1] });
      }
    }
  }
  return mstWeight;
}