public class Utility {
  public static void main(String[] args) {   }
  { int[][] prefix = new int[n + 2][m + 2];
    for (int i = 1; i <= n; i++) {
      for (int j = 1; j <= m; j++) {
        int g = (s[i - 1][j - 1] == 1) ? 1 : 0;
        prefix[i][j] = prefix[i - 1][j] + 
        prefix[i][j - 1] - prefix[i - 1][j - 1] + g;
      }
    }
    int totalG = 0;
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < m; j++) {
        totalG += (s[i][j] == 1) ? 1 : 0;   }
    }
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < m; j++) {
        // checking for the sum of 2k * 2k grid.
        int r1 = Math.max(0, i - k + 1); // top row
        int r2 = Math.min(n, i + k); // bottom row (exclusive)
        int c1 = Math.max(0, j - k + 1); // left col
        int c2 = Math.min(m, j + k); // right col (exclusive)
        // Number of 1s in the rec. (r1, c1) to (r2-1, c2-1)
        int count = prefix[r2][c2] - 
        prefix[r2][c1] - prefix[r1][c2] + prefix[r1][c1];
      }
    }
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
  // Returns min. swaps required to sort arr[] in asc order
  static int minSwaps(int[] arr) {
    int n = arr.length;
    int[][] paired = new int[n][2];
    for (int i = 0; i < n; i++) {
      paired[i][0] = arr[i]; paired[i][1] = i;
    }
    Arrays.sort(paired, (a, b) -> Integer.compare(a[0], b[0]));
    boolean[] visited = new boolean[n];
    int swaps = 0;
    for (int i = 0; i < n; i++) {
      if (visited[i] || paired[i][1] == i) continue;
      int cycleSize = 0,j = i;
      while (!visited[j]) {
        visited[j] = true;
        j = paired[j][1]; cycleSize++;
      }
      if (cycleSize > 1) swaps += (cycleSize - 1);
    }
    return swaps;
  }
  private static long maxSubarraySum(long[] a, int left, int right) {
    long curr = 0, maxSum = 0;
    for (int i = left; i <= right; i++) {
      curr += a[i];
      maxSum = Math.max(maxSum, curr);
      if (curr < 0) curr = 0;
    }
    return maxSum;
  }
  private static long minSubarraySum(long[] a, int left, int right) {
    long curr = 0, maxSum = 0;
    for (int i = left; i <= right; i++) {
      curr -= a[i];
      maxSum = Math.max(maxSum, curr);
      if (curr < 0) curr = 0;
    }
    return -maxSum;
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
      if ((exp & 1) == 1)
        result = (result * base) % mod;
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
  static void derangement() {
    int k = 4;
    int[] derangements = new int[k + 1];
    derangements[0] = 1; // D(0) =
    if (k > 0)
      derangements[1] = 0; // D(1) =
    for (int i = 2; i <= k; i++) {
      derangements[i] = (i - 1) * 
      (derangements[i - 1] + derangements[i - 2]);
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
    if (x > 1) map.put(x, map.getOrDefault(x, 0) + 1);
  } // Find primes in range
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
    if (L == 1)
      isPrime.set(0, false);
    return isPrime;
  }
  // int bit = (num >> i) & 1;
  int flipBit(int n, int j) {
    return n ^ (1 << j);
  }
  // mex calculate for the arr of permutation
  // long mex = (n * (n + 1) / 2) - sum;
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
  static int[] dijkstra(List<List<Pair>> graph, int src, int n) {
    PriorityQueue<Pair> pq = new PriorityQueue<>();
    int[] dist = new int[n];
    Arrays.fill(dist, Integer.MAX_VALUE);
    dist[src] = 0;
    pq.add(new Pair(src, 0));
    while (!pq.isEmpty()) {
      Pair p = pq.poll();
      int u = p.node;
      if (p.weight > dist[u]) continue;
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
        int u = edge[0],v = edge[1], wt = edge[2];
        if (dist[u] != (int) 1e9 && dist[u] + wt < dist[v]) {
          dist[v] = dist[u] + wt;
          any = true;
        }
      }
      if (!any) break;
      if (i == n - 1) return new int[] {};
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
  // TOPOSORT and all that stuff toposort + cycle detection
  public static boolean dfs(int node, int[] used, List<List<Integer>> adj, List<Integer> ans) {
    used[node] = 1; // in recurtion stack
    for (int adjNode : adj.get(node)) {
      if (used[adjNode] == 1) {
        return false; // detected a cycle
      } else if (used[adjNode] == 0) {
        // not visited
        if (!dfs(adjNode, used, adj, ans))
          return false;
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
    } return count != n; // If count < n, there is a cycle
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
  public static void dfsTopo(int node, List<List<Integer>> adj,
     boolean[] visited, List<Integer> topo) {
    visited[node] = true;
    for (int neighbor : adj.get(node)) {
      if (!visited[neighbor])
        dfsTopo(neighbor, adj, visited, topo);
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
      if (inDegree[i] == 0) q.add(i);
    }
    List<Integer> topo = new ArrayList<>();
    while (!q.isEmpty()) {
      int u = q.poll();
      topo.add(u);
      for (int v : adj.get(u)) {
        if (--inDegree[v] == 0) q.add(v);
      }
    }
    return topo.size() == n ? topo : new ArrayList<>();
  }
} // MST using DSU (Krushkal ALgorythm)
  int n;// Nodes int m; // Edges
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
    this.u = u; this.v = v; this.w = w;
  }
  public int compareTo(Edge o) {
    return Integer.compare(this.w, o.w);
  }
} // MST using PriorityQueue Prims Algorythm
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