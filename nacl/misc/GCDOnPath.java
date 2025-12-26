import java.util.*;
public class GCDOnPath {
  static final int MAX_LOG = 20;
  static final int N = (int) 2e5 + 1;
  static int[][] parent = new int[N][MAX_LOG];
  static int[][] gcdVal = new int[N][MAX_LOG];
  static int[] depth = new int[N];
  static int[] arr = new int[N];
  static List<List<Integer>> adj;

  public static void main(String[] args) { }

  private static void solve() {
    dfs(1, 0);
  }
  private static void dfs(int node, int par) {
    depth[node] = depth[par] + 1;
    parent[node][0] = par;
    gcdVal[node][0] = gcd(arr[node], arr[par]);
    for (int j = 1; j < MAX_LOG; j++) {
      int midParent = parent[node][j - 1];
      parent[node][j] = parent[midParent][j - 1];
      gcdVal[node][j] = gcd(gcdVal[node][j - 1], 
        gcdVal[midParent][j - 1]);
    }
    for (int child : adj.get(node)) {
      if (child != par) {
        dfs(child, node);
      }
    }
  }
  private static int getGCDOnPath(int a, int b) {
    int g = gcd(arr[a], arr[b]);
    if (depth[a] < depth[b]) {
      int temp = a;
      a = b; b = temp;
    }
    int diff = depth[a] - depth[b];
    for (int j = MAX_LOG - 1; j >= 0; j--) {
      if (((1 << j) & diff) != 0) {
        g = gcd(g, gcdVal[a][j]);
        a = parent[a][j];
      }
    }

    if (a == b)
      return g;
    for (int j = MAX_LOG - 1; j >= 0; j--) {
      if (parent[a][j] != parent[b][j]) {
        g = gcd(g, gcd(gcdVal[a][j], gcdVal[b][j]));
        a = parent[a][j];
        b = parent[b][j];
      }
    }
    g = gcd(g, gcd(gcdVal[a][0], gcdVal[b][0]));
    return g;
  }
  private static int gcd(int a, int b) {
    return b == 0 ? a : gcd(b, a % b);
  }
}