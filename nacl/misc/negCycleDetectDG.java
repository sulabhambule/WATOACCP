import java.io.*;
import java.util.*;
public class negCycleDetectDG {
  public static void main(String[] args) throws IOException { }
  static void solve() {
    int[] parent = new int[n + 1];
    long[] dist = new long[n + 1];
    Arrays.fill(dist, INF);
    Arrays.fill(parent, -1);
    dist[1] = 0;
    int startNode = -1;
    // Run Bellman-Ford n times
    for (int i = 0; i < n; i++) {
      startNode = -1;
      for (long[] e : edges) {
        int u = (int) e[0], v = (int) e[1];
        long w = e[2];
        if (dist[u] + w < dist[v]) {
          dist[v] = dist[u] + w;
          parent[v] = u;
          startNode = v;
        }
      }
    }
    if (startNode == -1) {
      out.println("NO");
      return;
    }
    // To ensure we are inside the cycle
    for (int i = 0; i < n; i++) {
      startNode = parent[startNode];
    }
    List<Integer> cycle = new ArrayList<>();
    int v = startNode;
    do {
      cycle.add(v);
      v = parent[v];
    } while (v != startNode);
    cycle.add(startNode);
    Collections.reverse(cycle);
    out.println("YES");
    for (int node : cycle) {
      out.print(node + " ");
    }
    out.println();
  }
}