// here we are finding the max dist in subTree in down1[node] 
// and second max leaf dist in down2[node], also the up[node]
//= max dist not in the subTree. and the heavy[node] give in
//subTree frim which node the max distance means down1[node] is comming . 
public class UpDownDist {
  public static void main(String[] args) {  }
  static List<List<Integer>> adj;
  static int[] depth, parent, down1, down2, heavy, up;
  static void solve() {
    depth = new int[n + 1];
    down1 = new int[n + 1];
    down2 = new int[n + 1];
    heavy = new int[n + 1];
    up = new int[n + 1];
    parent = new int[n + 1];
    dfsDepth(1, -1);
    dfsDown(1, -1);
    up[1] = 0;
    dfsUp(1, -1);
    long ans = -INF;
    for (int node = 1; node <= n; node++) {
      long curr = k * (long) Math.max(down1[node],
         up[node]) - c * (long) depth[node];
      ans = Math.max(ans, curr);
    }
    out.println(ans);
  }
  static void dfsUp(int node, int par) {
    for (int adjNode : adj.get(node)) {
      if (adjNode == par)
        continue;
      int curr = (heavy[node] == adjNode ? 
        down2[node] : down1[node]);
      up[adjNode] = 1 + Math.max(up[node], curr);
      dfsUp(adjNode, node);
    }
  }
  static void dfsDown(int node, int p) {
    down1[node] = down2[node] = 0;
    heavy[node] = -1;
    for (int adjNode : adj.get(node)) {
      if (adjNode == p)
        continue;
      dfsDown(adjNode, node);
      int curr = 1 + down1[adjNode];
      if (curr > down1[node]) {
        down2[node] = down1[node];
        down1[node] = curr;
        heavy[node] = adjNode;
      } else if (curr > down2[node]) {
        down2[node] = curr;
      }
    }
  }
  static void dfsDepth(int node, int p) {
    parent[node] = p;
    for (int adjNode : adj.get(node)) {
      if (adjNode == p)
        continue;
      depth[adjNode] = 1 + depth[node];
      dfsDepth(adjNode, node);
    }
  }
}