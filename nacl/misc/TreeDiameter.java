import java.io.*;
import java.util.*;
public class TreeDiameter {
    public static void main(String[] args) {
        solve();// out.close();
    }
    private static void solve() {
        int n = in.nextInt();
        List<List<Integer>> edges = new ArrayList<>();
        for (int i = 0; i <= n; i++) {
            edges.add(new ArrayList<>());
        }
        for (int i = 0; i < n - 1; i++) {
            int u = in.nextInt();
            int v = in.nextInt();
            edges.get(u).add(v);
            edges.get(v).add(u);
        }
        int[] distX = new int[n + 1];
        int[] distY = new int[n + 1];
        Arrays.fill(distX, -1);
        Arrays.fill(distY, -1);
        int x = 1;
        // First DFS from a random node to find a 
        // farthest node
        dfs(x, edges, -1, distX);
        int y = farthestNode(n, distX);
        // Second DFS from farthest node to 
        // find the farthest node from it
        dfs(y, edges, -1, distY);
        int z = farthestNode(n, distY);
        // Print the diameter of the tree
        System.out.println(distY[z]);
    }
    private static void dfs(int curr, List<List<Integer>> edges, int parent, int[] level) {
        if (parent == -1) {
            level[curr] = 0;
        } else {
            level[curr] = level[parent] + 1;
        }
        for (int neighbor : edges.get(curr)) {
            if (neighbor != parent) {
                dfs(neighbor, edges, curr, level);
            }
        }
    }
    // Find the farthest node from a given node
    private static int farthestNode(int n, int[] dist) {
        int farthest = 0;
        for (int i = 0; i <= n; i++) {
            if (dist[i] > dist[farthest]) {
                farthest = i;
            }
        }
        return farthest;
    }
}
