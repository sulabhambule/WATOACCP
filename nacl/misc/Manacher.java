public class Manacher {
  public static void main(String[] args) {
    String s = "aabaac";
    manacher m = new manacher(s);
    System.out.println(m.getLongestString());
  }
}
class manacher {
  String s, t;
  int[] p;
  public manacher(String s) {
    this.s = s; build();
  }
  public void build() {
    StringBuilder sb = new StringBuilder("#");
    for (char ch : s.toCharArray())
      sb.append(ch).append('#');
    t = sb.toString();
    int n = t.length();
    p = new int[n];
    int l = 0, r = 0;
    for (int i = 0; i < n; i++) {
      int mirror = l + r - i;
      if (i < r)
        p[i] = Math.min(r - i, p[mirror]);
      while (i + p[i] + 1 < n && i - p[i] - 1 >= 0 
        && t.charAt(i + p[i] + 1) == t.charAt(i - p[i] - 1))
        p[i]++;
      if (i + p[i] > r) {
        l = i - p[i];
        r = i + p[i];
      }
    }
  }
  public boolean isPal(int l, int r) {
    int center = l + r + 1;
    int length = r - l + 1;
    return p[center] >= length;
  }
  // Returns the length of the longest palindrome
  public int getLongest() {
    int maxLen = 0;
    for (int x : p)
      if (x > maxLen)
        maxLen = x;
    return maxLen;
  }
  // Returns the actual longest palindromic substring
  public String getLongestString() {
    int maxLen = 0, center = 0;
    for (int i = 0; i < p.length; i++) {
      if (p[i] > maxLen) {
        maxLen = p[i];
        center = i;
      }
    }
    // Map back to original string
    int start = (center - maxLen) / 2;
    return s.substring(start, start + maxLen);
  }
}
