import java.util.*;
public class KMP {
  private int n, m;
  private String text, pattern;
  private int[] LPS;
  public KMP(String text, String pattern) {
    this.text = text;
    this.pattern = pattern;
    this.n = text.length();
    this.m = pattern.length();
    this.LPS = new int[m];
    generateLPS();
  }
  private void generateLPS() {
    int len = 0;
    int i = 1;
    while (i < m) {
      if (pattern.charAt(i) == pattern.charAt(len)) {
        LPS[i++] = ++len;
      } else {
        if (len != 0) {
          len = LPS[len - 1];
        } else {
          LPS[i++] = 0;
        }
      }
    }
  }
  public List<int[]> countOccurrences() {
    List<int[]> result = new ArrayList<>();
    int i = 0, j = 0;
    while (i < n) {
      if (text.charAt(i) == pattern.charAt(j)) {
        i++;
        j++;
      }
      if (j == m) {
        int start = i - m;
        int end = i - 1;
        result.add(new int[] { start, end });
        j = LPS[j - 1];
      } else if (i < n && text.charAt(i) != pattern.charAt(j)) {
        if (j != 0) {
          j = LPS[j - 1];
        } else {
          i++;
        }
      }
    }
    return result;
  }
}