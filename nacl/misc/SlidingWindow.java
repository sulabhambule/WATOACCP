import java.util.*;
// Dequeue Optimization-->

public class SlidingWindow {
  public static void main(String[] args) { }
  // Function to find the minimum in each subarray of size k
  private static List<Integer> sliding_wind_min(int[] arr, int k) {
    List<Integer> ans = new ArrayList<>();
    int n = arr.length;
    Deque<Integer> deque = new LinkedList<>();
    for (int i = 0; i < n; i++) {
      // Remove elements out of the current window
      if (!deque.isEmpty() && deque.getFirst() < i - k + 1) {
        deque.pollFirst();
      }
      // Remove elements from the deque that are 
      // greater than or equal to the current
      // element
      while (!deque.isEmpty() && arr[deque.getLast()] >= arr[i]) {
        deque.pollLast();
      }
      // Add the current element index to the deque
      deque.offerLast(i);
      // Once the first window is fully traversed,
      //  start adding results
      if (i >= k - 1) 
        ans.add(arr[deque.getFirst()]);
    }
    return ans;
  }

  // code to find the sliding window maximum of size k.
  public int[] maxSlidingWindow(int[] nums, int k) {
    int n = nums.length;
    int[] ans = new int[n + 1 - k];
    TreeMap<Integer, Integer> map = new TreeMap<>();
    int l = 0;
    for (int r = 0; r < n; r++) {
      map.put(nums[r], map.getOrDefault(nums[r], 0) + 1);
      if (r - l + 1 == k) {
        ans[l] = map.lastKey();
        int val = nums[l];
        if (map.get(val) == 1) {
          map.remove(val);
        } else {
          map.put(val, map.get(val) - 1);
        }
        l++;
      }
    }
    return ans;
  }
  // max num is sliding window of size k. 

  public int[] maxSlidingWindow2(int[] nums, int k) {
    int n = nums.length;
    int[] ans = new int[n - k + 1];
    int idx = 0;
    Deque<Integer> deque = new LinkedList<>();
    for (int i = 0; i < n; i++) {
      if (!deque.isEmpty() && deque.getFirst() < i - k + 1) {
        deque.pollFirst();
      }
      while (!deque.isEmpty() && nums[deque.getLast()] <= nums[i]) {
        deque.pollLast();
      }
      deque.offerLast(i);
      if (i >= k - 1) {
        ans[idx++] = nums[deque.getFirst()];
      }
    }
    return ans;
  }
  // Function to find the sliding window Meadian.
  public double[] medianSlidingWindow(int[] nums, int k) {
    TreeSet<Integer> minSet = new TreeSet<>(
        (a, b) -> nums[a] == nums[b] ? a - b
            : Integer.compare(nums[a], nums[b]));
    TreeSet<Integer> maxSet = new TreeSet<>(
        (a, b) -> nums[a] == nums[b] ? a - b
            : Integer.compare(nums[a], nums[b]));

    double[] ans = new double[nums.length - k + 1];

    for (int i = 0; i < nums.length; i++) {
      minSet.add(i); // add the index in the low
      maxSet.add(minSet.pollLast()); 
      // add the last of minSet to max.
      if (minSet.size() < maxSet.size()) {
        // if low < high add the first from the high to the low set.
        minSet.add(maxSet.pollFirst());
      }
      if (i >= k - 1) {
        if (k % 2 == 0) {
          ans[i - k + 1] = ((double) nums[minSet.last()]
              + nums[maxSet.first()]) / 2;
        } else {
          ans[i - k + 1] = (double) nums[minSet.last()];
        }
        if (!minSet.remove(i - k + 1)) 
          maxSet.remove(i - k + 1);
      }
    }
    return ans;
  }
}
