import java.util.Random;

class random {
    static final Random rng = new Random();

    static int randInt(int l, int r) {
        return l + rng.nextInt(r - l + 1);
    }

    static long randLong(long l, long r) {
        return l + (Math.abs(rng.nextLong()) % (r - l + 1));
    }
    // use inside the main 
    // int a = randInt(1, 10);
    // long b = randLong(100, 1000);
}

// ---------- RANDOM (CP TEMPLATE) ----------
// mt19937_64 rng(chrono::steady_clock::now().time_since_epoch().count());

// inline int rnd(int l, int r) {
//     return uniform_int_distribution<int>(l, r)(rng);
// }

// inline long long rndll(long long l, long long r) {
//     return uniform_int_distribution<long long>(l, r)(rng);
// }
