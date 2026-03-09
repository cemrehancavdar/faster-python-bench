"""N-body simulation — Mojo implementation.

Mojo is a compiled language with Python-like syntax. This is a direct
translation of the official Benchmarks Game n-body algorithm.

Attribution:
  Based on the Benchmarks Game Python #1 implementation
  (https://benchmarksgame-team.pages.debian.net/benchmarksgame/)
"""

from math import sqrt


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

comptime NUM_BODIES = 5
comptime DEFAULT_N = 500_000
comptime PI: Float64 = 3.14159265358979323
comptime SOLAR_MASS: Float64 = 4.0 * PI * PI
comptime DAYS_PER_YEAR: Float64 = 365.24
comptime DT: Float64 = 0.01


# ---------------------------------------------------------------------------
# N-body system (struct-based to avoid global vars)
# ---------------------------------------------------------------------------

struct NBodySystem:
    var x: InlineArray[Float64, NUM_BODIES]
    var y: InlineArray[Float64, NUM_BODIES]
    var z: InlineArray[Float64, NUM_BODIES]
    var vx: InlineArray[Float64, NUM_BODIES]
    var vy: InlineArray[Float64, NUM_BODIES]
    var vz: InlineArray[Float64, NUM_BODIES]
    var mass: InlineArray[Float64, NUM_BODIES]

    fn __init__(out self):
        self.x = InlineArray[Float64, NUM_BODIES](fill=0)
        self.y = InlineArray[Float64, NUM_BODIES](fill=0)
        self.z = InlineArray[Float64, NUM_BODIES](fill=0)
        self.vx = InlineArray[Float64, NUM_BODIES](fill=0)
        self.vy = InlineArray[Float64, NUM_BODIES](fill=0)
        self.vz = InlineArray[Float64, NUM_BODIES](fill=0)
        self.mass = InlineArray[Float64, NUM_BODIES](fill=0)

        # Sun
        self.mass[0] = SOLAR_MASS

        # Jupiter
        self.x[1] = 4.84143144246472090e+00
        self.y[1] = -1.16032004402742839e+00
        self.z[1] = -1.03622044471123109e-01
        self.vx[1] = 1.66007664274403694e-03 * DAYS_PER_YEAR
        self.vy[1] = 7.69901118419740425e-03 * DAYS_PER_YEAR
        self.vz[1] = -6.90460016972063023e-05 * DAYS_PER_YEAR
        self.mass[1] = 9.54791938424326609e-04 * SOLAR_MASS

        # Saturn
        self.x[2] = 8.34336671824457987e+00
        self.y[2] = 4.12479856412430479e+00
        self.z[2] = -4.03523417114321381e-01
        self.vx[2] = -2.76742510726862411e-03 * DAYS_PER_YEAR
        self.vy[2] = 4.99852801234917238e-03 * DAYS_PER_YEAR
        self.vz[2] = 2.30417297573763929e-05 * DAYS_PER_YEAR
        self.mass[2] = 2.85885980666130812e-04 * SOLAR_MASS

        # Uranus
        self.x[3] = 1.28943695621391310e+01
        self.y[3] = -1.51111514016986312e+01
        self.z[3] = -2.23307578892655734e-01
        self.vx[3] = 2.96460137564761618e-03 * DAYS_PER_YEAR
        self.vy[3] = 2.37847173959480950e-03 * DAYS_PER_YEAR
        self.vz[3] = -2.96589568540237556e-05 * DAYS_PER_YEAR
        self.mass[3] = 4.36624404335156298e-05 * SOLAR_MASS

        # Neptune
        self.x[4] = 1.53796971148509165e+01
        self.y[4] = -2.59193146099879641e+01
        self.z[4] = 1.79258772950371181e-01
        self.vx[4] = 2.68067772490389322e-03 * DAYS_PER_YEAR
        self.vy[4] = 1.62824170038242295e-03 * DAYS_PER_YEAR
        self.vz[4] = -9.51592254519715870e-05 * DAYS_PER_YEAR
        self.mass[4] = 5.15138902046611451e-05 * SOLAR_MASS

        # Offset momentum
        var px: Float64 = 0.0
        var py: Float64 = 0.0
        var pz: Float64 = 0.0
        for i in range(NUM_BODIES):
            px -= self.vx[i] * self.mass[i]
            py -= self.vy[i] * self.mass[i]
            pz -= self.vz[i] * self.mass[i]
        self.vx[0] = px / SOLAR_MASS
        self.vy[0] = py / SOLAR_MASS
        self.vz[0] = pz / SOLAR_MASS

    fn advance(mut self, n: Int):
        for _ in range(n):
            for i in range(NUM_BODIES):
                for j in range(i + 1, NUM_BODIES):
                    var dx = self.x[i] - self.x[j]
                    var dy = self.y[i] - self.y[j]
                    var dz = self.z[i] - self.z[j]
                    var dsq = dx * dx + dy * dy + dz * dz
                    var mag = DT / (dsq * sqrt(dsq))
                    var mi = self.mass[i] * mag
                    var mj = self.mass[j] * mag
                    self.vx[i] -= dx * mj
                    self.vy[i] -= dy * mj
                    self.vz[i] -= dz * mj
                    self.vx[j] += dx * mi
                    self.vy[j] += dy * mi
                    self.vz[j] += dz * mi
            for i in range(NUM_BODIES):
                self.x[i] += DT * self.vx[i]
                self.y[i] += DT * self.vy[i]
                self.z[i] += DT * self.vz[i]

    fn energy(self) -> Float64:
        var e: Float64 = 0.0
        for i in range(NUM_BODIES):
            e += self.mass[i] * (
                self.vx[i] * self.vx[i]
                + self.vy[i] * self.vy[i]
                + self.vz[i] * self.vz[i]
            ) / 2.0
            for j in range(i + 1, NUM_BODIES):
                var dx = self.x[i] - self.x[j]
                var dy = self.y[i] - self.y[j]
                var dz = self.z[i] - self.z[j]
                e -= (self.mass[i] * self.mass[j]) / sqrt(
                    dx * dx + dy * dy + dz * dz
                )
        return e


fn run_nbody(n: Int = DEFAULT_N) -> List[Float64]:
    """Returns [energy_before, energy_after]."""
    var sys = NBodySystem()
    var e_before = sys.energy()
    sys.advance(n)
    var e_after = sys.energy()
    var result = List[Float64]()
    result.append(e_before)
    result.append(e_after)
    return result^
