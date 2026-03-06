//! Rust/PyO3 implementation of the JSON transform pipeline.
//!
//! Build: cd rust_ext && maturin develop --release
//! Then:  uv run -c "import pipeline_rust; ..."

use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde::Deserialize;
use std::collections::HashMap;

/// Raw event — use borrowed strings (&str) via zero-copy deserialization.
#[derive(Deserialize)]
struct RawEvent<'a> {
    user_id: i64,
    event_type: &'a str,
    timestamp: &'a str,
    metadata: RawMetadata<'a>,
}

/// Only extract the fields we actually need from metadata.
#[derive(Deserialize)]
struct RawMetadata<'a> {
    #[serde(default)]
    amount: Option<f64>,
    #[serde(default)]
    page: Option<&'a str>,
    // Remaining fields are ignored (serde skips them by default with deny_unknown_fields off)
}

/// Per-user aggregation state.
struct UserAgg {
    user_id: i64,
    event_count: i64,
    total_amount: f64,
    high_value_count: i64,
    first_seen: f64,
    last_seen: f64,
    event_types: HashMap<&'static str, i64>,
    pages: HashMap<&'static str, i64>,
}

const FILTER_START_TS: f64 = 1740787200.0;
const FILTER_END_TS: f64 = 1759276799.0;
const HIGH_VALUE_THRESHOLD: f64 = 100.0;

const EPOCH_DAYS: [i64; 60] = {
    let mut table = [0i64; 60];
    let mut d: i64 = 0;
    let mut y: i64 = 1970;
    while y < 2030 {
        table[(y - 1970) as usize] = d;
        if (y % 4 == 0 && y % 100 != 0) || y % 400 == 0 {
            d += 366;
        } else {
            d += 365;
        }
        y += 1;
    }
    table
};

const DAYS_BEFORE_MONTH: [i64; 13] = [0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];

#[inline(always)]
fn parse_iso_to_epoch(iso: &[u8]) -> f64 {
    let digit = |i: usize| -> i64 { (iso[i] - b'0') as i64 };

    let year = digit(0) * 1000 + digit(1) * 100 + digit(2) * 10 + digit(3);
    let month = digit(5) * 10 + digit(6);
    let day = digit(8) * 10 + digit(9);
    let hour = digit(11) * 10 + digit(12);
    let minute = digit(14) * 10 + digit(15);
    let second = digit(17) * 10 + digit(18);

    let mut days =
        EPOCH_DAYS[(year - 1970) as usize] + DAYS_BEFORE_MONTH[month as usize] + day - 1;
    if month > 2 && ((year % 4 == 0 && year % 100 != 0) || year % 400 == 0) {
        days += 1;
    }

    (days * 86400 + hour * 3600 + minute * 60 + second) as f64
}

/// Map event_type str to uppercase. Avoids allocation via static match.
#[inline(always)]
fn event_type_upper(et: &str) -> Option<&'static str> {
    match et {
        "page_view" => Some("PAGE_VIEW"),
        "purchase" => Some("PURCHASE"),
        "click" => Some("CLICK"),
        "search" => Some("SEARCH"),
        "add_to_cart" => Some("ADD_TO_CART"),
        _ => None, // Also serves as the filter — unknown types return None
    }
}

/// Intern page string to static reference. Small fixed set of known pages.
#[inline(always)]
fn intern_page(page: &str) -> &'static str {
    match page {
        "/about" => "/about",
        "/account" => "/account",
        "/blog" => "/blog",
        "/cart" => "/cart",
        "/checkout" => "/checkout",
        "/contact" => "/contact",
        "/home" => "/home",
        "/products" => "/products",
        "/products/123" => "/products/123",
        "/search" => "/search",
        _ => "",
    }
}

fn most_common(map: &HashMap<&'static str, i64>) -> &'static str {
    map.iter()
        .max_by_key(|(_, &v)| v)
        .map(|(k, _)| *k)
        .unwrap_or("")
}

/// Core pipeline in pure Rust.
fn run_pipeline_core(json_bytes: &[u8]) -> Result<(Vec<UserAgg>, i64, f64, i64), String> {
    // Use from_str for zero-copy borrowed &str deserialization
    let json_str =
        std::str::from_utf8(json_bytes).map_err(|e| format!("UTF-8 error: {e}"))?;
    let events: Vec<RawEvent<'_>> =
        serde_json::from_str(json_str).map_err(|e| format!("JSON parse error: {e}"))?;

    let mut users: HashMap<i64, UserAgg> = HashMap::with_capacity(5000);

    for event in &events {
        // Filter + uppercase in one step (None = filtered out)
        let et_upper = match event_type_upper(event.event_type) {
            Some(u) => u,
            None => continue,
        };

        let ts = parse_iso_to_epoch(event.timestamp.as_bytes());
        if ts < FILTER_START_TS || ts > FILTER_END_TS {
            continue;
        }

        let amount = event.metadata.amount.unwrap_or(0.0);
        let page = event.metadata.page.unwrap_or("");

        let user = users.entry(event.user_id).or_insert_with(|| UserAgg {
            user_id: event.user_id,
            event_count: 0,
            total_amount: 0.0,
            high_value_count: 0,
            first_seen: ts,
            last_seen: ts,
            event_types: HashMap::with_capacity(5),
            pages: HashMap::with_capacity(10),
        });

        user.event_count += 1;
        user.total_amount += amount;
        if amount > HIGH_VALUE_THRESHOLD {
            user.high_value_count += 1;
        }
        if ts < user.first_seen {
            user.first_seen = ts;
        }
        if ts > user.last_seen {
            user.last_seen = ts;
        }
        *user
            .event_types
            .entry(et_upper)
            .or_insert(0) += 1;
        *user.pages.entry(intern_page(page)).or_insert(0) += 1;
    }

    let mut sorted_users: Vec<UserAgg> = users.into_values().collect();
    sorted_users.sort_by_key(|u| u.user_id);

    let mut total_events: i64 = 0;
    let mut total_amount: f64 = 0.0;
    let mut total_hv: i64 = 0;
    for u in &sorted_users {
        total_events += u.event_count;
        total_amount += u.total_amount;
        total_hv += u.high_value_count;
    }

    Ok((sorted_users, total_events, total_amount, total_hv))
}

/// Full pipeline returning Python dicts.
#[pyfunction]
fn run_pipeline_from_json(py: Python<'_>, json_bytes: &[u8]) -> PyResult<PyObject> {
    let (sorted_users, total_events, total_amount, total_hv) =
        run_pipeline_core(json_bytes).map_err(pyo3::exceptions::PyValueError::new_err)?;

    let py_users: Vec<PyObject> = sorted_users
        .iter()
        .map(|u| {
            let duration = u.last_seen - u.first_seen;
            let top_event = most_common(&u.event_types);
            let top_page = most_common(&u.pages);

            let dict = PyDict::new(py);
            dict.set_item("user_id", u.user_id).unwrap();
            dict.set_item("event_count", u.event_count).unwrap();
            dict.set_item("total_amount", (u.total_amount * 100.0).round() / 100.0)
                .unwrap();
            dict.set_item("high_value_count", u.high_value_count)
                .unwrap();
            dict.set_item("duration_seconds", (duration * 100.0).round() / 100.0)
                .unwrap();
            dict.set_item("top_event_type", top_event).unwrap();
            dict.set_item("top_page", top_page).unwrap();
            dict.unbind().into()
        })
        .collect();

    let result = PyDict::new(py);
    result.set_item("total_users", sorted_users.len())?;
    result.set_item("total_events", total_events)?;
    result.set_item("total_amount", (total_amount * 100.0).round() / 100.0)?;
    result.set_item("total_high_value", total_hv)?;
    result.set_item("users", py_users)?;

    Ok(result.unbind().into())
}

/// Pipeline returning only summary tuple.
#[pyfunction]
fn run_pipeline_summary(json_bytes: &[u8]) -> PyResult<(usize, i64, f64, i64)> {
    let (sorted_users, total_events, total_amount, total_hv) =
        run_pipeline_core(json_bytes).map_err(pyo3::exceptions::PyValueError::new_err)?;

    Ok((
        sorted_users.len(),
        total_events,
        (total_amount * 100.0).round() / 100.0,
        total_hv,
    ))
}

/// Pipeline from pre-parsed Python list[dict] — same input as Cython/Mypyc.
#[pyfunction]
fn run_pipeline_from_dicts(events: &Bound<'_, pyo3::types::PyList>) -> PyResult<(usize, i64, f64, i64)> {
    use pyo3::types::PyAnyMethods;

    let mut users: HashMap<i64, UserAgg> = HashMap::with_capacity(5000);

    let n = events.len();
    for i in 0..n {
        let event = events.get_item(i)?;
        let event_type: String = event.get_item("event_type")?.extract()?;

        let et_upper = match event_type_upper(&event_type) {
            Some(u) => u,
            None => continue,
        };

        let ts_str: String = event.get_item("timestamp")?.extract()?;
        let ts = parse_iso_to_epoch(ts_str.as_bytes());
        if ts < FILTER_START_TS || ts > FILTER_END_TS {
            continue;
        }

        let uid: i64 = event.get_item("user_id")?.extract()?;
        let meta = event.get_item("metadata")?;

        let amount: f64 = match meta.get_item("amount") {
            Ok(v) => v.extract().unwrap_or(0.0),
            Err(_) => 0.0,
        };
        let page: String = match meta.get_item("page") {
            Ok(v) => v.extract().unwrap_or_default(),
            Err(_) => String::new(),
        };

        let user = users.entry(uid).or_insert_with(|| UserAgg {
            user_id: uid,
            event_count: 0,
            total_amount: 0.0,
            high_value_count: 0,
            first_seen: ts,
            last_seen: ts,
            event_types: HashMap::with_capacity(5),
            pages: HashMap::with_capacity(10),
        });

        user.event_count += 1;
        user.total_amount += amount;
        if amount > HIGH_VALUE_THRESHOLD {
            user.high_value_count += 1;
        }
        if ts < user.first_seen {
            user.first_seen = ts;
        }
        if ts > user.last_seen {
            user.last_seen = ts;
        }
        *user.event_types.entry(et_upper).or_insert(0) += 1;
        *user.pages.entry(intern_page(&page)).or_insert(0) += 1;
    }

    let mut sorted_users: Vec<&UserAgg> = users.values().collect();
    sorted_users.sort_by_key(|u| u.user_id);

    let mut total_events: i64 = 0;
    let mut total_amount: f64 = 0.0;
    let mut total_hv: i64 = 0;
    for u in &sorted_users {
        total_events += u.event_count;
        total_amount += u.total_amount;
        total_hv += u.high_value_count;
    }

    Ok((
        sorted_users.len(),
        total_events,
        (total_amount * 100.0).round() / 100.0,
        total_hv,
    ))
}

/// Benchmark pure Rust: runs N iterations, returns list of times in ms.
/// No Python object construction in the timed section.
#[pyfunction]
fn bench_pure_rust(json_bytes: &[u8], runs: usize) -> PyResult<Vec<f64>> {
    let mut times = Vec::with_capacity(runs);
    for _ in 0..runs {
        let start = std::time::Instant::now();
        let _ = run_pipeline_core(json_bytes);
        let elapsed = start.elapsed();
        times.push(elapsed.as_secs_f64() * 1000.0);
    }
    Ok(times)
}

// ============================================================================
// N-body simulation (Benchmarks Game)
// ============================================================================

const PI: f64 = 3.14159265358979323;
const SOLAR_MASS: f64 = 4.0 * PI * PI;
const DAYS_PER_YEAR: f64 = 365.24;
const NUM_BODIES: usize = 5;
const DT: f64 = 0.01;

struct NBody {
    x: [f64; NUM_BODIES],
    y: [f64; NUM_BODIES],
    z: [f64; NUM_BODIES],
    vx: [f64; NUM_BODIES],
    vy: [f64; NUM_BODIES],
    vz: [f64; NUM_BODIES],
    mass: [f64; NUM_BODIES],
}

impl NBody {
    fn new() -> Self {
        let mut s = NBody {
            // Sun
            x: [0.0, 4.84143144246472090e+00, 8.34336671824457987e+00,
                1.28943695621391310e+01, 1.53796971148509165e+01],
            y: [0.0, -1.16032004402742839e+00, 4.12479856412430479e+00,
                -1.51111514016986312e+01, -2.59193146099879641e+01],
            z: [0.0, -1.03622044471123109e-01, -4.03523417114321381e-01,
                -2.23307578892655734e-01, 1.79258772950371181e-01],
            vx: [0.0,
                 1.66007664274403694e-03 * DAYS_PER_YEAR,
                 -2.76742510726862411e-03 * DAYS_PER_YEAR,
                 2.96460137564761618e-03 * DAYS_PER_YEAR,
                 2.68067772490389322e-03 * DAYS_PER_YEAR],
            vy: [0.0,
                 7.69901118419740425e-03 * DAYS_PER_YEAR,
                 4.99852801234917238e-03 * DAYS_PER_YEAR,
                 2.37847173959480950e-03 * DAYS_PER_YEAR,
                 1.62824170038242295e-03 * DAYS_PER_YEAR],
            vz: [0.0,
                 -6.90460016972063023e-05 * DAYS_PER_YEAR,
                 2.30417297573763929e-05 * DAYS_PER_YEAR,
                 -2.96589568540237556e-05 * DAYS_PER_YEAR,
                 -9.51592254519715870e-05 * DAYS_PER_YEAR],
            mass: [SOLAR_MASS,
                   9.54791938424326609e-04 * SOLAR_MASS,
                   2.85885980666130812e-04 * SOLAR_MASS,
                   4.36624404335156298e-05 * SOLAR_MASS,
                   5.15138902046611451e-05 * SOLAR_MASS],
        };
        // Offset momentum
        let (mut px, mut py, mut pz) = (0.0, 0.0, 0.0);
        for i in 0..NUM_BODIES {
            px -= s.vx[i] * s.mass[i];
            py -= s.vy[i] * s.mass[i];
            pz -= s.vz[i] * s.mass[i];
        }
        s.vx[0] = px / SOLAR_MASS;
        s.vy[0] = py / SOLAR_MASS;
        s.vz[0] = pz / SOLAR_MASS;
        s
    }

    #[inline(never)]
    fn advance(&mut self, n: usize) {
        for _ in 0..n {
            for i in 0..NUM_BODIES {
                for j in (i + 1)..NUM_BODIES {
                    let dx = self.x[i] - self.x[j];
                    let dy = self.y[i] - self.y[j];
                    let dz = self.z[i] - self.z[j];
                    let dsq = dx * dx + dy * dy + dz * dz;
                    let mag = DT / (dsq * dsq.sqrt());
                    let mi = self.mass[i] * mag;
                    let mj = self.mass[j] * mag;
                    self.vx[i] -= dx * mj;
                    self.vy[i] -= dy * mj;
                    self.vz[i] -= dz * mj;
                    self.vx[j] += dx * mi;
                    self.vy[j] += dy * mi;
                    self.vz[j] += dz * mi;
                }
            }
            for i in 0..NUM_BODIES {
                self.x[i] += DT * self.vx[i];
                self.y[i] += DT * self.vy[i];
                self.z[i] += DT * self.vz[i];
            }
        }
    }

    fn energy(&self) -> f64 {
        let mut e = 0.0;
        for i in 0..NUM_BODIES {
            e += self.mass[i]
                * (self.vx[i] * self.vx[i]
                    + self.vy[i] * self.vy[i]
                    + self.vz[i] * self.vz[i])
                / 2.0;
            for j in (i + 1)..NUM_BODIES {
                let dx = self.x[i] - self.x[j];
                let dy = self.y[i] - self.y[j];
                let dz = self.z[i] - self.z[j];
                e -= (self.mass[i] * self.mass[j])
                    / (dx * dx + dy * dy + dz * dz).sqrt();
            }
        }
        e
    }
}

/// Run n-body simulation. Returns (energy_before, energy_after).
#[pyfunction]
fn nbody_benchmark(n: usize) -> PyResult<(f64, f64)> {
    let mut sys = NBody::new();
    let e_before = sys.energy();
    sys.advance(n);
    let e_after = sys.energy();
    Ok((e_before, e_after))
}

// ============================================================================
// Spectral-norm (Benchmarks Game)
// ============================================================================

#[inline(always)]
fn sn_eval_a(i: usize, j: usize) -> f64 {
    let ij = (i + j) as f64;
    1.0 / (ij * (ij + 1.0) / 2.0 + i as f64 + 1.0)
}

fn sn_eval_ata_times_u(n: usize, u: &[f64], out: &mut [f64]) {
    let mut tmp = vec![0.0f64; n];
    // tmp = A * u
    for i in 0..n {
        let mut s = 0.0;
        for j in 0..n {
            s += sn_eval_a(i, j) * u[j];
        }
        tmp[i] = s;
    }
    // out = A^T * tmp
    for i in 0..n {
        let mut s = 0.0;
        for j in 0..n {
            s += sn_eval_a(j, i) * tmp[j];
        }
        out[i] = s;
    }
}

/// Run spectral-norm. Returns the computed norm.
#[pyfunction]
fn spectral_norm_benchmark(n: usize) -> PyResult<f64> {
    let mut u = vec![1.0f64; n];
    let mut v = vec![0.0f64; n];

    for _ in 0..10 {
        sn_eval_ata_times_u(n, &u, &mut v);
        sn_eval_ata_times_u(n, &v, &mut u);
    }

    let mut vbv = 0.0;
    let mut vv = 0.0;
    for i in 0..n {
        vbv += u[i] * v[i];
        vv += v[i] * v[i];
    }
    Ok((vbv / vv).sqrt())
}

#[pymodule]
fn pipeline_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_pipeline_from_json, m)?)?;
    m.add_function(wrap_pyfunction!(run_pipeline_summary, m)?)?;
    m.add_function(wrap_pyfunction!(run_pipeline_from_dicts, m)?)?;
    m.add_function(wrap_pyfunction!(bench_pure_rust, m)?)?;

    m.add_function(wrap_pyfunction!(nbody_benchmark, m)?)?;
    m.add_function(wrap_pyfunction!(spectral_norm_benchmark, m)?)?;
    Ok(())
}
