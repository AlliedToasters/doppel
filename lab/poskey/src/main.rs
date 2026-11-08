//! poskey: Polyglot Zobrist position keys and a Bloom-fronted position counter (CPU only).
//!
//! Key: the standard 64-bit Polyglot Zobrist hash of the position after each ply (pieces, side to move,
//! castling rights, en passant file only when a pawn of the side to move stands next to the double-pushed
//! pawn). Equals python-chess `chess.polyglot.zobrist_hash`, so any client can look positions up.
//!
//! Counter: the key space is split into 1024 shards by the top 10 bits. Each shard holds a slice of one Bloom
//! filter and an exact hash map, behind one lock. The first sighting of a key only sets Bloom bits. A key the
//! Bloom filter has already seen goes into the exact map at count 2, then increments. A Bloom false positive
//! makes a key's count one too high (or puts a true singleton in the table at 2); `--exact` skips the Bloom
//! filter and counts every key in the map, to measure that.
//!
//! Probes: positions (and (position, move) edges) of evaluation games get exact counts in the same pass,
//! at 9 nested subsample levels (game in level j with probability 2^-j), for the depth-vs-corpus curve.
//!
//!     poskey keys < lines            # "FEN|uci uci ..." per line (empty FEN = start) -> hex keys per line
//!     poskey count --out DIR [--probe eval.parquet] train/*.parquet

use std::fs::File;
use std::io::{self, BufRead, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use arrow_array::{Array, Int64Array, LargeListArray, LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field};
use clap::{Parser, Subcommand};
use hashbrown::HashMap;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ProjectionMask;
use rayon::prelude::*;
use shakmaty::fen::Fen;
use shakmaty::uci::UciMove;
use shakmaty::zobrist::{Zobrist64, ZobristHash};
use shakmaty::{CastlingMode, Chess, EnPassantMode, Position};

const LEVELS: usize = 9;
const SHARD_BITS: u32 = 10;
const SHARDS: usize = 1 << SHARD_BITS;
const CHUNK: usize = 1024; // games per work unit

// --- hashing ------------------------------------------------------------------------------------------------

#[inline]
fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

#[inline]
fn fnv(b: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &c in b {
        h ^= c as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

/// Polyglot key. PseudoLegal en passant = python-chess / Polyglot rule (a capturing pawn is adjacent; pins ignored).
#[inline]
fn poly(pos: &Chess, ep: EnPassantMode) -> u64 {
    pos.zobrist_hash::<Zobrist64>(ep).0
}

/// (position, move) edge key: the parent position's Polyglot key mixed with the UCI move string.
#[inline]
fn edge_key(parent: u64, uci: &[u8]) -> u64 {
    splitmix(parent ^ splitmix(fnv(uci)))
}

/// Replay up to `cap` UCI moves from the start position; f(ply, position key, edge key) after each.
/// Stops at the first unparsable or illegal move. Returns plies replayed.
fn replay<'a, I, F>(moves: I, cap: usize, mut f: F) -> usize
where
    I: Iterator<Item = &'a [u8]>,
    F: FnMut(usize, u64, u64),
{
    let mut pos = Chess::default();
    let mut parent = poly(&pos, EnPassantMode::PseudoLegal);
    let mut n = 0;
    for (i, mv) in moves.take(cap).enumerate() {
        let Ok(u) = UciMove::from_ascii(mv) else { break };
        let Ok(m) = u.to_move(&pos) else { break };
        pos.play_unchecked(&m);
        let k = poly(&pos, EnPassantMode::PseudoLegal);
        f(i, k, edge_key(parent, mv));
        parent = k;
        n = i + 1;
    }
    n
}

/// Hasher for keys that are already random: one multiply spreads low bits into the top bits hashbrown uses
/// (all keys in a shard share their own top 10 bits).
#[derive(Default, Clone, Copy)]
struct KeyHasher(u64);
impl std::hash::Hasher for KeyHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, _: &[u8]) {
        unreachable!("u64 keys only")
    }
    fn write_u64(&mut self, x: u64) {
        self.0 = (x ^ (x >> 29)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
}
type BH = std::hash::BuildHasherDefault<KeyHasher>;
type Map<V> = HashMap<u64, V, BH>;

// --- Bloom + exact shards -----------------------------------------------------------------------------------

struct Shard {
    bloom: Vec<u64>,
    map: Map<u32>,
}

struct Counter {
    shards: Vec<Mutex<Shard>>,
    nbits: u64, // per shard
    k: u32,
    exact: bool,
}

impl Counter {
    fn new(n_expected: f64, fpr: f64, exact: bool) -> Counter {
        let ln2 = std::f64::consts::LN_2;
        let m = (-n_expected * fpr.ln() / (ln2 * ln2)).ceil();
        let k = ((m / n_expected) * ln2).round().max(1.0) as u32;
        let words = if exact { 0 } else { ((m / SHARDS as f64) / 64.0).ceil() as usize };
        let shards = (0..SHARDS)
            .map(|_| Mutex::new(Shard { bloom: vec![0u64; words], map: Map::default() }))
            .collect();
        Counter { shards, nbits: words as u64 * 64, k, exact }
    }

    fn bloom_bytes(&self) -> u64 {
        self.nbits / 8 * SHARDS as u64
    }

    /// Insert one shard's keys; returns, per key, whether it was a first sighting (via `first`).
    fn insert(&self, s: usize, keys: &[(u64, u8)], first: &mut [u64]) {
        let mut g = self.shards[s].lock().unwrap();
        let sh = &mut *g;
        for &(key, ply) in keys {
            if self.exact {
                let c = sh.map.entry(key).or_insert(0);
                if *c == 0 {
                    first[ply as usize] += 1;
                }
                *c += 1;
            } else if let Some(c) = sh.map.get_mut(&key) {
                *c += 1;
            } else if bloom_test_set(&mut sh.bloom, self.nbits, self.k, key) {
                sh.map.insert(key, 2);
            } else {
                first[ply as usize] += 1;
            }
        }
    }
}

/// Set the key's k bits; true if all were already set ("seen").
#[inline]
fn bloom_test_set(bits: &mut [u64], nbits: u64, k: u32, key: u64) -> bool {
    let h1 = splitmix(key ^ 0x5bd1_e995_7f4a_7c15);
    let h2 = splitmix(h1) | 1;
    let mut seen = true;
    for i in 0..k as u64 {
        let g = h1.wrapping_add(i.wrapping_mul(h2));
        let idx = ((g as u128 * nbits as u128) >> 64) as usize;
        let (w, b) = (idx >> 6, 1u64 << (idx & 63));
        if bits[w] & b == 0 {
            seen = false;
            bits[w] |= b;
        }
    }
    seen
}

// --- probes (evaluation games: exact counts at every subsample level) -------------------------------------

struct Probe {
    pos_idx: Map<u32>,
    edge_idx: Map<u32>,
    pos_cnt: Vec<AtomicU32>,
    edge_cnt: Vec<AtomicU32>,
    plies: Vec<(u64, u32, u32)>, // per probe ply: position key, position uid, edge uid
    lens: Vec<i64>,
}

impl Probe {
    fn load(path: &Path, cap: usize) -> io::Result<Probe> {
        let (mut pos_idx, mut edge_idx) = (Map::<u32>::default(), Map::<u32>::default());
        let (mut plies, mut lens) = (Vec::new(), Vec::new());
        read_parquet(path, false, |b| {
            for i in 0..b.len() {
                let n = replay(b.moves(i), cap, |_, k, e| {
                    let np = pos_idx.len() as u32;
                    let u = *pos_idx.entry(k).or_insert(np);
                    let ne = edge_idx.len() as u32;
                    let v = *edge_idx.entry(e).or_insert(ne);
                    plies.push((k, u, v));
                });
                lens.push(n as i64);
            }
        })?;
        let z = |n: usize| (0..n * LEVELS).map(|_| AtomicU32::new(0)).collect();
        let (pc, ec) = (z(pos_idx.len()), z(edge_idx.len()));
        Ok(Probe { pos_idx, edge_idx, pos_cnt: pc, edge_cnt: ec, plies, lens })
    }

    #[inline]
    fn hit(&self, k: u64, e: u64, level: usize) {
        if let Some(&u) = self.pos_idx.get(&k) {
            for j in 0..=level {
                self.pos_cnt[u as usize * LEVELS + j].fetch_add(1, Relaxed);
            }
        }
        if let Some(&u) = self.edge_idx.get(&e) {
            for j in 0..=level {
                self.edge_cnt[u as usize * LEVELS + j].fetch_add(1, Relaxed);
            }
        }
    }

    fn write(&self, out: &Path) -> io::Result<()> {
        let expand = |cnt: &Vec<AtomicU32>, pick: fn(&(u64, u32, u32)) -> u32| {
            let mut v = Vec::with_capacity(self.plies.len() * LEVELS);
            for p in &self.plies {
                let u = pick(p) as usize;
                for j in 0..LEVELS {
                    v.push(cnt[u * LEVELS + j].load(Relaxed));
                }
            }
            v
        };
        let n = self.plies.len();
        write_npy(&out.join("probe_counts.npy"), "<u4", &[n, LEVELS], &expand(&self.pos_cnt, |p| p.1))?;
        write_npy(&out.join("probe_edge_counts.npy"), "<u4", &[n, LEVELS], &expand(&self.edge_cnt, |p| p.2))?;
        write_npy(&out.join("probe_keys.npy"), "<u8", &[n], &self.plies.iter().map(|p| p.0).collect::<Vec<_>>())?;
        write_npy(&out.join("probe_lens.npy"), "<i8", &[self.lens.len()], &self.lens)
    }
}

// --- parquet input --------------------------------------------------------------------------------------------

struct Batch {
    offs: Vec<i64>,
    vals: LargeStringArray,
    band: Option<Int64Array>,
    ids: Option<LargeStringArray>,
}

impl Batch {
    fn len(&self) -> usize {
        self.offs.len() - 1
    }
    fn moves(&self, i: usize) -> impl Iterator<Item = &[u8]> + '_ {
        (self.offs[i] as usize..self.offs[i + 1] as usize).map(move |j| self.vals.value(j).as_bytes())
    }
    fn band(&self, i: usize) -> Option<i64> {
        self.band.as_ref().filter(|b| !b.is_null(i)).map(|b| b.value(i))
    }
}

fn read_parquet(path: &Path, meta_cols: bool, mut f: impl FnMut(&Batch)) -> io::Result<()> {
    let file = File::open(path)?;
    let b = ParquetRecordBatchReaderBuilder::try_new(file).map_err(io::Error::other)?;
    let schema = b.parquet_schema();
    let want: Vec<usize> = (0..schema.num_columns())
        .filter(|&i| {
            let col = schema.column(i);
            let n = col.path().parts()[0].as_str();
            n == "moves" || (meta_cols && (n == "band" || n == "game_id"))
        })
        .collect();
    let mask = ProjectionMask::leaves(schema, want);
    let reader = b.with_projection(mask).with_batch_size(65536).build().map_err(io::Error::other)?;
    let list_t = DataType::LargeList(Arc::new(Field::new("item", DataType::LargeUtf8, true)));
    for rb in reader {
        let rb: RecordBatch = rb.map_err(io::Error::other)?;
        let cast = |name: &str, t: &DataType| {
            rb.column_by_name(name).map(|c| arrow_cast::cast(c, t).expect("cast"))
        };
        let m = cast("moves", &list_t).expect("no moves column");
        let m = m.as_any().downcast_ref::<LargeListArray>().unwrap();
        let vals = m.values().as_any().downcast_ref::<LargeStringArray>().unwrap().clone();
        // a null list has equal start and end offsets (pyarrow and polars writers), i.e. an empty game
        let offs = m.value_offsets().to_vec();
        let band = cast("band", &DataType::Int64).map(|c| c.as_any().downcast_ref::<Int64Array>().unwrap().clone());
        let ids = cast("game_id", &DataType::LargeUtf8)
            .map(|c| c.as_any().downcast_ref::<LargeStringArray>().unwrap().clone());
        f(&Batch { offs, vals, band, ids });
    }
    Ok(())
}

fn parquet_rows(path: &Path) -> u64 {
    File::open(path)
        .ok()
        .and_then(|f| ParquetRecordBatchReaderBuilder::try_new(f).ok())
        .map(|b| b.metadata().file_metadata().num_rows() as u64)
        .unwrap_or(0)
}

// --- output ---------------------------------------------------------------------------------------------------

trait Le: Copy {
    fn put(self, w: &mut impl Write) -> io::Result<()>;
}
impl Le for u64 {
    fn put(self, w: &mut impl Write) -> io::Result<()> {
        w.write_all(&self.to_le_bytes())
    }
}
impl Le for u32 {
    fn put(self, w: &mut impl Write) -> io::Result<()> {
        w.write_all(&self.to_le_bytes())
    }
}
impl Le for i64 {
    fn put(self, w: &mut impl Write) -> io::Result<()> {
        w.write_all(&self.to_le_bytes())
    }
}

fn write_npy<T: Le>(path: &Path, descr: &str, shape: &[usize], data: &[T]) -> io::Result<()> {
    let shp = match shape {
        [n] => format!("({n},)"),
        s => format!("({})", s.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", ")),
    };
    let mut h = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shp}, }}");
    while (10 + h.len() + 1) % 64 != 0 {
        h.push(' ');
    }
    h.push('\n');
    let mut w = BufWriter::with_capacity(1 << 22, File::create(path)?);
    w.write_all(b"\x93NUMPY\x01\x00")?;
    w.write_all(&(h.len() as u16).to_le_bytes())?;
    w.write_all(h.as_bytes())?;
    for &x in data {
        x.put(&mut w)?;
    }
    w.flush()
}

fn peak_rss_gb() -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| s.lines().find(|l| l.starts_with("VmHWM:")).map(|l| l.to_string()))
        .and_then(|l| l.split_whitespace().nth(1).and_then(|x| x.parse::<f64>().ok()))
        .map(|kb| kb / 1e6 * 1.024)
        .unwrap_or(0.0)
}

fn json_arr<T: ToString>(v: &[T]) -> String {
    format!("[{}]", v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "))
}

// --- commands -------------------------------------------------------------------------------------------------

#[derive(Parser)]
#[command(about = "Polyglot Zobrist keys and a Bloom-fronted position counter")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// stdin lines "FEN|uci uci ..." (empty FEN = start position) -> hex keys: start, then after each move
    Keys {
        /// en passant rule: pseudo (Polyglot / python-chess) or legal
        #[arg(long, default_value = "pseudo")]
        ep: String,
    },
    /// Count position keys over parquet files (a `moves` list-of-UCI column; optional `band`, `game_id`)
    Count {
        #[arg(long)]
        out: PathBuf,
        /// Bloom false-positive target at --expected distinct keys
        #[arg(long, default_value_t = 0.01)]
        fpr: f64,
        /// expected distinct keys (default: 49 per input game, the 2026-10 move-order rate)
        #[arg(long)]
        expected: Option<f64>,
        /// no Bloom filter: every key goes into the exact map (validation; RAM grows with distinct keys)
        #[arg(long)]
        exact: bool,
        #[arg(long, default_value_t = 80)]
        cap: usize,
        #[arg(long, default_value_t = 26)]
        threads: usize,
        /// only count games with band in [band_min, band_max] (band-specific tables)
        #[arg(long)]
        band_min: Option<i64>,
        #[arg(long)]
        band_max: Option<i64>,
        /// evaluation games (parquet with `moves`): exact per-ply counts at every subsample level
        #[arg(long)]
        probe: Option<PathBuf>,
        files: Vec<PathBuf>,
    },
}

fn keys_cmd(ep: &str) -> io::Result<()> {
    let ep = if ep == "legal" { EnPassantMode::Legal } else { EnPassantMode::PseudoLegal };
    let out = io::stdout();
    let mut w = BufWriter::new(out.lock());
    for line in io::stdin().lock().lines() {
        let line = line?;
        let (fen, mv) = line.split_once('|').unwrap_or((line.as_str(), ""));
        let mut pos: Chess = if fen.trim().is_empty() {
            Chess::default()
        } else {
            Fen::from_ascii(fen.trim().as_bytes())
                .map_err(io::Error::other)?
                .into_position(CastlingMode::Standard)
                .map_err(io::Error::other)?
        };
        let mut ks = vec![format!("{:016x}", poly(&pos, ep))];
        for u in mv.split_whitespace() {
            let m = UciMove::from_ascii(u.as_bytes()).map_err(io::Error::other)?.to_move(&pos).map_err(io::Error::other)?;
            pos.play_unchecked(&m);
            ks.push(format!("{:016x}", poly(&pos, ep)));
        }
        writeln!(w, "{}", ks.join(" "))?;
    }
    w.flush()
}

#[allow(clippy::too_many_arguments)]
fn count_cmd(
    out: &Path, fpr: f64, expected: Option<f64>, exact: bool, cap: usize, band: (Option<i64>, Option<i64>),
    probe: Option<&Path>, files: &[PathBuf],
) -> io::Result<()> {
    let t0 = Instant::now();
    std::fs::create_dir_all(out)?;
    let rows: u64 = files.par_iter().map(|f| parquet_rows(f)).sum();
    let n_exp = expected.unwrap_or(rows as f64 * 49.0).max(1.0);
    let probe = match probe {
        Some(p) => {
            let pr = Probe::load(p, cap)?;
            eprintln!("probe: {} games, {} plies, {} distinct positions, {} edges ({:.0}s)", pr.lens.len(),
                      pr.plies.len(), pr.pos_idx.len(), pr.edge_idx.len(), t0.elapsed().as_secs_f64());
            Some(pr)
        }
        None => None,
    };
    let ctr = Counter::new(n_exp, fpr, exact);
    eprintln!("{} files, {} rows; {}: expected {:.3e} distinct keys, {} bits/shard x {} shards = {:.2} GB, k = {}",
              files.len(), rows, if exact { "exact (no Bloom)" } else { "Bloom + exact" }, n_exp, ctr.nbits,
              SHARDS, ctr.bloom_bytes() as f64 / 1e9, ctr.k);
    let first: Vec<AtomicU64> = (0..cap).map(|_| AtomicU64::new(0)).collect();
    let by_ply: Vec<AtomicU64> = (0..cap).map(|_| AtomicU64::new(0)).collect();
    let (games, skipped, short, done) = (AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0));
    let (bmin, bmax) = (band.0.unwrap_or(i64::MIN), band.1.unwrap_or(i64::MAX));
    let filtering = band.0.is_some() || band.1.is_some();

    files.par_iter().enumerate().try_for_each(|(fi, path)| -> io::Result<()> {
        let mut bno = 0u64;
        read_parquet(path, true, |b| {
            let n = b.len();
            bno += 1;
            let bseed = ((fi as u64) << 48) | (bno << 24);
            (0..n.div_ceil(CHUNK)).into_par_iter().for_each(|c| {
                let mut keys: Vec<(u64, u8)> = Vec::with_capacity(CHUNK * 64);
                let mut lply = vec![0u64; cap];
                let (mut ng, mut nskip, mut nshort) = (0u64, 0u64, 0u64);
                for i in c * CHUNK..((c + 1) * CHUNK).min(n) {
                    if filtering && !b.band(i).is_some_and(|x| x >= bmin && x <= bmax) {
                        nskip += 1;
                        continue;
                    }
                    let gid = match &b.ids {
                        Some(ids) if !ids.is_null(i) => fnv(ids.value(i).as_bytes()),
                        _ => bseed | i as u64,
                    };
                    let level = (splitmix(gid).trailing_zeros() as usize).min(LEVELS - 1);
                    let len = (b.offs[i + 1] - b.offs[i]) as usize;
                    let r = replay(b.moves(i), cap, |ply, k, e| {
                        if let Some(p) = &probe {
                            p.hit(k, e, level);
                        }
                        keys.push((k, ply as u8));
                        lply[ply] += 1;
                    });
                    if r < len.min(cap) {
                        nshort += 1;
                    }
                    ng += 1;
                }
                // counting sort by shard, then one lock per shard
                let mut cnt = vec![0usize; SHARDS + 1];
                for &(k, _) in &keys {
                    cnt[(k >> (64 - SHARD_BITS)) as usize + 1] += 1;
                }
                for s in 0..SHARDS {
                    cnt[s + 1] += cnt[s];
                }
                let mut sorted = vec![(0u64, 0u8); keys.len()];
                let mut at = cnt.clone();
                for &kv in &keys {
                    let s = (kv.0 >> (64 - SHARD_BITS)) as usize;
                    sorted[at[s]] = kv;
                    at[s] += 1;
                }
                let mut lfirst = vec![0u64; cap];
                for s in 0..SHARDS {
                    if cnt[s + 1] > cnt[s] {
                        ctr.insert(s, &sorted[cnt[s]..cnt[s + 1]], &mut lfirst);
                    }
                }
                for p in 0..cap {
                    first[p].fetch_add(lfirst[p], Relaxed);
                    by_ply[p].fetch_add(lply[p], Relaxed);
                }
                games.fetch_add(ng, Relaxed);
                skipped.fetch_add(nskip, Relaxed);
                short.fetch_add(nshort, Relaxed);
            });
        })?;
        let d = done.fetch_add(1, Relaxed) + 1;
        eprintln!("{d}/{} files, {} games, {:.0}s, peak RSS {:.1} GB", files.len(), games.load(Relaxed),
                  t0.elapsed().as_secs_f64(), peak_rss_gb());
        Ok(())
    })?;
    let t_count = t0.elapsed().as_secs_f64();

    // table: count >= 2, sorted by key
    let mut fill_bits = 0u64;
    let mut map_bytes = 0u64;
    let mut table: Vec<(u64, u32)> = Vec::new();
    for sh in &ctr.shards {
        let g = sh.lock().unwrap();
        fill_bits += g.bloom.iter().map(|w| w.count_ones() as u64).sum::<u64>();
        map_bytes += g.map.capacity() as u64 * 17;
        table.extend(g.map.iter().filter(|(_, &c)| c >= 2).map(|(&k, &c)| (k, c)));
    }
    let map_entries: u64 = ctr.shards.iter().map(|s| s.lock().unwrap().map.len() as u64).sum();
    let (k, bloom_bytes, total_bits) = (ctr.k as f64, ctr.bloom_bytes(), (ctr.nbits * SHARDS as u64).max(1) as f64);
    drop(ctr);
    table.par_sort_unstable_by_key(|x| x.0);
    write_npy(&out.join("table_keys.npy"), "<u8", &[table.len()], &table.iter().map(|x| x.0).collect::<Vec<_>>())?;
    write_npy(&out.join("table_counts.npy"), "<u4", &[table.len()], &table.iter().map(|x| x.1).collect::<Vec<_>>())?;
    if let Some(p) = &probe {
        p.write(out)?;
    }
    let first: Vec<u64> = first.iter().map(|x| x.load(Relaxed)).collect();
    let by_ply: Vec<u64> = by_ply.iter().map(|x| x.load(Relaxed)).collect();
    let n_first: u64 = first.iter().sum();
    let n_keys: u64 = by_ply.iter().sum();
    let fill = fill_bits as f64 / total_bits;
    let g = games.load(Relaxed);
    let elapsed = t0.elapsed().as_secs_f64();
    let j = format!(
        "{{\n \"mode\": \"{}\",\n \"files\": {},\n \"games\": {},\n \"games_band_skipped\": {},\n \"games_stopped_early\": {},\n \
         \"cap\": {},\n \"keys\": {},\n \"first_sightings\": {},\n \"table_keys\": {},\n \"map_entries\": {},\n \
         \"singletons_est\": {},\n \"bloom_expected_distinct\": {},\n \"bloom_target_fpr\": {},\n \"bloom_k\": {},\n \
         \"bloom_bytes\": {},\n \"bloom_fill\": {:.6},\n \"bloom_fpr_at_end\": {:.6},\n \"map_bytes_est\": {},\n \
         \"seconds_count\": {:.1},\n \"seconds_total\": {:.1},\n \"games_per_s\": {:.0},\n \"peak_rss_gb\": {:.2},\n \
         \"threads\": {},\n \"first_sightings_by_ply\": {},\n \"keys_by_ply\": {}\n}}\n",
        if exact { "exact" } else { "bloom" }, files.len(), g, skipped.load(Relaxed), short.load(Relaxed), cap, n_keys,
        n_first, table.len(), map_entries, n_first.saturating_sub(table.len() as u64), n_exp, fpr,
        if exact { 0 } else { k as u32 }, bloom_bytes, fill,
        if exact { 0.0 } else { fill.powf(k) }, map_bytes, t_count, elapsed, g as f64 / t_count, peak_rss_gb(),
        rayon::current_num_threads(), json_arr(&first), json_arr(&by_ply));
    std::fs::write(out.join("stats.json"), &j)?;
    eprint!("{j}");
    Ok(())
}

fn main() -> io::Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Keys { ep } => keys_cmd(&ep),
        Cmd::Count { out, fpr, expected, exact, cap, threads, band_min, band_max, probe, files } => {
            assert!(cap <= 255, "cap must fit in u8");
            rayon::ThreadPoolBuilder::new().num_threads(threads).build_global().unwrap();
            count_cmd(&out, fpr, expected, exact, cap, (band_min, band_max), probe.as_deref(), &files)
        }
    }
}
