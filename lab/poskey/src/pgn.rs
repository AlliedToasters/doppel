//! PGN stream -> compact moves-only parquet (`poskey pgn`), so PGN databases can be counted like the corpus.
//!
//! Reads PGN text on stdin (pipe `zstd -dc` in front), replays SAN with shakmaty up to `cap` plies and writes
//! parquet parts of `--per-file` games with the columns `count` reads (`moves` as UCI, `band`, `game_id`) and the
//! headers an analysis needs. Comments `{}` and `;`, variations `()`, NAGs `$n`, move numbers (`1.e4`, `1...e5`)
//! and `!?` suffixes are skipped. A game stops at a null move `--` or the first SAN that does not parse or is
//! illegal (`stop` column: 0 = end or cap, 1 = null move, 2 = bad SAN).

use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;
use std::sync::Arc;

use arrow_array::builder::{Int64Builder, ListBuilder, StringBuilder};
use arrow_array::{ArrayRef, RecordBatch};
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;
use rayon::prelude::*;
use shakmaty::san::San;
use shakmaty::{CastlingMode, Chess, Position};

#[derive(Default)]
struct RawGame {
    headers: Vec<(String, String)>,
    movetext: String,
}

struct Row {
    white: String,
    black: String,
    white_elo: Option<i64>,
    black_elo: Option<i64>,
    white_fide: Option<i64>,
    black_fide: Option<i64>,
    year: Option<i64>,
    eco: String,
    result: String,
    event: String,
    time_control: String,
    plycount: Option<i64>,
    moves: Vec<String>,
    stop: i64,
}

fn header_line(line: &str) -> Option<(String, String)> {
    let s = line.trim();
    let s = s.strip_prefix('[')?.strip_suffix(']')?;
    let (tag, rest) = s.split_once(' ')?;
    let v = rest.trim().strip_prefix('"')?.strip_suffix('"')?;
    Some((tag.to_string(), v.replace("\\\"", "\"").replace("\\\\", "\\")))
}

fn parse_int(v: Option<&str>) -> Option<i64> {
    v.and_then(|s| s.trim().parse::<i64>().ok()).filter(|&x| x > 0)
}

/// SAN tokens of the mainline (no comments, variations, NAGs, numbers or results).
fn mainline_tokens(t: &str) -> Vec<&str> {
    let b = t.as_bytes();
    let mut out = Vec::new();
    let (mut i, mut depth) = (0usize, 0usize);
    while i < b.len() {
        match b[i] {
            b'{' => {
                while i < b.len() && b[i] != b'}' {
                    i += 1;
                }
                i += 1;
            }
            b';' => {
                while i < b.len() && b[i] != b'\n' {
                    i += 1;
                }
            }
            b'(' => {
                depth += 1;
                i += 1;
            }
            b')' => {
                depth = depth.saturating_sub(1);
                i += 1;
            }
            c if c.is_ascii_whitespace() => i += 1,
            _ => {
                let s = i;
                while i < b.len() && !b[i].is_ascii_whitespace() && !matches!(b[i], b'{' | b'(' | b')' | b';') {
                    i += 1;
                }
                if depth == 0 {
                    out.push(&t[s..i]);
                }
            }
        }
    }
    out
}

fn replay_san(movetext: &str, cap: usize) -> (Vec<String>, i64) {
    let mut pos = Chess::default();
    let mut moves = Vec::new();
    for tok in mainline_tokens(movetext) {
        if moves.len() >= cap {
            break;
        }
        if tok.starts_with('$') {
            continue;
        }
        // strip a move number: "12." / "12..." / "12.e4"
        let t = tok.trim_start_matches(|c: char| c.is_ascii_digit());
        let t = if t.len() < tok.len() && t.starts_with('.') { t.trim_start_matches('.') } else { tok };
        if t.is_empty() {
            continue;
        }
        if matches!(t, "1-0" | "0-1" | "1/2-1/2" | "*") {
            break;
        }
        let t = t.trim_end_matches(['!', '?']);
        let fixed;
        let t = if t.starts_with("0-0") {
            fixed = t.replace('0', "O");
            fixed.as_str()
        } else {
            t
        };
        if t == "--" || t == "Z0" {
            return (moves, 1);
        }
        let Ok(san) = San::from_ascii(t.as_bytes()) else { return (moves, 2) };
        let Ok(m) = san.to_move(&pos) else { return (moves, 2) };
        moves.push(m.to_uci(CastlingMode::Standard).to_string());
        pos.play_unchecked(&m);
    }
    (moves, 0)
}

/// corpus.py `keep`: rated blitz, no BOT account, Termination Normal or Time forfeit, both Elo in 600-2800,
/// moves carrying [%clk].
fn lichess_blitz(g: &RawGame) -> bool {
    let h = |k: &str| g.headers.iter().find(|(t, _)| t == k).map(|(_, v)| v.as_str());
    let elo_ok = |k: &str| h(k).and_then(|s| s.parse::<i64>().ok()).is_some_and(|e| (600..=2800).contains(&e));
    h("Event") == Some("Rated Blitz game")
        && h("WhiteTitle") != Some("BOT")
        && h("BlackTitle") != Some("BOT")
        && matches!(h("Termination"), Some("Normal") | Some("Time forfeit"))
        && elo_ok("WhiteElo")
        && elo_ok("BlackElo")
        && g.movetext.contains("%clk")
}

fn to_row(g: &RawGame, cap: usize) -> Row {
    let h = |k: &str| g.headers.iter().find(|(t, _)| t == k).map(|(_, v)| v.as_str());
    let (moves, stop) = replay_san(&g.movetext, cap);
    let date = h("Date").unwrap_or("");
    Row {
        white: h("White").unwrap_or("").to_string(),
        black: h("Black").unwrap_or("").to_string(),
        white_elo: parse_int(h("WhiteElo")),
        black_elo: parse_int(h("BlackElo")),
        white_fide: parse_int(h("WhiteFideId")),
        black_fide: parse_int(h("BlackFideId")),
        year: date.get(0..4).and_then(|y| y.parse::<i64>().ok()),
        eco: h("ECO").unwrap_or("").chars().take(3).collect(),
        result: h("Result").unwrap_or("").to_string(),
        event: h("Event").unwrap_or("").to_string(),
        time_control: h("TimeControl").unwrap_or("").to_string(),
        plycount: h("PlyCount").and_then(|s| s.trim().parse::<i64>().ok()),
        moves,
        stop,
    }
}

fn write_part(path: &Path, rows: &[Row], first_id: u64) -> io::Result<()> {
    let n = rows.len();
    let s = |f: &dyn Fn(&Row) -> &str| {
        let mut b = StringBuilder::with_capacity(n, n * 8);
        rows.iter().for_each(|r| b.append_value(f(r)));
        Arc::new(b.finish()) as ArrayRef
    };
    let i = |f: &dyn Fn(&Row) -> Option<i64>| {
        let mut b = Int64Builder::with_capacity(n);
        rows.iter().for_each(|r| b.append_option(f(r)));
        Arc::new(b.finish()) as ArrayRef
    };
    let mut ids = StringBuilder::with_capacity(n, n * 10);
    (0..n).for_each(|k| ids.append_value(format!("L{}", first_id + k as u64)));
    let mut mv = ListBuilder::new(StringBuilder::new());
    for r in rows {
        for m in &r.moves {
            mv.values().append_value(m);
        }
        mv.append(true);
    }
    // band: mean of both Elos floored to 100, only when both are present (as the corpus's band column)
    let band = |r: &Row| match (r.white_elo, r.black_elo) {
        (Some(a), Some(b)) => Some((a + b) / 2 / 100 * 100),
        _ => None,
    };
    let batch = RecordBatch::try_from_iter(vec![
        ("game_id", Arc::new(ids.finish()) as ArrayRef),
        ("white", s(&|r| &r.white)),
        ("black", s(&|r| &r.black)),
        ("white_elo", i(&|r| r.white_elo)),
        ("black_elo", i(&|r| r.black_elo)),
        ("white_fide", i(&|r| r.white_fide)),
        ("black_fide", i(&|r| r.black_fide)),
        ("band", i(&band)),
        ("year", i(&|r| r.year)),
        ("eco", s(&|r| &r.eco)),
        ("result", s(&|r| &r.result)),
        ("event", s(&|r| &r.event)),
        ("time_control", s(&|r| &r.time_control)),
        ("plycount", i(&|r| r.plycount)),
        ("stop", i(&|r| Some(r.stop))),
        ("moves", Arc::new(mv.finish()) as ArrayRef),
    ])
    .map_err(io::Error::other)?;
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::try_new(3).unwrap()))
        .build();
    let mut w = ArrowWriter::try_new(File::create(path)?, batch.schema(), Some(props)).map_err(io::Error::other)?;
    w.write(&batch).map_err(io::Error::other)?;
    w.close().map_err(io::Error::other)?;
    Ok(())
}

pub fn pgn_cmd(out: &Path, cap: usize, per_file: usize, blitz_only: bool) -> io::Result<()> {
    std::fs::create_dir_all(out)?;
    let t0 = std::time::Instant::now();
    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();
    let (mut part, mut total, mut stops) = (0usize, 0u64, [0u64; 3]);
    let mut cur = RawGame::default();
    let mut games: Vec<RawGame> = Vec::with_capacity(per_file);
    let flush = |games: &mut Vec<RawGame>, part: &mut usize, total: &mut u64, stops: &mut [u64; 3]| -> io::Result<()> {
        if games.is_empty() {
            return Ok(());
        }
        let rows: Vec<Row> =
            games.par_iter().filter(|g| !blitz_only || lichess_blitz(g)).map(|g| to_row(g, cap)).collect();
        for r in &rows {
            stops[r.stop as usize] += 1;
        }
        if rows.is_empty() {
            games.clear();
            return Ok(());
        }
        write_part(&out.join(format!("part-{:04}.parquet", *part)), &rows, *total)?;
        *total += rows.len() as u64;
        *part += 1;
        games.clear();
        eprintln!("{} games, {} parts, stops {:?}, {:.0}s", total, part, stops, t0.elapsed().as_secs_f64());
        Ok(())
    };
    loop {
        let line = match lines.next() {
            Some(l) => l?,
            None => break,
        };
        if line.starts_with('[') {
            if let Some(kv) = header_line(&line) {
                if !cur.movetext.is_empty() || (kv.0 == "Event" && !cur.headers.is_empty()) {
                    games.push(std::mem::take(&mut cur));
                    if games.len() >= per_file {
                        flush(&mut games, &mut part, &mut total, &mut stops)?;
                    }
                }
                cur.headers.push(kv);
                continue;
            }
        }
        if !line.trim().is_empty() {
            cur.movetext.push_str(&line);
            cur.movetext.push('\n');
        }
    }
    if !cur.headers.is_empty() || !cur.movetext.is_empty() {
        games.push(cur);
    }
    flush(&mut games, &mut part, &mut total, &mut stops)?;
    eprintln!("done: {total} games in {part} parts, stops (end/cap, null move, bad SAN) {stops:?}");
    Ok(())
}
