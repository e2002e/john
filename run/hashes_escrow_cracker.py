#!/usr/bin/env python3
"""
hashes.com escrow auto-cracker (streaming founds + newest-first sweep)
======================================================================

Polls the hashes.com escrow board for MD5 "left" (unfound) lists and works them
with John the Ripper.

Behaviour
---------
  * STREAMING     - recovered hash:plaintext pairs are read straight from the
                    per-job pot and uploaded back to the escrow job as they are
                    cracked (on a short interval), not after the whole run ends.
  * SWEEP         - the board is swept newest -> oldest, one --slice time-slice
                    per list. If the slice expires before the keyspace is
                    exhausted the list is left "open" (resumed via john --restore
                    on a later pass). After a list's slice the next-older list is
                    picked, so no single hard list hogs the GPU; once every list
                    has had a slice the sweep restarts from the most recent. The
                    serve order is in-memory, so every startup begins a fresh
                    sweep at the newest list rather than resurfacing old backlog.
  * NEW-FIRST     - a brand-new list (never served this run) sorts to the front,
                    so it is picked on the next slice boundary. If a list newer
                    than anything known when the current slice began appears
                    mid-crack, the running John is aborted (saving its session)
                    so the newcomer is attacked straight away.

A list is retired (recorded in state, never revisited) only when John actually
finishes its keyspace. Detection: John keeps a <session>.rec restore file while a
session is incomplete and removes it on natural completion.

Founds are taken by parsing the per-job pot directly rather than via
`john --show`: on a username-less hash file `--show` prints "?:plaintext" with no
hash, which is useless for upload. The pot holds the real "hash:plaintext".

API reference: https://hashes.com/en/docs
  GET  /en/api/jobs        -> public escrow jobs (algorithmId + leftList path)
  <host>/<leftList>        -> the unfound list (one bare hash per line for raw MD5)
  POST /en/api/founds      -> key, algo, userfile=@founds.txt  (hash[:salt]:plaintext)

Auth is an API key (Account -> API on hashes.com), NOT a username/password.

Usage
-----
  export HASHES_COM_API_KEY=xxxxxxxx
  ./hashes_escrow_cracker.py --slice 900 --preempt-interval 60
  ./hashes_escrow_cracker.py --once --slice 600
  ./hashes_escrow_cracker.py --format raw-md5-opencl \
      --attack "--mask=?a --min-length=5 --max-length=16" --min-hashes 2000

  # Single-job override: crack ONLY one job by id and exit (no sweep/scheduling)
  ./hashes_escrow_cracker.py --job-id 123456
  ./hashes_escrow_cracker.py --job-id 123456 --format raw-md5-opencl \
      --attack "--mask=?a --min-length=5 --max-length=16"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urljoin

import requests

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

BASE_URL = "https://hashes.com"
JOBS_URL = f"{BASE_URL}/en/api/jobs"
FOUNDS_URL = f"{BASE_URL}/en/api/founds"

MD5_ALGO_ID = 0
DEFAULT_JOHN_FORMAT = "raw-md5"
KILL_GRACE_SECONDS = 30        # wait after SIGINT before SIGKILL backstop

# raw-MD5 pot lines are '<hash>:<pw>' or '$dynamic_0$<hash>:<pw>'.
POT_LINE_RE = re.compile(r"^(?:\$dynamic_0\$)?([0-9a-fA-F]{32}):(.*)$")

log = logging.getLogger("escrow")

# Set when SIGINT/SIGTERM arrives; the running John (if any) is tracked so the
# handler can tell it to save its session before we exit.
_shutdown = False
_current_john = None   # type: subprocess.Popen | None


def _install_signal_handlers() -> None:
    """On SIGINT or SIGTERM: forward SIGINT to John (so it saves its .rec for a
    clean --restore) and ask the scheduler to stop after the current list."""
    def handler(signum, _frame):
        global _shutdown
        _shutdown = True
        log.warning("signal %s: stopping; telling John to save its session", signum)
        j = _current_john
        if j is not None and j.poll() is None:
            try:
                j.send_signal(signal.SIGINT)
            except Exception:
                pass
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _is_our_john(pid: int, sess: Path) -> bool:
    """Best-effort check that <pid> is a John working *this* session (guards
    against a recycled PID). Linux /proc only; returns False if unsure."""
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except Exception:
        return False
    return str(sess) in raw.replace(b"\0", b" ").decode("utf-8", "ignore")


def reap_stale_john(job_dir: Path, sess: Path) -> None:
    """If a previous run left an orphaned John cracking this list, stop it (let
    it save first) before we touch the session, so we never run two at once."""
    pidfile = job_dir / "john.pid"
    if not pidfile.exists():
        return
    try:
        pid = int(pidfile.read_text().strip())
    except Exception:
        pidfile.unlink(missing_ok=True)
        return
    if not _pid_alive(pid) or not _is_our_john(pid, sess):
        pidfile.unlink(missing_ok=True)
        return
    log.warning("orphaned John pid %d still on this list; stopping it first", pid)
    try:
        os.kill(pid, signal.SIGINT)            # let it save .rec
        for _ in range(KILL_GRACE_SECONDS):
            if not _pid_alive(pid):
                break
            time.sleep(1)
        if _pid_alive(pid):
            os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    pidfile.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# State: job ids that are fully done (keyspace exhausted), never revisited
# --------------------------------------------------------------------------- #

class State:
    def __init__(self, path: Path):
        self.path = path
        self.done: set[int] = set()
        if path.exists():
            try:
                self.done = set(json.loads(path.read_text()))
            except Exception as exc:
                log.warning("could not read state %s: %s", path, exc)

    def mark(self, job_id: int) -> None:
        self.done.add(job_id)
        # Atomic: write a temp file then rename, so a crash mid-write can't
        # corrupt (and thereby lose) the whole retired-jobs set.
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(sorted(self.done)))
        os.replace(tmp, self.path)

    def __contains__(self, job_id: int) -> bool:
        return job_id in self.done


# --------------------------------------------------------------------------- #
# hashes.com API
# --------------------------------------------------------------------------- #

def get_session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = "escrow-autocracker/1.4"
    return s


def fetch_jobs(session: requests.Session, api_key: str) -> list[dict]:
    r = session.get(JOBS_URL, params={"key": api_key}, timeout=30)
    r.raise_for_status()
    data = r.json()
    if not data.get("success"):
        raise RuntimeError(f"/api/jobs returned failure: {data}")
    return data.get("list", [])


def fetch_job(session: requests.Session, api_key: str, job_id: int) -> dict | None:
    """Find a single escrow job by id in the public board listing."""
    for j in fetch_jobs(session, api_key):
        if j.get("id") == job_id:
            return j
    return None


def download_left_list(session: requests.Session, left_path: str, dest: Path) -> int:
    url = urljoin(BASE_URL, left_path)
    r = session.get(url, timeout=120)
    r.raise_for_status()
    dest.write_text(r.text)
    return sum(1 for line in r.text.splitlines() if line.strip())


def upload_founds_file(session: requests.Session, api_key: str, algo_id: int,
                       founds_file: Path) -> dict:
    with founds_file.open("rb") as fh:
        files = {"userfile": (founds_file.name, fh, "text/plain")}
        data = {"key": api_key, "algo": str(algo_id)}
        r = session.post(FOUNDS_URL, data=data, files=files, timeout=120)
    r.raise_for_status()
    return r.json()


# --------------------------------------------------------------------------- #
# Founds + upload
# --------------------------------------------------------------------------- #

def read_founds_from_pot(pot_file: Path) -> list[str]:
    """'hash:plaintext' lines straight from the per-job pot, normalised to the
    bare lowercase 32-hex hash the escrow board expects.

    Reading the pot directly (instead of `john --show`) is robust: on a
    username-less hash file `--show` prints '?:plaintext' with no hash, and it
    needs no subprocess or GPU. The per-job pot only ever holds this job's cracks
    (it is wiped on a fresh start), so no intersection with the left list is
    needed."""
    if not pot_file.exists():
        return []
    out = []
    for ln in pot_file.read_text(errors="replace").splitlines():
        m = POT_LINE_RE.match(ln.strip())
        if m:
            out.append(f"{m.group(1).lower()}:{m.group(2)}")
    return out


def flush_new_founds(session, api_key: str, algo_id: int, lines: list[str],
                     uploaded: set[str], delta_file: Path, uploaded_file: Path,
                     dry_run: bool) -> int:
    """Upload founds whose hash isn't already uploaded; persist successes so a
    resumed pass never re-POSTs them. A found whose upload FAILS stays out of the
    persisted set and is retried on the next flush/pass."""
    new = []
    for ln in lines:
        h = ln.split(":", 1)[0].lower()
        if h not in uploaded:
            new.append((h, ln))
    if not new:
        return 0
    if not dry_run:
        delta_file.write_text("\n".join(ln for _, ln in new) + "\n")
        resp = upload_founds_file(session, api_key, algo_id, delta_file)
        if not resp.get("success"):
            log.error("delta upload failed (%d founds): %s", len(new), resp)
            return 0
        with uploaded_file.open("a") as fh:
            for h, _ in new:
                fh.write(h + "\n")
    for h, _ in new:
        uploaded.add(h)
    return len(new)


# --------------------------------------------------------------------------- #
# Scheduling: sweep newest -> oldest, one slice each, repeating; new lists jump in
# --------------------------------------------------------------------------- #

def _job_dir(workdir: Path, job_id) -> Path:
    return (workdir / str(job_id)).resolve()


def _created_key(job: dict) -> str:
    """A comparable, sortable representation of a job's createdAt that tolerates
    str (ISO-8601) or numeric (epoch) values and missing/empty ones (sort
    oldest). Numerics are zero-padded so they order correctly as strings, and
    everything is returned as a single str so heterogeneous values never raise
    TypeError when compared (mixed types from one API response shouldn't happen,
    but a stray null must not crash scheduling)."""
    v = job.get("createdAt")
    if v is None or v == "":
        return ""
    if isinstance(v, (int, float)):
        return f"{int(v):020d}"
    return str(v)


def _newest_key(job: dict):
    """Unambiguous 'newest-first' sort key: primary createdAt (_created_key),
    tie-broken by numeric job id (higher id == newer on hashes.com). The id
    tiebreak matters because createdAt can be coarse (date- or second-grained);
    without it max() would return an arbitrary one of several same-timestamp
    jobs, so 'the very newest' would not be guaranteed on a (re)start."""
    try:
        jid = int(job.get("id", 0))
    except (TypeError, ValueError):
        jid = 0
    return (_created_key(job), jid)


def choose_next(candidates: list[dict], served: dict) -> dict | None:
    """Pick the next list so the daemon sweeps newest -> oldest, one slice each,
    then repeats from the top.

    Among the lists waiting longest (least-recently-served *this run*; a
    never-served list counts as 0.0, so on startup every list is "waiting") take
    the newest-created. After a list gets its slice it is stamped served=now, so
    the next pick is the next-older still-waiting list; once every list has had a
    slice the least-recently-served one is again the most recent, so the sweep
    restarts from the top.

    `served` is in-memory (reset every daemon run), so each startup/resume begins
    a fresh sweep at the most recent list instead of resurfacing old backlog from
    a persisted stamp. New arrivals (served=0.0) naturally jump to the front.
    """
    if not candidates:
        return None
    least = min(served.get(j["id"], 0.0) for j in candidates)
    waiting = [j for j in candidates if served.get(j["id"], 0.0) <= least + 1e-6]
    return max(waiting, key=_newest_key)


def check_newer_job(session, api_key: str, watermark,
                    state: State, min_hashes: int,
                    current_job_id: int) -> dict | None:
    """Return the newest eligible job whose newest-key is strictly greater than
    the watermark (= newest list known when the current slice began), i.e. a
    list that *appeared after* we started. Only such a genuinely-new list
    preempts; already-known open lists are left for the normal sweep."""
    try:
        jobs = fetch_jobs(session, api_key)
    except Exception as exc:
        log.debug("check_newer_job poll failed: %s", exc)
        return None
    newer = [
        j for j in jobs
        if j.get("algorithmId") == MD5_ALGO_ID
        and j.get("leftHashes", 0) >= min_hashes
        and j["id"] not in state
        and j["id"] != current_job_id
        and j.get("createdAt")
        and _newest_key(j) > watermark
    ]
    return max(newer, key=_newest_key) if newer else None


# --------------------------------------------------------------------------- #
# Per-job pipeline (streaming, resumable, time-sliced)
# --------------------------------------------------------------------------- #

def process_job(job: dict, session, api_key: str, workdir: Path, john_bin: str,
                attack: list[str], fmt: str, slice_seconds: int,
                upload_interval: int, preempt_interval: int, dry_run: bool,
                state: State, min_hashes: int, watermark=("", 0)) -> bool:
    """Work one list for up to `slice_seconds` seconds.
    Returns True  if John exhausted its keyspace (retire the job),
            False if the slice expired mid-attack or a newer job appeared."""
    global _current_john

    job_id = job["id"]
    left = job.get("leftList")
    if not left:
        log.info("job %s has no leftList, skipping", job_id)
        return True

    job_dir = _job_dir(workdir, job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    hash_file = job_dir / "left.txt"
    pot_file = job_dir / "job.pot"
    delta_file = job_dir / "delta.txt"
    uploaded_file = job_dir / "uploaded_hashes.txt"
    sess = job_dir / "sess"            # John session base
    rec_file = job_dir / "sess.rec"    # John restore file (present == incomplete)
    algo_id = job.get("algorithmId", MD5_ALGO_ID)

    # Clean up any orphaned John from a previous run.
    reap_stale_john(job_dir, sess)

    resuming = rec_file.exists() and hash_file.exists()
    if resuming:
        n = sum(1 for l in hash_file.read_text().splitlines() if l.strip())
    else:
        # Fresh start: (re)download and clear any stale session/pot for this id.
        n = download_left_list(session, left, hash_file)
        for stale in (rec_file, pot_file, uploaded_file):
            stale.unlink(missing_ok=True)
    log.info("job %s: %d hashes (%s)%s", job_id, n, job["algorithmName"],
             " [resuming]" if resuming else "")
    if n == 0:
        return True

    # Seed the already-uploaded set from the durable per-job record.
    uploaded: set[str] = set()
    if uploaded_file.exists():
        uploaded = {l.strip().lower()
                    for l in uploaded_file.read_text().splitlines() if l.strip()}

    if resuming:
        cmd = [john_bin, f"--restore={sess}"]
    else:
        cmd = [john_bin, f"--format={fmt}", f"--pot={pot_file}",
               f"--session={sess}", *attack, str(hash_file)]
    log.info("job %s: %s", job_id, " ".join(cmd))

    # John's stderr goes to a per-job logfile, NOT a pipe: an undrained PIPE
    # (we never read it mid-run) fills its ~64 KB buffer over a long slice and
    # deadlocks John on write. A file never blocks and is handy for debugging.
    john_log = job_dir / "john.log"
    errlog = john_log.open("w")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=errlog)
    _current_john = proc                   # allow signal handler to manage this John
    # Store PID for future orphan checks (even if we crash).
    pidfile = job_dir / "john.pid"
    pidfile.write_text(str(proc.pid))

    deadline = time.time() + slice_seconds if slice_seconds > 0 else None
    last_check = time.time()

    def flush():
        # Never let a transient upload error (5xx/timeout/connection blip) abort
        # the crack: founds that fail to POST stay out of `uploaded` and are
        # retried on the next flush, so swallowing here is safe and resilient.
        try:
            got = flush_new_founds(session, api_key, algo_id,
                                   read_founds_from_pot(pot_file),
                                   uploaded, delta_file, uploaded_file, dry_run)
        except Exception as exc:
            log.warning("job %s: founds upload failed (will retry next flush): %s",
                        job_id, exc)
            return
        if got:
            verb = "would upload" if dry_run else "uploaded"
            log.info("job %s: %s +%d (uploaded %d/%d)", job_id, verb,
                     got, len(uploaded), n)

    try:
        while proc.poll() is None:
            time.sleep(upload_interval)
            flush()

            # Preempt for a genuinely newer list (every preempt_interval seconds).
            if preempt_interval > 0 and time.time() - last_check >= preempt_interval:
                last_check = time.time()
                newer = check_newer_job(session, api_key, watermark,
                                        state, min_hashes, job_id)
                if newer:
                    log.info("job %s: newer list %s appeared (created %s), aborting current",
                             job_id, newer["id"], newer["createdAt"])
                    proc.send_signal(signal.SIGINT)   # save .rec
                    try:
                        proc.wait(timeout=KILL_GRACE_SECONDS)
                    except subprocess.TimeoutExpired:
                        log.warning("job %s: John ignored SIGINT, killing", job_id)
                        proc.kill()
                        proc.wait()
                    return False   # not completed; resume later (finally cleans up)

            if deadline and time.time() > deadline:
                log.warning("job %s: time slice up, SIGINT (John saves session)", job_id)
                proc.send_signal(signal.SIGINT)   # John: abort + save .rec for --restore
                try:
                    proc.wait(timeout=KILL_GRACE_SECONDS)
                except subprocess.TimeoutExpired:
                    log.warning("job %s: John ignored SIGINT, killing", job_id)
                    proc.kill()
                break
        proc.wait()
    finally:
        # Clear our handles FIRST so a raising final flush can't leave a dangling
        # _current_john / stale john.pid / open logfile.
        _current_john = None
        pidfile.unlink(missing_ok=True)
        errlog.close()
        flush()   # final sweep (guarded internally)

    try:
        err = john_log.read_text(errors="replace")[-1000:]
    except Exception:
        err = ""
    if proc.returncode not in (0, None) and err.strip():
        log.warning("job %s: John rc=%s stderr tail: %s", job_id, proc.returncode, err)

    completed = not rec_file.exists()
    log.info("job %s: %s, uploaded %d/%d this run", job_id,
             "COMPLETE (retiring)" if completed else "interrupted (resume next pass)",
             len(uploaded), n)
    return completed


# --------------------------------------------------------------------------- #
# Main scheduler
# --------------------------------------------------------------------------- #

def scheduler(session, api_key: str, state: State, workdir: Path, john_bin: str,
              attack: list[str], fmt: str, min_hashes: int, slice_seconds: int,
              upload_interval: int, preempt_interval: int, idle_interval: int,
              once: bool, dry_run: bool) -> None:
    """Work one list at a time. After every slice (or early abort) re-poll the
    board and pick the next list per choose_next: a newest-first sweep that
    advances to the next-older list each slice and restarts at the top once all
    have been served (new arrivals jump to the front)."""
    global _shutdown

    # Crash backoff: a job that returns without completing AND without leaving a
    # resumable .rec made no progress; after a few such passes quarantine it for
    # this run so it can't hot-loop (re-pick, re-fail) and starve real work.
    QUARANTINE_AFTER = 3
    failures: dict[int, int] = {}
    quarantined: set[int] = set()
    # In-memory serve stamps (job_id -> last slice-start time). Reset every run so
    # each startup sweeps newest -> oldest from the top; see choose_next.
    served: dict[int, float] = {}

    while not _shutdown:
        try:
            jobs = fetch_jobs(session, api_key)
        except Exception as exc:
            log.error("board poll failed: %s", exc)
            if once or _shutdown:
                return
            time.sleep(idle_interval)
            continue

        candidates = [
            j for j in jobs
            if j.get("algorithmId") == MD5_ALGO_ID
            and j.get("leftHashes", 0) >= min_hashes
            and j["id"] not in state
            and j["id"] not in quarantined
        ]
        if not candidates:
            if _shutdown:
                return
            log.info("no MD5 work available")
            if once:
                return
            time.sleep(idle_interval)
            continue

        watermark = max((_newest_key(c) for c in candidates), default=("", 0))
        job = choose_next(candidates, served)
        served[job["id"]] = time.time()   # slice start: so the next pick advances
                                          # to the next-older list, not this one
        is_resume = _job_dir(workdir, job["id"]).is_dir()
        log.info("picking job %s (created %s) -> %s  [%d candidate(s)]",
                 job["id"], job.get("createdAt", "?"),
                 "RESUME" if is_resume else "NEW", len(candidates))
        try:
            completed = process_job(job, session, api_key, workdir, john_bin,
                                    attack, fmt, slice_seconds, upload_interval,
                                    preempt_interval, dry_run, state, min_hashes,
                                    watermark=watermark)
            jid = job["id"]
            resumable = (_job_dir(workdir, jid) / "sess.rec").exists()
            if completed:
                if not dry_run:
                    state.mark(jid)
                failures.pop(jid, None)
            elif resumable:
                failures.pop(jid, None)   # slice/preempt interrupt = real progress
            else:
                failures[jid] = failures.get(jid, 0) + 1
                if failures[jid] >= QUARANTINE_AFTER:
                    quarantined.add(jid)
                    log.error("job %s failed %d× with no resumable progress; "
                              "quarantining for this run", jid, failures[jid])
        except requests.HTTPError as exc:
            log.error("job %s HTTP error: %s", job.get("id"), exc)
        except Exception as exc:
            log.exception("job %s failed: %s", job.get("id"), exc)
        # Loop straight back - no idle wait when work is present.


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="hashes.com escrow MD5 auto-cracker "
                    "(streaming founds + newest-first sweep, resumable)")

    # --- credentials / binaries ---------------------------------------- #
    p.add_argument("--api-key", default=os.environ.get("HASHES_COM_API_KEY", ""),
                   help="hashes.com API key (default: $HASHES_COM_API_KEY)")
    p.add_argument("--john", default=os.environ.get("JOHN_BIN", "john"),
                   help="path to the John the Ripper binary")
    p.add_argument("--workdir", default="./escrow_work", type=Path,
                   help="scratch + state directory (per-job sessions live here)")

    # --- attack ---------------------------------------------------------- #
    p.add_argument("--format", dest="fmt", default=DEFAULT_JOHN_FORMAT,
                   help=f"John --format (default: {DEFAULT_JOHN_FORMAT})")
    p.add_argument("--attack",
                   default="--wordlist=/usr/share/wordlists/rockyou.txt --rules=best64",
                   help="space-separated John attack flags for a fresh start "
                        "(a resumed list continues its saved attack)")
    p.add_argument("--min-hashes", type=int, default=1,
                   help="skip jobs with fewer than N left hashes (default: 1)")

    # --- timing: three distinct knobs ----------------------------------- #
    p.add_argument("--slice", dest="slice_seconds", type=int, default=900,
                   help="per-list time slice in seconds before advancing to the "
                        "next list in the sweep (0 = run each list to completion; "
                        "default: 900)")
    p.add_argument("--upload-interval", type=int, default=15,
                   help="seconds between founds uploads while John runs "
                        "(default: 15)")
    p.add_argument("--preempt-interval", type=int, default=60,
                   help="seconds between checks for a newer list; if one appears "
                        "the current crack is aborted at once (0 = never preempt; "
                        "default: 60)")
    p.add_argument("--idle-interval", type=int, default=300,
                   help="seconds to wait when the board has no work; otherwise the "
                        "board is re-polled after every slice (default: 300)")

    # --- modes ----------------------------------------------------------- #
    p.add_argument("--job-id", type=int, default=None,
                   help="crack ONLY this one escrow job (fetched by id) and exit; "
                        "overrides the default newest-first board sweep (no "
                        "scheduling, no preemption)")
    p.add_argument("--once", action="store_true",
                   help="work available lists until the board is empty, then exit")
    p.add_argument("--dry-run", action="store_true",
                   help="crack but do not upload, and do not retire jobs")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.api_key:
        log.error("No API key. Set HASHES_COM_API_KEY or pass --api-key.")
        return 2
    if args.upload_interval < 1:
        log.error("--upload-interval must be >= 1")
        return 2

    args.workdir.mkdir(parents=True, exist_ok=True)
    # This file holds retired (keyspace-exhausted) job IDs, not uploaded hashes
    # (those live per-job in uploaded_hashes.txt). Migrate the old misleading name.
    retired_path = args.workdir / "retired.json"
    legacy_path = args.workdir / "uploaded.json"
    if legacy_path.exists() and not retired_path.exists():
        legacy_path.rename(retired_path)
    state = State(retired_path)
    session = get_session()
    attack = args.attack.split()

    _install_signal_handlers()

    # Single-job override: fetch one job by id, crack it, exit. No sweep,
    # no preemption, no retire-state. Streaming founds reporting is unchanged.
    if args.job_id is not None:
        try:
            job = fetch_job(session, args.api_key, args.job_id)
        except Exception as exc:
            log.error("could not fetch job %s: %s", args.job_id, exc)
            return 1
        if job is None:
            log.error("job %s not found on the escrow board", args.job_id)
            return 1
        log.info("single-job mode: job %s (created %s, %s left)",
                 job["id"], job.get("createdAt", "?"), job.get("leftHashes", "?"))
        try:
            completed = process_job(
                job, session, args.api_key, args.workdir, args.john, attack,
                args.fmt, args.slice_seconds, args.upload_interval,
                preempt_interval=0, dry_run=args.dry_run, state=state,
                min_hashes=args.min_hashes)
            if completed and not args.dry_run:
                state.mark(job["id"])
        except KeyboardInterrupt:
            log.info("bye")
        except Exception as exc:
            log.exception("job %s failed: %s", args.job_id, exc)
            return 1
        return 0

    if not args.once:
        log.info("daemon: newest-first sweep, re-pick after each slice "
                 "(idle re-poll every %ds, ctrl-c to stop)", args.idle_interval)
    try:
        scheduler(session, args.api_key, state, args.workdir, args.john, attack,
                  args.fmt, args.min_hashes, args.slice_seconds,
                  args.upload_interval, args.preempt_interval, args.idle_interval,
                  args.once, args.dry_run)
    except KeyboardInterrupt:
        log.info("bye")
    return 0


if __name__ == "__main__":
    sys.exit(main())
