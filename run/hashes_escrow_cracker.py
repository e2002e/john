#!/usr/bin/env python3
"""
hashes.com escrow auto-cracker (streaming founds + newest-first sweep)
======================================================================

Polls the hashes.com escrow board for MD5 and/or NTLM "left" (unfound) lists and
works them with John the Ripper on the GPU. The hash type(s) to work are chosen
with --hash-type, the per-list crack order with --order, and the John OpenCL
format is selected automatically per job from its hash type.

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
  * COMBINE       - with --combine the per-job sweep is replaced by a per-hash-type
                    UNION: every open job of a type is merged into one hash file and
                    cracked in a SINGLE John run, so the keyspace is swept once for
                    all of them instead of once per job. Founds upload per algo (the
                    board routes each hash to its job); a job is retired once all its
                    hashes are cracked. See process_algo_union().

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

  # Work only NTLM lists, picking the list with the most hashes still left
  ./hashes_escrow_cracker.py --hash-type ntlm --order left \
      --attack "--mask=?a --min-length=5 --max-length=16" --min-hashes 2000

  # Both MD5 and NTLM, favouring lists we've already cracked the most of
  ./hashes_escrow_cracker.py --hash-type both --order cracked

  # COMBINE: crack every open MD5 job in one merged run (one keyspace sweep for
  # all of them), re-merging the union after each slice as jobs come and go.
  ./hashes_escrow_cracker.py --hash-type md5 --combine \
      --attack "--mask=?a --min-length=1 --max-length=8" --slice 1800

  # Single-job override: crack ONLY one job by id and exit (no sweep/scheduling).
  # The OpenCL format is auto-selected from the job's hash type.
  ./hashes_escrow_cracker.py --job-id 123456

The John OpenCL format is always chosen automatically from each job's hash type
(md5 -> raw-md5-opencl, ntlm -> NT-opencl); there is no --format option.
"""

from __future__ import annotations

import argparse
import hashlib
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

KILL_GRACE_SECONDS = 30        # wait after SIGINT before SIGKILL backstop

# --------------------------------------------------------------------------- #
# Supported hash types
#
# Each entry maps a hash-type key (used by --hash-type) to:
#   ids       - hashes.com algorithmId value(s) for this type
#   names     - lowercased algorithmName aliases (a robust fallback in case the
#               numeric id ever drifts or is unknown for a board entry)
#   john_base - the John CPU format label; "-opencl" is appended automatically
#               when the crack command is built (so md5 -> raw-md5-opencl,
#               ntlm -> NT-opencl)
#   pot_re    - matches this format's pot lines and captures (hash, plaintext).
#               Both raw-MD5 and NT escrow lists are bare 32-hex, but John's pot
#               stores the format-tagged ciphertext ($dynamic_0$.. / $NT$..),
#               which we strip back to the bare lowercase hash the board wants.
# --------------------------------------------------------------------------- #
_HEX32 = r"([0-9a-fA-F]{32})"
ALGOS: dict[str, dict] = {
    "md5": {
        "ids": {0},
        "names": {"md5"},
        "john_base": "raw-md5",
        "pot_re": re.compile(r"^(?:\$dynamic_0\$)?" + _HEX32 + r":(.*)$"),
    },
    "ntlm": {
        "ids": {1000},
        "names": {"ntlm", "nt"},
        "john_base": "NT",
        "pot_re": re.compile(r"^(?:\$NT\$)?" + _HEX32 + r":(.*)$"),
    },
}
DEFAULT_ALGO_ID = 0            # founds-upload fallback when a job omits its id


def classify_job(job: dict) -> str | None:
    """Return the ALGOS key for a board job (by algorithmId, else algorithmName),
    or None if it is not a type we support."""
    aid = job.get("algorithmId")
    name = str(job.get("algorithmName", "")).strip().lower()
    for key, spec in ALGOS.items():
        if aid in spec["ids"] or name in spec["names"]:
            return key
    return None


def john_format_for(htype: str) -> str:
    """John OpenCL format label for a hash-type key (e.g. md5 -> raw-md5-opencl)."""
    return ALGOS[htype]["john_base"] + "-opencl"


def algo_upload_id(htype: str) -> int:
    """Canonical algorithmId to POST founds under for a hash-type key. The founds
    endpoint is keyed by algo (not job), so one upload covers every job of the type
    and the board routes each hash:plaintext to whatever open job(s) contain it."""
    ids = ALGOS[htype]["ids"]
    return min(ids) if ids else DEFAULT_ALGO_ID

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

def read_founds_from_pot(pot_file: Path, pot_re: re.Pattern) -> list[str]:
    """'hash:plaintext' lines straight from the per-job pot, normalised to the
    bare lowercase 32-hex hash the escrow board expects.

    Reading the pot directly (instead of `john --show`) is robust: on a
    username-less hash file `--show` prints '?:plaintext' with no hash, and it
    needs no subprocess or GPU. The per-job pot only ever holds this job's cracks
    (it is wiped on a fresh start), so no intersection with the left list is
    needed. `pot_re` is the per-format matcher (see ALGOS)."""
    if not pot_file.exists():
        return []
    out = []
    for ln in pot_file.read_text(errors="replace").splitlines():
        m = pot_re.match(ln.strip())
        if m:
            out.append(f"{m.group(1).lower()}:{m.group(2)}")
    return out


def local_cracked_count(job_dir: Path) -> int:
    """How many keys we have cracked locally for this job (non-blank per-job pot
    lines). Used by the 'cracked' crack order. 0 for a job never worked."""
    pot = job_dir / "job.pot"
    if not pot.exists():
        return 0
    try:
        return sum(1 for l in pot.read_text(errors="replace").splitlines() if l.strip())
    except Exception:
        return 0


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


def _left_hashes(job: dict) -> int:
    try:
        return int(job.get("leftHashes", 0) or 0)
    except (TypeError, ValueError):
        return 0


def make_order_key(order: str, workdir: Path):
    """Build the ranking key used to pick the next list. The scheduler always
    takes the candidate with the GREATEST key, so each key is shaped so that
    'most preferred' compares highest. Ties fall back to newest-first for a
    deterministic, stable choice.

      date    - newest created first (the original behaviour)
      left    - most hashes still left first (largest crackable opportunity)
      cracked - most keys cracked locally (by us) first (favour productive lists)
    """
    if order == "left":
        return lambda j: (_left_hashes(j), _newest_key(j))
    if order == "cracked":
        return lambda j: (local_cracked_count(_job_dir(workdir, j["id"])),
                          _newest_key(j))
    # default / "date"
    return _newest_key


def choose_next(candidates: list[dict], served: dict, order_key) -> dict | None:
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
    return max(waiting, key=order_key)


def check_newer_job(session, api_key: str, watermark, state: State,
                    min_hashes: int, current_job_id: int,
                    selected: set[str], order_key) -> dict | None:
    """Return the eligible job that out-ranks the watermark (= the best order_key
    among the candidates known when the current slice began), i.e. a list that
    *appeared after* we started and that the chosen crack order prefers over
    everything we already knew. Only such a genuinely-better newcomer preempts;
    already-known open lists are left for the normal sweep."""
    try:
        jobs = fetch_jobs(session, api_key)
    except Exception as exc:
        log.debug("check_newer_job poll failed: %s", exc)
        return None
    newer = [
        j for j in jobs
        if classify_job(j) in selected
        and _left_hashes(j) >= min_hashes
        and j["id"] not in state
        and j["id"] != current_job_id
        and j.get("createdAt")
        and order_key(j) > watermark
    ]
    return max(newer, key=order_key) if newer else None


# --------------------------------------------------------------------------- #
# Per-job pipeline (streaming, resumable, time-sliced)
# --------------------------------------------------------------------------- #

def process_job(job: dict, session, api_key: str, workdir: Path, john_bin: str,
                attack: list[str], slice_seconds: int,
                upload_interval: int, preempt_interval: int, dry_run: bool,
                state: State, min_hashes: int, selected: set[str], order_key,
                flush_recovery: bool = False, watermark=("", 0)) -> bool:
    """Work one list for up to `slice_seconds` seconds. The John OpenCL format
    and pot matcher are chosen automatically from the job's hash type.
    Returns True  if John exhausted its keyspace (retire the job),
            False if the slice expired mid-attack or a newer job appeared."""
    global _current_john

    job_id = job["id"]
    htype = classify_job(job)
    if htype is None:
        log.info("job %s has unsupported hash type %r, skipping", job_id,
                 job.get("algorithmName"))
        return True
    fmt = john_format_for(htype)
    pot_re = ALGOS[htype]["pot_re"]

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
    algo_id = job.get("algorithmId", DEFAULT_ALGO_ID)

    # Clean up any orphaned John from a previous run.
    reap_stale_john(job_dir, sess)

    # --flush-recovery: drop John's restore/session state so this list restarts
    # its keyspace from scratch, but NEVER touch the pot - already-cracked hashes
    # stay recorded (John skips them on load and we re-report them), so nothing is
    # lost. This forces the fresh-start path below (resuming becomes False).
    if flush_recovery:
        for f in sorted(job_dir.glob("sess.rec*")) + [job_dir / "sess.log"]:
            f.unlink(missing_ok=True)

    resuming = rec_file.exists() and hash_file.exists()
    if resuming:
        n = sum(1 for l in hash_file.read_text().splitlines() if l.strip())
    else:
        # Fresh start: pull the FRESHEST left list for this job - re-fetch its
        # current leftList so hashes cracked recently (by us or anyone on the
        # board) are dropped - then clear stale session state. A normal fresh
        # start also wipes the pot + uploaded record; --flush-recovery keeps both
        # (only the restore state was dropped, the pot is preserved).
        try:
            fresh = fetch_job(session, api_key, job_id)
            if fresh and fresh.get("leftList"):
                left = fresh["leftList"]
        except Exception as exc:
            log.debug("job %s: left-list refresh failed, using scheduled list: %s",
                      job_id, exc)
        n = download_left_list(session, left, hash_file)
        stale_files = [rec_file] if flush_recovery else [rec_file, pot_file,
                                                         uploaded_file]
        for stale in stale_files:
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
                                   read_founds_from_pot(pot_file, pot_re),
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
                newer = check_newer_job(session, api_key, watermark, state,
                                        min_hashes, job_id, selected, order_key)
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
# Combine mode: one big per-algo union list, one John run per hash type
#
# The GPU spends ~all its time generating+hashing candidates; checking them
# against the loaded hashes is an O(1) probe whose cost barely moves with the
# number of hashes. Per-job runs therefore re-sweep the identical keyspace once
# per job. Merging every open job of a type into one hash file sweeps the
# keyspace ONCE and cracks them all together (~N x less GPU work for N jobs).
#
# Submission needs no per-job mapping: POST /api/founds is keyed by algo, so we
# upload the whole per-algo pot and the board routes each hash to its job(s). A
# hash->job map is still kept, but only to retire jobs we've fully cracked.
# --------------------------------------------------------------------------- #

# Degenerate sentinel "hashes" that some left lists carry. The all-ff value in
# particular breaks John's 128-bit perfect-hash table builder: add128() in
# bt_hash_type_128.c overflows on every offset probe ("128 bit add overflow"),
# so the bt builder can never place it collision-free and aborts the whole run
# with "Error building tables ... No. of collisions:1". They are never real
# password hashes, so drop them before they poison the union.
_BOGUS_HASHES = {"f" * 32, "0" * 32}


def _hashes_from_text(text: str) -> set[str]:
    """Bare lowercase 32-hex hashes from a downloaded left list (one per line)."""
    out = set()
    for ln in text.splitlines():
        h = ln.strip().lower()
        if (len(h) == 32 and all(c in "0123456789abcdef" for c in h)
                and h not in _BOGUS_HASHES):
            out.add(h)
    return out


def collect_job_hashes(jobs: list[dict], session, api_key: str, lists_dir: Path,
                       download: bool) -> dict[int, set[str]]:
    """Map each job id -> set of its still-left hashes, caching per-job downloads.

    A cached list is reused when its line count already matches the board's
    leftHashes (the left list IS one bare hash per unfound entry), so a stable
    board re-poll costs no downloads. With download=False (a --restore cycle, job
    set unchanged) we only read existing caches and never hit the network."""
    lists_dir.mkdir(parents=True, exist_ok=True)
    job_hashes: dict[int, set[str]] = {}
    for job in jobs:
        jid = job["id"]
        cache = lists_dir / f"{jid}.txt"
        want = _left_hashes(job)
        fresh_enough = False
        if cache.exists():
            try:
                have = sum(1 for l in cache.read_text().splitlines() if l.strip())
                fresh_enough = (want == 0) or (have == want)
            except Exception:
                fresh_enough = False
        if download and not fresh_enough and job.get("leftList"):
            try:
                download_left_list(session, job["leftList"], cache)
            except Exception as exc:
                log.warning("union: download job %s failed: %s", jid, exc)
        if cache.exists():
            job_hashes[jid] = _hashes_from_text(cache.read_text(errors="replace"))
    return job_hashes


def process_algo_union(htype: str, jobs: list[dict], session, api_key: str,
                       workdir: Path, john_bin: str, attack: list[str],
                       slice_seconds: int, upload_interval: int, dry_run: bool,
                       state: State, flush_recovery: bool) -> None:
    """Crack the union of all `jobs` (already filtered to one hash type) in a
    single John run for up to `slice_seconds`, streaming founds to the board under
    the type's algo id. Retires any job whose every hash is now in the pot.

    Restore vs fresh start is decided on the candidate JOB-ID SET, not the exact
    hashes: while the set is unchanged we --restore so the keyspace keeps
    advancing (we tolerate an existing list shrinking as others crack it - at
    worst we re-offer an already-found hash, which the board simply ignores). When
    a job appears or disappears the union is rebuilt and John fresh-starts (the pot
    is kept, so cracked hashes stay skipped+reported)."""
    global _current_john

    fmt = john_format_for(htype)
    pot_re = ALGOS[htype]["pot_re"]
    algo_id = algo_upload_id(htype)

    algo_dir = (workdir / f"_union_{htype}").resolve()
    lists_dir = algo_dir / "lists"
    algo_dir.mkdir(parents=True, exist_ok=True)
    hash_file = algo_dir / "left.txt"
    pot_file = algo_dir / "union.pot"
    delta_file = algo_dir / "delta.txt"
    uploaded_file = algo_dir / "uploaded_hashes.txt"
    digest_file = algo_dir / "jobset.txt"
    sess = algo_dir / "sess"
    rec_file = algo_dir / "sess.rec"

    reap_stale_john(algo_dir, sess)

    jobset_digest = hashlib.sha1(
        ",".join(str(j["id"]) for j in sorted(jobs, key=lambda x: x["id"]))
        .encode()).hexdigest()
    prev_digest = digest_file.read_text().strip() if digest_file.exists() else ""
    resuming = (not flush_recovery and rec_file.exists()
                and jobset_digest == prev_digest and hash_file.exists())

    # job->hashes for the retire check (and the union build on a fresh start).
    # On resume the set is unchanged, so just read caches - no downloads.
    job_hashes = collect_job_hashes(jobs, session, api_key, lists_dir,
                                    download=not resuming)
    union = set().union(*job_hashes.values()) if job_hashes else set()
    n = len(union)
    if n == 0:
        log.info("union %s: no hashes across %d job(s)", htype, len(jobs))
        return

    if not resuming:
        # Fresh start: write the new union, keep the pot (already-cracked stay
        # skipped+reported), drop stale restore state so John reloads the union.
        hash_file.write_text("\n".join(sorted(union)) + "\n")
        for f in sorted(algo_dir.glob("sess.rec*")) + [algo_dir / "sess.log"]:
            f.unlink(missing_ok=True)
        digest_file.write_text(jobset_digest)

    log.info("union %s: %d hashes from %d job(s)%s", htype, n, len(job_hashes),
             " [resuming]" if resuming else "")

    uploaded: set[str] = set()
    if uploaded_file.exists():
        uploaded = {l.strip().lower()
                    for l in uploaded_file.read_text().splitlines() if l.strip()}

    if resuming:
        cmd = [john_bin, f"--restore={sess}"]
    else:
        cmd = [john_bin, f"--format={fmt}", f"--pot={pot_file}",
               f"--session={sess}", *attack, str(hash_file)]
    log.info("union %s: %s", htype, " ".join(cmd))

    john_log = algo_dir / "john.log"
    errlog = john_log.open("w")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=errlog)
    _current_john = proc
    pidfile = algo_dir / "john.pid"
    pidfile.write_text(str(proc.pid))

    deadline = time.time() + slice_seconds if slice_seconds > 0 else None

    def flush():
        try:
            got = flush_new_founds(session, api_key, algo_id,
                                   read_founds_from_pot(pot_file, pot_re),
                                   uploaded, delta_file, uploaded_file, dry_run)
        except Exception as exc:
            log.warning("union %s: founds upload failed (retry next flush): %s",
                        htype, exc)
            return
        if got:
            verb = "would upload" if dry_run else "uploaded"
            log.info("union %s: %s +%d (uploaded %d/%d)", htype, verb,
                     got, len(uploaded), n)

    try:
        while proc.poll() is None:
            time.sleep(upload_interval)
            flush()
            if (deadline and time.time() > deadline) or _shutdown:
                why = "slice up" if (deadline and time.time() > deadline) else "shutdown"
                log.warning("union %s: %s, SIGINT (John saves session)", htype, why)
                proc.send_signal(signal.SIGINT)
                try:
                    proc.wait(timeout=KILL_GRACE_SECONDS)
                except subprocess.TimeoutExpired:
                    log.warning("union %s: John ignored SIGINT, killing", htype)
                    proc.kill()
                break
        proc.wait()
    finally:
        _current_john = None
        pidfile.unlink(missing_ok=True)
        errlog.close()
        flush()

    try:
        err = john_log.read_text(errors="replace")[-1000:]
    except Exception:
        err = ""
    if proc.returncode not in (0, None) and err.strip():
        log.warning("union %s: John rc=%s stderr tail: %s", htype, proc.returncode, err)

    # Retire jobs whose every hash is now cracked (trims them from future unions).
    cracked = {ln.split(":", 1)[0].lower()
               for ln in read_founds_from_pot(pot_file, pot_re)}
    completed = not rec_file.exists()
    retired = 0
    if not dry_run:
        for jid, hs in job_hashes.items():
            if hs and hs <= cracked and jid not in state:
                state.mark(jid)
                retired += 1
    log.info("union %s: %s; %d/%d cracked this run, retired %d job(s)", htype,
             "COMPLETE" if completed else "sliced",
             len(cracked & union), n, retired)


# --------------------------------------------------------------------------- #
# Main scheduler
# --------------------------------------------------------------------------- #

def scheduler(session, api_key: str, state: State, workdir: Path, john_bin: str,
              attack: list[str], selected: set[str], order: str, min_hashes: int,
              slice_seconds: int, upload_interval: int, preempt_interval: int,
              idle_interval: int, once: bool, dry_run: bool,
              flush_recovery: bool = False, max_lists: int = 0) -> None:
    """Work one list at a time. After every slice (or early abort) re-poll the
    board and pick the next list per choose_next: a fair sweep (least-recently-
    served this run) that, within that, picks the most-preferred list under the
    chosen --order, advancing each slice and restarting once all are served
    (a newcomer that out-ranks everything known at slice start jumps in)."""
    global _shutdown

    order_key = make_order_key(order, workdir)

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
            if classify_job(j) in selected
            and _left_hashes(j) >= min_hashes
            and j["id"] not in state
            and j["id"] not in quarantined
        ]
        if not candidates:
            if _shutdown:
                return
            log.info("no work available for hash type(s): %s",
                     ", ".join(sorted(selected)))
            if once:
                return
            time.sleep(idle_interval)
            continue

        # Watermark (for preemption) is the best of the FULL candidate set, so
        # only a list better than everything on the board can preempt - capping
        # below must not let a lesser list cut in.
        watermark = max((order_key(c) for c in candidates), default=("", 0))
        # --max-lists: confine the rotation to the N most-preferred lists. The
        # set is recomputed each poll; once all N are served choose_next finds
        # them equally-stale and restarts the loop at the most-preferred one.
        if max_lists and len(candidates) > max_lists:
            candidates = sorted(candidates, key=order_key, reverse=True)[:max_lists]
        job = choose_next(candidates, served, order_key)
        served[job["id"]] = time.time()   # slice start: so the next pick advances
                                          # to the next-older list, not this one
        is_resume = _job_dir(workdir, job["id"]).is_dir()
        log.info("picking job %s (created %s) -> %s  [%d candidate(s)]",
                 job["id"], job.get("createdAt", "?"),
                 "RESUME" if is_resume else "NEW", len(candidates))
        try:
            completed = process_job(job, session, api_key, workdir, john_bin,
                                    attack, slice_seconds, upload_interval,
                                    preempt_interval, dry_run, state, min_hashes,
                                    selected, order_key,
                                    flush_recovery=flush_recovery,
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


def scheduler_combine(session, api_key: str, state: State, workdir: Path,
                      john_bin: str, attack: list[str], selected: set[str],
                      min_hashes: int, slice_seconds: int, upload_interval: int,
                      idle_interval: int, once: bool, dry_run: bool,
                      flush_recovery: bool = False, max_lists: int = 0) -> None:
    """Union scheduler: each cycle re-polls the board and gives each selected hash
    type one time-slice as a single merged John run (process_algo_union), so all
    of a type's jobs are cracked in one keyspace sweep. New/finished jobs are
    picked up at the next slice boundary (the union is rebuilt then).

    --order and --preempt-interval do not apply here (there is nothing to choose
    between or preempt - every job of a type is worked at once). --flush-recovery
    only forces a from-scratch restart on the FIRST cycle; later cycles --restore
    while the job set is unchanged so the keyspace keeps advancing."""
    first_flush = flush_recovery
    while not _shutdown:
        try:
            jobs = fetch_jobs(session, api_key)
        except Exception as exc:
            log.error("board poll failed: %s", exc)
            if once or _shutdown:
                return
            time.sleep(idle_interval)
            continue

        worked = False
        for htype in sorted(selected):
            if _shutdown:
                break
            cands = [j for j in jobs
                     if classify_job(j) == htype
                     and _left_hashes(j) >= min_hashes
                     and j["id"] not in state]
            if not cands:
                continue
            # --max-lists: union only the N newest lists of this type.
            if max_lists and len(cands) > max_lists:
                cands = sorted(cands, key=_newest_key, reverse=True)[:max_lists]
            worked = True
            total_left = sum(_left_hashes(j) for j in cands)
            log.info("union %s: %d job(s), %d hashes left on board", htype,
                     len(cands), total_left)
            try:
                process_algo_union(htype, cands, session, api_key, workdir,
                                   john_bin, attack, slice_seconds,
                                   upload_interval, dry_run, state,
                                   flush_recovery=first_flush)
            except requests.HTTPError as exc:
                log.error("union %s HTTP error: %s", htype, exc)
            except Exception as exc:
                log.exception("union %s failed: %s", htype, exc)
        first_flush = False   # only the first cycle honours --flush-recovery

        if not worked:
            if _shutdown or once:
                return
            log.info("no work available for hash type(s): %s",
                     ", ".join(sorted(selected)))
            time.sleep(idle_interval)
            continue
        if once:
            return   # one slice per type, then stop (use --slice 0 to drain fully)


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

    # --- selection / attack ---------------------------------------------- #
    p.add_argument("--hash-type", dest="hash_type", default="both",
                   choices=["md5", "ntlm", "both"],
                   help="which escrow hash type(s) to work; the John OpenCL "
                        "format is then chosen automatically per job "
                        "(md5 -> raw-md5-opencl, ntlm -> NT-opencl). "
                        "(default: both)")
    p.add_argument("--order", default="date",
                   choices=["date", "left", "cracked"],
                   help="crack order for picking the next list: "
                        "'date' = newest first; "
                        "'left' = most hashes still left first; "
                        "'cracked' = most keys WE cracked locally (our own pot) "
                        "first. (default: date)")
    p.add_argument("--attack",
                   default="--wordlist=/usr/share/wordlists/rockyou.txt --rules=best64",
                   help="space-separated John attack flags for a fresh start "
                        "(a resumed list continues its saved attack)")
    p.add_argument("--min-hashes", type=int, default=1,
                   help="skip jobs with fewer than N left hashes (default: 1)")
    p.add_argument("--max-lists", type=int, default=0, metavar="N",
                   help="confine the sweep to the N most-preferred lists (per "
                        "--order; newest-first in --combine): loop over just those "
                        "N, then restart looping from the start. The top-N set is "
                        "recomputed each poll, so genuinely newer lists still enter "
                        "(0 = no limit, sweep every list; default: 0)")
    p.add_argument("--flush-recovery", action="store_true",
                   help="when a list is (re)selected, delete John's restore files "
                        "(sess.rec*/sess.log) first and start it from scratch "
                        "instead of --restore. The pot is NEVER deleted, so "
                        "already-cracked hashes are kept (skipped + re-reported).")

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
    p.add_argument("--combine", action="store_true",
                   help="UNION mode: merge ALL open jobs of each hash type into "
                        "one big list and crack them in a single John run per type "
                        "(one keyspace sweep for every hash of that type, instead "
                        "of one sweep per job). Founds are uploaded per algo (the "
                        "board routes each hash to its job), and a job is retired "
                        "once we've cracked all of its hashes. --order and "
                        "--preempt-interval are ignored in this mode. Overrides "
                        "the per-job newest-first sweep; ignored when --job-id is "
                        "given.")
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
    selected = {"md5", "ntlm"} if args.hash_type == "both" else {args.hash_type}

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
        if classify_job(job) is None:
            log.error("job %s has unsupported hash type %r",
                      args.job_id, job.get("algorithmName"))
            return 1
        log.info("single-job mode: job %s (%s, created %s, %s left)",
                 job["id"], job.get("algorithmName", "?"),
                 job.get("createdAt", "?"), job.get("leftHashes", "?"))
        try:
            completed = process_job(
                job, session, args.api_key, args.workdir, args.john, attack,
                args.slice_seconds, args.upload_interval,
                preempt_interval=0, dry_run=args.dry_run, state=state,
                min_hashes=args.min_hashes, selected=selected,
                order_key=make_order_key(args.order, args.workdir),
                flush_recovery=args.flush_recovery)
            if completed and not args.dry_run:
                state.mark(job["id"])
        except KeyboardInterrupt:
            log.info("bye")
        except Exception as exc:
            log.exception("job %s failed: %s", args.job_id, exc)
            return 1
        return 0

    # Union mode: one merged John run per hash type instead of the per-job sweep.
    if args.combine:
        if not args.once:
            log.info("daemon: COMBINE union mode over hash type(s) [%s], one "
                     "merged run per type, %ds slice each (idle re-poll every %ds, "
                     "ctrl-c to stop)", ", ".join(sorted(selected)),
                     args.slice_seconds, args.idle_interval)
        try:
            scheduler_combine(session, args.api_key, state, args.workdir, args.john,
                              attack, selected, args.min_hashes, args.slice_seconds,
                              args.upload_interval, args.idle_interval, args.once,
                              args.dry_run, flush_recovery=args.flush_recovery,
                              max_lists=args.max_lists)
        except KeyboardInterrupt:
            log.info("bye")
        return 0

    if not args.once:
        log.info("daemon: %s sweep over hash type(s) [%s], re-pick after each "
                 "slice (idle re-poll every %ds, ctrl-c to stop)", args.order,
                 ", ".join(sorted(selected)), args.idle_interval)
    try:
        scheduler(session, args.api_key, state, args.workdir, args.john, attack,
                  selected, args.order, args.min_hashes, args.slice_seconds,
                  args.upload_interval, args.preempt_interval, args.idle_interval,
                  args.once, args.dry_run, flush_recovery=args.flush_recovery,
                  max_lists=args.max_lists)
    except KeyboardInterrupt:
        log.info("bye")
    return 0


if __name__ == "__main__":
    sys.exit(main())
