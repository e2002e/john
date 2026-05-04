#!/usr/bin/env python3
"""
Generate inc2.freq for John the Ripper inc2 mode.
Supports any character set: uppercase, lowercase, digits, specials.
Special letters line is no longer limited to 8 – it contains all distinct
characters found in the wordlist (sorted by frequency) unless you explicitly
set --special-count.
Chains are always complete (all following letters that actually appear).
"""

import argparse
import sys
from collections import Counter, defaultdict
import string

def main():
    p = argparse.ArgumentParser(
        description="Create a John the Ripper inc2.freq file from a wordlist."
    )
    p.add_argument("-w", "--wordlist", required=True, help="Input wordlist")
    p.add_argument("--charset", default=None,
                   help="Full character set to use (default: printable ASCII 0x20-0x7e). "
                        "Example: --charset='abcdefghijklmnopqrstuvwxyz' for a-z only.")
    p.add_argument("--min-len", type=int, default=1)
    p.add_argument("--max-len", type=int, default=20)
    p.add_argument("--special-count", type=int, default=0,
                   help="Number of most frequent special letters to keep. "
                        "0 (default) = all distinct characters that appear.")
    p.add_argument("-o", "--outfile", default="inc2.freq")
    args = p.parse_args()

    # ---- Character set ----
    if args.charset is not None:
        charset = args.charset
    else:
        # all printable ASCII (space … ~)
        charset = ''.join(chr(i) for i in range(32, 127))

    # Ensure charset has no duplicates and a fixed order
    seen = set()
    fixed_charset = []
    for c in charset:
        if c not in seen:
            seen.add(c)
            fixed_charset.append(c)
    charset = ''.join(fixed_charset)
    charset_set = set(charset)

    # ---- Read and filter words ----
    with open(args.wordlist, encoding="utf-8", errors="ignore") as f:
        raw = f.read().splitlines()

    words = []
    for w in raw:
        w = w.strip()
        # keep only characters that are in the defined charset
        filtered = ''.join(c for c in w if c in charset_set)
        if args.min_len <= len(filtered) <= args.max_len:
            words.append(filtered)

    if not words:
        sys.exit("No valid words found (characters must be from the defined charset).")

    # ---- Global frequencies -> special letters ----
    global_freq = Counter()
    for w in words:
        global_freq.update(w)

    # Sort all appearing characters by frequency, then by charset order for ties.
    appearing = [c for c in charset if c in global_freq]
    appearing.sort(key=lambda c: (-global_freq[c], charset.index(c)))

    if args.special_count <= 0:
        special_letters = appearing
    else:
        special_letters = appearing[:args.special_count]

    # ---- Positional and bigram frequencies ----
    maxlen = args.max_len
    pos_freq = [Counter() for _ in range(maxlen)]
    bigram = [defaultdict(Counter) for _ in range(maxlen - 1)]

    for w in words:
        for i, ch in enumerate(w):
            if i < maxlen:
                pos_freq[i].update(ch)
            if i > 0 and i < maxlen:
                bigram[i - 1][w[i - 1]].update(w[i])

    # ---- Base orders for each position (full charset) ----
    base_order = []
    for i in range(maxlen):
        # Frequency-sorted characters that appear at this position
        freq_sorted = [c for c, _ in pos_freq[i].most_common() if c in charset_set]
        # Append missing charset characters in their original charset order
        present = set(freq_sorted)
        for c in charset:
            if c not in present:
                freq_sorted.append(c)
        base_order.append(''.join(freq_sorted))

    # ---- Chains and counters (always complete) ----
    chain_rows = []
    counter_rows = []
    for L in range(1, maxlen):
        row_chains = []
        row_counters = []
        for sp in special_letters:
            if sp in bigram[L - 1]:
                counts = bigram[L - 1][sp]
                # All following letters that appear, sorted by frequency
                sorted_letters = [c for c, _ in counts.most_common() if c in charset_set]
                chain = sorted_letters                 # complete chain
                # Counter = charset letters not in the chain, in charset order
                chain_set = set(chain)
                counter = [c for c in charset if c not in chain_set]
                row_chains.append(''.join(chain))
                row_counters.append(''.join(counter))
            else:
                # Special letter never appears at this position
                row_chains.append('')
                row_counters.append(charset)
        chain_rows.append(row_chains)
        counter_rows.append(row_counters)

    # ---- Write the .freq file ----
    with open(args.outfile, "w") as f:
        f.write(f"{maxlen}\n")
        f.write(f"{''.join(special_letters)}\n")
        for i in range(maxlen):
            f.write(f"{base_order[i]}\n")
        for L in range(maxlen - 1):
            for sp_idx in range(len(special_letters)):
                chain = chain_rows[L][sp_idx]
                counter = counter_rows[L][sp_idx]
                f.write(f"{len(chain)}\n")
                if chain:
                    f.write(f"{chain}\n")
                else:
                    f.write("\n")
                f.write(f"{counter}\n")

    print(f"Wrote {args.outfile}: maxlen={maxlen}, "
          f"special letters ({len(special_letters)}): {''.join(special_letters)}")

if __name__ == "__main__":
    main()
