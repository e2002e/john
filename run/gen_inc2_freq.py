#!/usr/bin/env python3
"""
Generate inc2.freq for John the Ripper inc2 mode (trigram‑aware).
Adds optional trigram chains after the bigram section.
"""

import argparse
import sys
from collections import Counter, defaultdict

def main():
    p = argparse.ArgumentParser(
        description="Create a John the Ripper inc2.freq file with trigram chains."
    )
    p.add_argument("-w", "--wordlist", required=True, help="Input wordlist")
    p.add_argument("--charset", default=None,
                   help="Full character set (default: printable ASCII 32-126)")
    p.add_argument("--min-len", type=int, default=1)
    p.add_argument("--max-len", type=int, default=20)
    p.add_argument("--special-count", type=int, default=0,
                   help="Number of most frequent special letters (0=all)")
    p.add_argument("--trigram-count", type=int, default=0,
                   help="Number of most frequent bigrams to keep for trigram chains (0=all)")
    p.add_argument("-o", "--outfile", default="inc2.freq")
    args = p.parse_args()

    # ---- Character set ----
    if args.charset is not None:
        charset = args.charset
    else:
        charset = ''.join(chr(i) for i in range(32, 127))
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
        filtered = ''.join(c for c in w if c in charset_set)
        if args.min_len <= len(filtered) <= args.max_len:
            words.append(filtered)
    if not words:
        sys.exit("No valid words found.")

    # ---- Global frequencies -> special letters ----
    global_freq = Counter()
    for w in words:
        global_freq.update(w)
    appearing = [c for c in charset if c in global_freq]
    appearing.sort(key=lambda c: (-global_freq[c], charset.index(c)))
    if args.special_count <= 0:
        special_letters = appearing
    else:
        special_letters = appearing[:args.special_count]

    # ---- Positional and bigram/trigram frequencies ----
    maxlen = args.max_len
    pos_freq = [Counter() for _ in range(maxlen)]
    bigram = [defaultdict(Counter) for _ in range(maxlen - 1)]
    trigram = [defaultdict(Counter) for _ in range(2, maxlen)]  # index = L-2

    for w in words:
        for i, ch in enumerate(w):
            if i < maxlen:
                pos_freq[i].update(ch)
            if i > 0 and i < maxlen:
                bigram[i - 1][w[i - 1]].update(w[i])
            if i > 1 and i < maxlen:
                trigram[i - 2][(w[i - 2], w[i - 1])].update(w[i])

    # ---- Base orders (full charset) ----
    base_order = []
    for i in range(maxlen):
        freq_sorted = [c for c, _ in pos_freq[i].most_common() if c in charset_set]
        present = set(freq_sorted)
        for c in charset:
            if c not in present:
                freq_sorted.append(c)
        base_order.append(''.join(freq_sorted))

    # ---- Bigram chains ----
    chain_rows = []
    counter_rows = []
    for L in range(1, maxlen):
        row_chains = []
        row_counters = []
        for sp in special_letters:
            if sp in bigram[L - 1]:
                counts = bigram[L - 1][sp]
                sorted_letters = [c for c, _ in counts.most_common() if c in charset_set]
                chain = sorted_letters
                chain_set = set(chain)
                counter = [c for c in charset if c not in chain_set]
                row_chains.append(''.join(chain))
                row_counters.append(''.join(counter))
            else:
                row_chains.append('')
                row_counters.append(charset)
        chain_rows.append(row_chains)
        counter_rows.append(row_counters)

    # ---- Trigram bigram selection ----
    # Collect all distinct bigrams (both chars in charset) and their total frequency
    bigram_total_freq = Counter()
    for L in range(2, maxlen):
        for (c1, c2), cnt in trigram[L - 2].items():
            if c1 in charset_set and c2 in charset_set:
                bigram_total_freq[(c1, c2)] += sum(cnt.values())  # sum across positions? Actually sum total trigram count
    # Sort by total frequency
    sorted_bigrams = sorted(bigram_total_freq, key=lambda x: -bigram_total_freq[x])
    if args.trigram_count <= 0:
        special_bigrams = sorted_bigrams
    else:
        special_bigrams = sorted_bigrams[:args.trigram_count]

    # Map bigram tuple to string for output
    bigram_str_list = [''.join(pair) for pair in special_bigrams]

    # ---- Trigram chains for each position L >= 2 ----
    trigram_chain_rows = []
    trigram_counter_rows = []
    for L in range(2, maxlen):
        chains = []
        counters = []
        for pair in special_bigrams:
            if pair in trigram[L - 2]:
                counts = trigram[L - 2][pair]
                sorted_letters = [c for c, _ in counts.most_common() if c in charset_set]
                chain = sorted_letters
                chain_set = set(chain)
                counter = [c for c in charset if c not in chain_set]
                chains.append(''.join(chain))
                counters.append(''.join(counter))
            else:
                chains.append('')
                counters.append(charset)
        trigram_chain_rows.append(chains)
        trigram_counter_rows.append(counters)

    # ---- Write .freq file ----
    with open(args.outfile, "w") as f:
        f.write(f"{maxlen}\n")
        f.write(f"{''.join(special_letters)}\n")
        for i in range(maxlen):
            f.write(f"{base_order[i]}\n")
        # bigram section
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
        # trigram section header
        f.write(f"TRIGRAM {len(special_bigrams)}\n")
        for bigram_str in bigram_str_list:
            f.write(f"{bigram_str}\n")
        # trigram chains for L=2..maxlen-1
        for L_idx in range(maxlen - 2):
            for sp_bigram_idx in range(len(special_bigrams)):
                chain = trigram_chain_rows[L_idx][sp_bigram_idx]
                counter = trigram_counter_rows[L_idx][sp_bigram_idx]
                f.write(f"{len(chain)}\n")
                if chain:
                    f.write(f"{chain}\n")
                else:
                    f.write("\n")
                f.write(f"{counter}\n")

    print(f"Wrote {args.outfile}: maxlen={maxlen}, "
          f"special letters ({len(special_letters)}): {''.join(special_letters)}, "
          f"trigram bigrams ({len(special_bigrams)})")

if __name__ == "__main__":
    main()
