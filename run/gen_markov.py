import sys
import math
from collections import defaultdict

# Quantification des probabilités en "niveaux" (style OMEN).
#   level = round(-log2(P) * MARKOV_LEVEL_SCALE), borné à [0, MARKOV_MAXLEVEL]
# Un niveau BAS = caractère probable ; un niveau HAUT = improbable. Le seuillage
# (threshold) côté C garde les caractères dont level <= cutoff. Ces constantes
# doivent rester synchronisées avec les #define homonymes dans markov_tables.h
# (émis ci-dessous) et leur lecture dans mask.c.
MARKOV_LEVEL_SCALE = 16
MARKOV_MAXLEVEL = 255

def prob_to_level(p):
    """Convertit une probabilité conditionnelle [0,1] en niveau quantifié (uint8)."""
    if p <= 0.0:
        return MARKOV_MAXLEVEL
    lvl = int(round(-math.log2(p) * MARKOV_LEVEL_SCALE))
    if lvl < 0:
        lvl = 0
    if lvl > MARKOV_MAXLEVEL:
        lvl = MARKOV_MAXLEVEL
    return lvl

def generate_tables(wordlist_path):
    # Longueur maximale suivie dans l'histogramme des longueurs (doit valoir
    # MARKOV_MAXLEN côté C). Les mots plus longs sont ignorés pour ce comptage.
    MARKOV_MAXLEN = 256

    # Dictionnaires pour compter les occurrences
    start_counts = defaultdict(int)
    transition_counts = defaultdict(lambda: defaultdict(int))
    # Histogramme des longueurs de mots (pour la pondération par longueur côté C)
    length_counts = defaultdict(int)

    print("[*] Analyse du dictionnaire en cours...", file=sys.stderr)

    # 1. Phase de comptage
    # Utilisation de latin-1 car les wordlists contiennent souvent des bytes non-UTF-8
    with open(wordlist_path, 'r', encoding='latin-1') as f:
        for line in f:
            word = line.strip()
            if not word:
                continue

            # Longueur du mot
            if len(word) < MARKOV_MAXLEN:
                length_counts[len(word)] += 1

            # Premier caractère
            start_counts[ord(word[0])] += 1

            # Transitions (Caractère N -> Caractère N+1)
            for i in range(len(word) - 1):
                prev_c = ord(word[i])
                next_c = ord(word[i+1])
                transition_counts[prev_c][next_c] += 1

    # 2. Tri par probabilité + niveau quantifié, alignés rang par rang.
    #    Renvoie (ranked, levels) : ranked[k] est le caractère de rang k (du plus
    #    probable au moins probable) et levels[k] son niveau -log2(P) quantifié.
    #    Les deux tables côté C (markov_*_nodes/table et markov_*_level) partagent
    #    donc le MÊME indice de rang, ce qui permet au seuillage de couper la
    #    queue improbable sans casser l'ordre best-first.
    def get_ranked_with_levels(counts_dict):
        total = sum(counts_dict.values())
        sorted_chars = sorted(counts_dict.items(), key=lambda x: x[1], reverse=True)

        ranked = []
        levels = []
        for char, count in sorted_chars:
            ranked.append(char)
            levels.append(prob_to_level(count / total) if total > 0 else MARKOV_MAXLEVEL)

        # Combler avec les caractères jamais vus (pour avoir exactement 256 valeurs).
        # Ils sont testés en dernier et reçoivent le niveau maximal (improbable),
        # donc tout seuil fini les élague.
        for i in range(256):
            if i not in counts_dict:
                ranked.append(i)
                levels.append(MARKOV_MAXLEVEL)
        return ranked, levels

    print("[*] Génération du code C...", file=sys.stderr)

    # 3. Écriture du fichier Header C
    print("/* Fichier autogénéré par le générateur Markov Avalanche */")
    print("#ifndef MARKOV_TABLES_H")
    print("#define MARKOV_TABLES_H\n")
    print(f"#define MARKOV_MAXLEN {MARKOV_MAXLEN}\n")

    # Paramètres de quantification des niveaux (probabilité -> level). Le seuillage
    # côté C convertit un seuil de probabilité tau en cutoff de niveau via
    #   cutoff = round(-log2(tau) * MARKOV_LEVEL_SCALE)
    # et garde les caractères dont level <= cutoff.
    print(f"#define MARKOV_LEVEL_SCALE {MARKOV_LEVEL_SCALE}")
    print(f"#define MARKOV_MAXLEVEL {MARKOV_MAXLEVEL}\n")

    # Table des nœuds de départ (markov_start_nodes) + niveaux alignés.
    start_ranked, start_levels = get_ranked_with_levels(start_counts)
    print("unsigned char markov_start_nodes[256] = {")
    print("    " + ", ".join(str(c) for c in start_ranked))
    print("};\n")
    print("unsigned char markov_start_level[256] = {")
    print("    " + ", ".join(str(l) for l in start_levels))
    print("};\n")

    # Table des transitions (markov_table) + niveaux alignés (markov_table_level).
    # On calcule rangs et niveaux ligne par ligne et on émet les deux tables.
    table_ranked = []
    table_levels = []
    for i in range(256):
        if i in transition_counts:
            ranked, levels = get_ranked_with_levels(transition_counts[i])
        else:
            # Fallback absolu si le caractère n'existe jamais comme préfixe :
            # ordre ASCII, tous improbables.
            ranked = list(range(256))
            levels = [MARKOV_MAXLEVEL] * 256
        table_ranked.append(ranked)
        table_levels.append(levels)

    print("unsigned char markov_table[256][256] = {")
    for i in range(256):
        # Petit commentaire pour rendre le C lisible (affiche le char ASCII si imprimable)
        char_label = chr(i) if 32 <= i <= 126 else f"HEX {hex(i)}"
        print(f"    /* {char_label} */ {{ " + ", ".join(str(c) for c in table_ranked[i]) + " },")
    print("};\n")

    print("unsigned char markov_table_level[256][256] = {")
    for i in range(256):
        char_label = chr(i) if 32 <= i <= 126 else f"HEX {hex(i)}"
        print(f"    /* {char_label} */ {{ " + ", ".join(str(l) for l in table_levels[i]) + " },")
    print("};")

    # Histogramme des longueurs : markov_len_count[L] = nombre de mots de
    # longueur L dans le corpus. Le côté C s'en sert pour donner à chaque
    # longueur un temps de génération proportionnel à P(longueur) (round-robin
    # pondéré). Un tableau entièrement nul = pas de données -> round-robin plat.
    print("\nunsigned long long markov_len_count[MARKOV_MAXLEN] = {")
    len_vals = [str(length_counts.get(i, 0)) for i in range(MARKOV_MAXLEN)]
    print("    " + ", ".join(len_vals))
    print("};")

    print("\n#endif /* MARKOV_TABLES_H */")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_markov.py <wordlist.txt> > markov_tables.h", file=sys.stderr)
        sys.exit(1)

    generate_tables(sys.argv[1])
