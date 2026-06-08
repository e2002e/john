import sys
from collections import defaultdict

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

    # 2. Fonction de tri par probabilité
    def get_sorted_ascii(counts_dict):
        # Trier par fréquence (du plus grand au plus petit)
        sorted_chars = sorted(counts_dict.items(), key=lambda x: x[1], reverse=True)
        ranked = [char for char, count in sorted_chars]

        # Combler avec les caractères restants (pour avoir exactement 256 valeurs)
        # Ceux-ci n'apparaissent jamais dans le set d'entraînement, ils seront testés en dernier
        for i in range(256):
            if i not in ranked:
                ranked.append(i)
        return ranked

    print("[*] Génération du code C...", file=sys.stderr)

    # 3. Écriture du fichier Header C
    print("/* Fichier autogénéré par le générateur Markov Avalanche */")
    print("#ifndef MARKOV_TABLES_H")
    print("#define MARKOV_TABLES_H\n")
    print(f"#define MARKOV_MAXLEN {MARKOV_MAXLEN}\n")

    # Table des nœuds de départ (markov_start_nodes)
    print("unsigned char markov_start_nodes[256] = {")
    start_ranked = get_sorted_ascii(start_counts)
    print("    " + ", ".join(str(c) for c in start_ranked))
    print("};\n")

    # Table des transitions (markov_table)
    print("unsigned char markov_table[256][256] = {")
    for i in range(256):
        if i in transition_counts:
            ranked = get_sorted_ascii(transition_counts[i])
        else:
            # Fallback absolu si le caractère n'existe pas dans le dataset
            ranked = list(range(256))

        # Petit commentaire pour rendre le C lisible (affiche le char ASCII si imprimable)
        char_label = chr(i) if 32 <= i <= 126 else f"HEX {hex(i)}"
        print(f"    /* {char_label} */ {{ " + ", ".join(str(c) for c in ranked) + " },")
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
