/*
 * inc2.c – Block‑interleaved lengths & first characters.
 *          First character rotates every BLOCK_SIZE words.
 *          Suffix digits now also rotate quickly by using a large
 *          coprime increment, so no single placeholder stays fixed
 *          for long.
 *
 * Full version: dynamic character set, arbitrary special letters,
 *               frequency tables loaded from a file.
 */
#include "os.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include "arch.h"
#include "john.h"
#include "loader.h"
#include "cracker.h"
#include "options.h"
#include "config.h"
#include "logger.h"
#include "status.h"
#include "signals.h"
#include "recovery.h"
#include "mask.h"

int inc2_cur_len;

#if JTR_HAVE_INT128
typedef uint128_t uint_big;
#define UINT_BIG_MAX UINT128_MAX
#else
typedef uint64_t uint_big;
#define UINT_BIG_MAX UINT64_MAX
#endif

#define MAX_CAND_LENGTH PLAINTEXT_BUFFER_SIZE
#define DEFAULT_MAX_LEN 16
#define INTERLEAVE_STRIDE 100000   /* words generated per (L,first) pair per iteration */
#define INTERLEAVE_SUB_STRIDE 10000   /* words generated per (L,sc) pair per iteration */

char word[PLAINTEXT_BUFFER_SIZE];

/* ------------------------------------------------------------------------- */
/*  Dynamic tables – will be rebuilt when a frequency file is loaded         */
/* ------------------------------------------------------------------------- */
struct inc2_tables {
    int charset_sz;                     /* number of distinct characters in the alphabet */
    int maxlength;
    int num_special;                    /* number of special letters */
    char *special_letters;              /* string of length num_special */
    char **base_order;                  /* [0..maxlen-1] each a permutation of the full charset */
    char ***chainFreq;                  /* [L-1][sp] -> chain string */
    char ***counterChainFreq;           /* [L-1][sp] -> counter string */
    char ***perm;                       /* [pos][first_idx] -> full permutation for that context */
    /* character mapping */
    int char_to_index[256];             /* ASCII -> 0..charset_sz-1, -1 if not in charset */
    char index_to_char[256];            /* 0..charset_sz-1 -> ASCII */
};

static struct inc2_tables tables;       /* active tables (filled by loader or defaults) */

/* Default 26‑letter tables – same as original */
static const char *default_base[] = {
    "taoiswcbpfmhdrenlguvykxjqz", "hoeanirfutslpcmdybvwgxkjqz",
    "ertadsonilmcupvgwyfbhkxjqz", "etirnlsaodhcmupgywkvfbxjqz",
    "eritsnloaudchgmypvkbfwxjqz", "enrsitadlcogmyhuvpwfbkxjqz",
    "etsinarldogycmuhpfbvwkxjqz", "eraciostpnlmuhfdgbwvxqzjky",
    "ersaiotclnpumfhgbdvwyxkjqz", "etarioscnluwpmhgdfbvykxjqz",
    "tiarhoeslwnmcufdbpgvykxjqz", "itaeohnrulswcdfbmvgpykxjqz",
    "enohaitrlscumgdbwvkxpfyjqz", "esdntyrfolgahmwcpibkuvxjqz",
    "etaoinsrhldcumfpgwybvkxjqz"
};

static const char *chainFreq_default[][8] = {
    {"hoei","ei","ontsc","anrsd","nrtsl","eai","nrfum","etdg"},
    {"eoih","ei","tsonc","radsn","rtsnl","eai","rnmuf","etdg"},
    {"eioh","ei","tnsoc","rnsad","trnls","eia","rnmuf","etdg"},
    {"eioh","ei","tsnoc","rsnad","rtsnl","eia","rnumf","etdg"},
    {"eioh","ei","nstco","nrsad","nrstl","eia","nrmuf","etdg"},
    {"eioh","ei","tsnoc","snard","tsnrl","eia","nrmuf","etdg"},
    {"eioh","ei","costn","rasnd","rstnl","eai","rnmuf","etdg"},
    {"eioh","ei","sotcn","rsand","rstln","eai","rnumf","etgd"},
    {"eioh","ei","toscn","arsnd","trsnl","eai","rnumf","etgd"},
    {"ihoe","ie","tosnc","arsnd","trsln","iae","rnmuf","tedg"},
    {"ieoh","ie","tonsc","anrsd","tnrls","iae","nrufm","tedg"},
    {"eohi","ei","notsc","narsd","ntrls","eai","nrumf","etgd"},
    {"eohi","ei","sntoc","sdnra","sntrl","eai","nrfmu","edtg"},
    {"eoih","ei","tonsc","ansrd","tnsrl","eai","nrumf","etdg"}
};

static const char *counterChainFreq_default[][8] = {
    {"anrfutslpcmdybvwgxkjqz","hoanrfutslpcmdybvwgxkjqz","heairfulpmdybvwgxkjqz","hoeifutlpcmybvwgxkjqz","hoeaifupcmdybvwgxkjqz","honrfutslpcmdybvwgxkjqz","hoeaitslpcdybvwgxkjqz","hoanirfuslpcmybvwxkjqz"},
    {"rtadsnlmcupvgwyfbkxjqz","rtadsonlmcupvgwyfbhkxjqz","eradilmupvgwyfbhkxjqz","etoilmcupvgwyfbhkxjqz","eadoimcupvgwyfbhkxjqz","rtdsonlmcupvgwyfbhkxjqz","etadsoilcpvgwybhkxjqz","rasonilmcupvwyfbhkxjqz"},
    {"trnlsadcmupgywkvfbxjqz","trnlsaodhcmupgywkvfbxjqz","eirladhmupgywkvfbxjqz","etilohcmupgywkvfbxjqz","eiaodhcmupgywkvfbxjqz","trnlsodhcmupgywkvfbxjqz","etilsaodhcpgywkvbxjqz","irnlsaohcmupywkvfbxjqz"},
    {"rtsnlaudcgmypvkbfwxjqz","rtsnloaudchgmypvkbfwxjqz","erilaudhgmypvkbfwxjqz","eitlouchgmypvkbfwxjqz","eioaudchgmypvkbfwxjqz","rtsnloudchgmypvkbfwxjqz","eitsloadchgypvkbwxjqz","risnloauchmypvkbfwxjqz"},
    {"nrstadlcgmyuvpwfbkxjqz","nrstadlcogmyhuvpwfbkxjqz","eriadlgmyhuvpwfbkxjqz","eitlcogmyhuvpwfbkxjqz","eiadcogmyhuvpwfbkxjqz","nrstdlcogmyhuvpwfbkxjqz","esitadlcogyhvpwbkxjqz","nrsialcomyhuvpwfbkxjqz"},
    {"tsnarldgycmupfbvwkxjqz","tsnarldogycmuhpfbvwkxjqz","eiarldgymuhpfbvwkxjqz","etilogycmuhpfbvwkxjqz","eiadogycmuhpfbvwkxjqz","tsnrldogycmuhpfbvwkxjqz","etsialdogychpbvwkxjqz","sinarloycmuhpfbvwkxjqz"},
    {"racstpnlmufdgbwvxqzjky","racostpnlmuhfdgbwvxqzjky","eraiplmuhfdgbwvxqzjky","eciotplmuhfgbwvxqzjky","eaciopmuhfdgbwvxqzjky","rcostpnlmuhfdgbwvxqzjky","eaciostplhdgbwvxqzjky","raciospnlmuhfbwvxqzjky"},
    {"rsatclnpumfgbdvwyxkjqz","rsaotclnpumfhgbdvwyxkjqz","erailpumfhgbdvwyxkjqz","eiotclpumfhgbvwyxkjqz","eaiocpumfhgbdvwyxkjqz","rsotclnpumfhgbdvwyxkjqz","esaiotclphgbdvwyxkjqz","rsaioclnpumfhbvwyxkjqz"},
    {"tarscnluwpmgdfbvykxjqz","taroscnluwpmhgdfbvykxjqz","eariluwpmhgdfbvykxjqz","etiocluwpmhgfbvykxjqz","eaiocuwpmhgdfbvykxjqz","troscnluwpmhgdfbvykxjqz","etaiosclwphgdbvykxjqz","arioscnluwpmhfbvykxjqz"},
    {"tarslwnmcufdbpgvykxjqz","tarhoslwnmcufdbpgvykxjqz","iarhelwmufdbpgvykxjqz","tihoelwmcufbpgvykxjqz","iahoewmcufdbpgvykxjqz","trhoslwnmcufdbpgvykxjqz","tiahoeslwcdbpgvykxjqz","iarhoslwnmcufbpvykxjqz"},
    {"tanrulswcdfbmvgpykxjqz","taohnrulswcdfbmvgpykxjqz","iaehrulwdfbmvgpykxjqz","iteohulwcfbmvgpykxjqz","iaeohuwcdfbmvgpykxjqz","tohnrulswcdfbmvgpykxjqz","itaeohlswcdbvgpykxjqz","iaohnrulswcfbmvpykxjqz"},
    {"natrlscumgdbwvkxpfyjqz","nohatrlscumgdbwvkxpfyjqz","ehairlumgdbwvkxpfyjqz","eohitlcumgbwvkxpfyjqz","eohaicumgdbwvkxpfyjqz","nohtrlscumgdbwvkxpfyjqz","eohaitlscgdbwvkxpyjqz","nohairlscumbwvkxpfyjqz"},
    {"sdntyrflgamwcpbkuvxjqz","sdntyrfolgahmwcpbkuvxjqz","edyrflgahmwpibkuvxjqz","etyfolghmwcpibkuvxjqz","edyfogahmwcpibkuvxjqz","sdntyrfolghmwcpbkuvxjqz","esdtyolgahwcpibkvxjqz","snyrfolahmwcpibkuvxjqz"},
    {"tansrldcumfpgwybvkxjqz","taonsrhldcumfpgwybvkxjqz","eairhldumfpgwybvkxjqz","etoihlcumfpgwybvkxjqz","eaoihdcumfpgwybvkxjqz","tonsrhldcumfpgwybvkxjqz","etaoishldcpgwybvkxjqz","aoinsrhlcumfpwybvkxjqz"}
};

static int **pair_exhausted;        /* kept as before */
static double *pair_total;          /* total suffixes for this (L,first) pair (for progress) */
static int *pair_exhausted_cnt;     /* count of exhausted pairs (for progress) */

static struct {
    uint8_t ***tails;       // [first][sc][tail_pos]  (L-2 éléments par sc)
    int **exhausted;        // [first][sc]
} per_len[MAX_CAND_LENGTH];

static int minlength, maxlength, total_pairs;
static int state_restored = 0;
static uint_big set = 0;              /* candidates generated by this node */

/* Node splitting */
static int node_id = 1;
static int node_count = 1;

/* ------------------------------------------------------------------------- */
/*  Helper: allocate a string and copy                                       */
/* ------------------------------------------------------------------------- */
static char *my_strdup(const char *s)
{
    char *d = malloc(strlen(s) + 1);
    if (d) strcpy(d, s);
    return d;
}

/* ------------------------------------------------------------------------- */
/*  Free all dynamically allocated tables                                    */
/* ------------------------------------------------------------------------- */
static void free_tables(struct inc2_tables *t)
{
    int i, j;
    if (!t) return;
    if (t->base_order) {
        for (i = 0; i < t->maxlength; i++)
            free(t->base_order[i]);
        free(t->base_order);
    }
    if (t->chainFreq) {
        for (i = 0; i < t->maxlength - 1; i++) {
            for (j = 0; j < t->num_special; j++)
                free(t->chainFreq[i][j]);
            free(t->chainFreq[i]);
        }
        free(t->chainFreq);
    }
    if (t->counterChainFreq) {
        for (i = 0; i < t->maxlength - 1; i++) {
            for (j = 0; j < t->num_special; j++)
                free(t->counterChainFreq[i][j]);
            free(t->counterChainFreq[i]);
        }
        free(t->counterChainFreq);
    }
    if (t->perm) {
        for (i = 0; i < t->maxlength; i++) {
            for (j = 0; j < t->charset_sz; j++)
                free(t->perm[i][j]);
            free(t->perm[i]);
        }
        free(t->perm);
    }
    free(t->special_letters);
    memset(t, 0, sizeof(*t));
}

/* ------------------------------------------------------------------------- */
/*  Build character mapping from charset string                            */
/* ------------------------------------------------------------------------- */
static void build_charmap(struct inc2_tables *t, const char *charset_str)
{
    int i;
    memset(t->char_to_index, -1, sizeof(t->char_to_index));
    for (i = 0; i < t->charset_sz; i++) {
        unsigned char c = (unsigned char)charset_str[i];
        t->char_to_index[c] = i;
        t->index_to_char[i] = c;
    }
}

/* ------------------------------------------------------------------------- */
/*  Load frequency file and populate tables                                  */
/* ------------------------------------------------------------------------- */
static int load_freq_from_file(const char *fname)
{
    FILE *fp = fopen(fname, "r");
    if (!fp) {
        fprintf(stderr, "inc2: cannot open '%s', using defaults\n", fname);
        return 0;
    }

    int file_maxlen, i, L, sp;
    char line[8192];
    int charset_sz;

    /* 1. maxlen */
    if (!fgets(line, sizeof(line), fp) || sscanf(line, "%d", &file_maxlen) != 1) {
        fprintf(stderr, "inc2: bad maxlen in freq file\n");
        fclose(fp);
        return 0;
    }
    if (file_maxlen < 1 || file_maxlen > MAX_CAND_LENGTH) {
        fprintf(stderr, "inc2: maxlen %d out of range, using defaults\n", file_maxlen);
        fclose(fp);
        return 0;
    }

    /* 2. special letters string */
    if (!fgets(line, sizeof(line), fp)) {
        fclose(fp);
        return 0;
    }
    line[strcspn(line, "\r\n")] = 0;
    int num_special = strlen(line);
    if (num_special < 1) {
        fprintf(stderr, "inc2: empty special letters\n");
        fclose(fp);
        return 0;
    }

    /* 3. first base_order row → determines charset size */
    if (!fgets(line, sizeof(line), fp)) {
        fclose(fp);
        return 0;
    }
    line[strcspn(line, "\r\n")] = 0;
    charset_sz = strlen(line);
    if (charset_sz < 1 || charset_sz > 256) {
        fprintf(stderr, "inc2: charset size %d invalid\n", charset_sz);
        fclose(fp);
        return 0;
    }

    struct inc2_tables new_tables;
    memset(&new_tables, 0, sizeof(new_tables));
    new_tables.maxlength = file_maxlen;
    new_tables.charset_sz = charset_sz;
    new_tables.num_special = num_special;
    new_tables.special_letters = my_strdup(line);

    /* Allocate base_order rows */
    new_tables.base_order = malloc(file_maxlen * sizeof(char *));
    for (i = 0; i < file_maxlen; i++)
        new_tables.base_order[i] = malloc(charset_sz + 1);

    /* Allocate chain/counter arrays */
    new_tables.chainFreq = malloc((file_maxlen - 1) * sizeof(char **));
    new_tables.counterChainFreq = malloc((file_maxlen - 1) * sizeof(char **));
    for (L = 0; L < file_maxlen - 1; L++) {
        new_tables.chainFreq[L] = malloc(num_special * sizeof(char *));
        new_tables.counterChainFreq[L] = malloc(num_special * sizeof(char *));
        for (sp = 0; sp < num_special; sp++) {
            new_tables.chainFreq[L][sp] = NULL;
            new_tables.counterChainFreq[L][sp] = NULL;
        }
    }

    /* Re-read from start */
    fseek(fp, 0, SEEK_SET);
    if (!fgets(line, sizeof(line), fp)) { fclose(fp); goto fail; }
    if (!fgets(line, sizeof(line), fp)) { fclose(fp); goto fail; }
    line[strcspn(line, "\r\n")] = 0;
    free(new_tables.special_letters);
    new_tables.special_letters = my_strdup(line);

    fclose(fp);
    fp = fopen(fname, "r");
    if (!fp) goto fail;
    fgets(line, sizeof(line), fp);   /* maxlen */
    fgets(line, sizeof(line), fp);   /* special letters */
    fgets(line, sizeof(line), fp);   /* first base order row */
    line[strcspn(line, "\r\n")] = 0;
    build_charmap(&new_tables, line);

    /* Read all base_order rows */
    strcpy(new_tables.base_order[0], line);
    for (i = 1; i < file_maxlen; i++) {
        if (!fgets(line, sizeof(line), fp)) goto fail;
        line[strcspn(line, "\r\n")] = 0;
        if ((int)strlen(line) != charset_sz) {
            fprintf(stderr, "inc2: base_order row %d length mismatch\n", i);
            goto fail;
        }
        strcpy(new_tables.base_order[i], line);
    }

    /* Read chains for L=1 .. file_maxlen-1 */
    for (L = 1; L < file_maxlen; L++) {
        for (sp = 0; sp < num_special; sp++) {
            int chain_len;
            if (!fgets(line, sizeof(line), fp) || sscanf(line, "%d", &chain_len) != 1) {
                fprintf(stderr, "inc2: bad chain_len at L=%d sp=%d\n", L, sp);
                goto fail;
            }
            if (chain_len > 0) {
                if (!fgets(line, sizeof(line), fp)) goto fail;
                line[strcspn(line, "\r\n")] = 0;
                if ((int)strlen(line) != chain_len) {
                    fprintf(stderr, "inc2: chain length mismatch L=%d sp=%d\n", L, sp);
                    goto fail;
                }
                new_tables.chainFreq[L-1][sp] = my_strdup(line);
            } else {
                if (!fgets(line, sizeof(line), fp)) goto fail;
                new_tables.chainFreq[L-1][sp] = my_strdup("");
            }
            /* counter */
            if (!fgets(line, sizeof(line), fp)) goto fail;
            line[strcspn(line, "\r\n")] = 0;
            int expected_counter_len = charset_sz - chain_len;
            if ((int)strlen(line) != expected_counter_len) {
                fprintf(stderr, "inc2: counter length mismatch L=%d sp=%d (got %zu, expected %d)\n",
                        L, sp, strlen(line), expected_counter_len);
                goto fail;
            }
            new_tables.counterChainFreq[L-1][sp] = my_strdup(line);
        }
    }
    fclose(fp);

    free_tables(&tables);
    memcpy(&tables, &new_tables, sizeof(tables));

    if (file_maxlen < maxlength) {
        fprintf(stderr, "inc2: reducing maxlen from %d to %d\n", maxlength, file_maxlen);
        maxlength = file_maxlen;
        if (minlength > maxlength) minlength = maxlength;
    }
    return 1;

fail:
    fclose(fp);
    free_tables(&new_tables);
    return 0;
}

/* ------------------------------------------------------------------------- */
/*  Build hardcoded 26‑letter tables                                         */
/* ------------------------------------------------------------------------- */
static void build_default_tables(void)
{
    struct inc2_tables t;
    memset(&t, 0, sizeof(t));
    t.charset_sz = 26;
    t.maxlength = maxlength;
    t.num_special = 8;
    t.special_letters = my_strdup("thiearon");

    build_charmap(&t, "abcdefghijklmnopqrstuvwxyz");

    int i, sp;
    t.base_order = malloc(maxlength * sizeof(char *));
    for (i = 0; i < maxlength; i++)
        t.base_order[i] = malloc(27);
    for (i = 0; i < maxlength; i++) {
        const char *src = (i < 15) ? default_base[i] : default_base[14];
        strcpy(t.base_order[i], src);
    }
    t.chainFreq = malloc((maxlength - 1) * sizeof(char **));
    t.counterChainFreq = malloc((maxlength - 1) * sizeof(char **));
    int row;
    for (i = 1; i < maxlength; i++) {
        row = i - 1;
        if (row >= (int)(sizeof(chainFreq_default)/sizeof(chainFreq_default[0])))
            row = (int)(sizeof(chainFreq_default)/sizeof(chainFreq_default[0])) - 1;
        t.chainFreq[i-1] = malloc(t.num_special * sizeof(char *));
        t.counterChainFreq[i-1] = malloc(t.num_special * sizeof(char *));
        for (sp = 0; sp < t.num_special; sp++) {
            t.chainFreq[i-1][sp] = my_strdup(chainFreq_default[row][sp]);
            t.counterChainFreq[i-1][sp] = my_strdup(counterChainFreq_default[row][sp]);
        }
    }

    free_tables(&tables);
    memcpy(&tables, &t, sizeof(tables));
}

/* ------------------------------------------------------------------------- */
/*  Build a full permutation for position 'pos' and previous char 'prev'     */
/* ------------------------------------------------------------------------- */
static void build_permutation(int pos, int prev_idx, char *out)
{
    int class = -1;
    char prev_char = tables.index_to_char[prev_idx];
    int sp;
    for (sp = 0; sp < tables.num_special; sp++) {
        if (tables.special_letters[sp] == prev_char) {
            class = sp;
            break;
        }
    }
    if (class == -1) {
        strcpy(out, tables.base_order[pos]);
        return;
    }
    const char *chain = tables.chainFreq[pos-1][class];
    const char *counter = tables.counterChainFreq[pos-1][class];
    int chain_len = strlen(chain);
    memcpy(out, chain, chain_len);
    strcpy(out + chain_len, counter);
}

/* ------------------------------------------------------------------------- */
/*  Init all permutations for every (position, first char) combination       */
/* ------------------------------------------------------------------------- */
static void init_permutations(void)
{
    int i, c;
    tables.perm = malloc(tables.maxlength * sizeof(char **));
    for (i = 0; i < tables.maxlength; i++) {
        tables.perm[i] = malloc(tables.charset_sz * sizeof(char *));
        for (c = 0; c < tables.charset_sz; c++) {
            tables.perm[i][c] = malloc(tables.charset_sz + 1);
            if (i == 0) {
                strcpy(tables.perm[i][c], tables.base_order[0]);
            } else {
                build_permutation(i, c, tables.perm[i][c]);
            }
        }
    }
}

/*
 * Advance suffix 'digits' of length 'd' (most-significant first)
 * by the repunit (111...1) modulo base^d.
 * Returns 1 if the suffix wrapped to all zeros (i.e., exhausted),
 * 0 otherwise.
 */
/*
 * Standard odometer: increment least significant digit first.
 * Digits are stored most-significant first (index 0 = first suffix char).
 * Returns 1 if all digits wrapped to 0 (exhausted), 0 otherwise.
 */
static int suffix_add_one(uint8_t *digits, int d, int base)
{
	int i;
    for (i = d - 1; i >= 0; i--) {
        if (digits[i] < base - 1) {
            digits[i]++;
            return 0;   /* no wrap */
        }
        digits[i] = 0;
    }
    return 1;   /* all digits back to 0 -> exhausted */
}

/* ------------------------------------------------------------------------- */
/*  Build candidate word from first char and suffix digits                   */
/* ------------------------------------------------------------------------- */
static void build_word(int L, int first_idx, uint8_t *suffix_digits, char *out)
{
    int i;
    out[0] = tables.base_order[0][first_idx];
    for (i = 1; i < L; i++) {
        int prev = tables.char_to_index[(unsigned char)out[i-1]];
        out[i] = tables.perm[i][prev][ suffix_digits[i-1] ];
    }
    out[L] = '\0';
}

/* ------------------------------------------------------------------------- */
/*  Progress – node's fraction of total candidates                           */
/* ------------------------------------------------------------------------- */
static double get_progress(void)
{
    static double node_total = -1.0;
    if (node_total < 0.0) {
        node_total = 0.0;
        for (int L = minlength; L <= maxlength; L++) {
            /* charset_sz^L may overflow double, but we are safe up to ~300 chars lengths */
            double space = pow((double)tables.charset_sz, (double)L);
            node_total += space;
        }
        node_total /= node_count;
    }
    if (node_total == 0.0) return -1.0;
    double done = (double)set / node_total;
    return (done > 1.0) ? 1.0 : done;
}

static void save_state(FILE *file)
{
    fprintf(file, "%d\n%d\n%d\n", minlength, maxlength, tables.charset_sz);
    fprintf(file, "%llu\n", (unsigned long long)set);

    for (int L = minlength; L <= maxlength; L++) {
        int tail_len = (L > 2) ? L - 2 : 0;
        for (int first = 0; first < tables.charset_sz; first++) {
            int pair_idx = (L - minlength) * tables.charset_sz + first;
            if (pair_idx % node_count != (node_id - 1))
                continue;

            /* Écriture de l'état de chaque second caractère non épuisé */
            for (int sc = 0; sc < tables.charset_sz; sc++) {
                if (!per_len[L-1].exhausted[first][sc]) {
                    fprintf(file, "%d %d %d ", L, first, sc);
                    for (int i = 0; i < tail_len; i++)
                        fprintf(file, "%02x", per_len[L-1].tails[first][sc][i]);
                    fprintf(file, "\n");
                }
            }
        }
    }
}

static int restore_state(FILE *file)
{
    int mn, mx, cs;
    unsigned long long st;
    if (fscanf(file, "%d\n%d\n%d\n", &mn, &mx, &cs) != 3) return 1;
    if (cs != tables.charset_sz) {
        fprintf(stderr, "inc2: charset size mismatch in recovery file, aborting restore\n");
        return 1;
    }
    if (fscanf(file, "%llu\n", &st) != 1) return 1;
    set = (uint_big)st;

    /* Réinitialisation complète de tous les états */
    for (int L = 1; L <= maxlength; L++) {
        int tail_len = (L > 2) ? L - 2 : 0;
        for (int first = 0; first < tables.charset_sz; first++) {
            for (int sc = 0; sc < tables.charset_sz; sc++) {
                if (tail_len > 0)
                    memset(per_len[L-1].tails[first][sc], 0, tail_len);
                per_len[L-1].exhausted[first][sc] = 0;
            }
            pair_exhausted[L-1][first] = 0;
        }
    }

    /* Lecture des lignes sauvegardées */
    int fileL, filefirst, filesc;
    char hex[2048];
    while (fscanf(file, "%d %d %d %s\n", &fileL, &filefirst, &filesc, hex) == 4) {
        int pair_idx = (fileL - mn) * tables.charset_sz + filefirst;
        if (pair_idx % node_count != (node_id - 1))
            continue;
        int tail_len = (fileL > 2) ? fileL - 2 : 0;
        int d = tail_len;
        if (d > 0 && (int)strlen(hex) != d * 2) continue;
        for (int i = 0; i < d; i++) {
            unsigned int byte;
            sscanf(hex + 2*i, "%2x", &byte);
            per_len[fileL-1].tails[filefirst][filesc][i] = (uint8_t)byte;
        }
        per_len[fileL-1].exhausted[filefirst][filesc] = 0;
        /* On ne peut pas déduire ici si la paire est épuisée, elle sera marquée plus tard */
    }
    state_restored = 1;
    return 0;
}

/* ------------------------------------------------------------------------- */
/*  Main entry point                                                         */
/* ------------------------------------------------------------------------- */
int do_inc2_crack(struct db_main *db, const char *freq_file)
{
    /* ---- length setup ---- */
    maxlength = MIN(MAX_CAND_LENGTH, options.eff_maxlength);
    minlength = MAX(options.eff_minlength, 1);
    if (!options.req_maxlength) maxlength = MIN(maxlength, DEFAULT_MAX_LEN);
    if (!options.req_minlength) minlength = 1;
    if (maxlength < minlength) {
        fprintf(stderr, "inc2: maxlength < minlength, aborting\n");
        return 1;
    }

    /* ---- load frequency tables ---- */
    if (freq_file && load_freq_from_file(freq_file)) {
        /* successfully loaded */
    } else {
        build_default_tables();
    }

    if (tables.maxlength < maxlength) {
        maxlength = tables.maxlength;
        if (minlength > maxlength) minlength = maxlength;
    }

    init_permutations();

    /* ---- node splitting ---- */
    node_count = (options.node_count > 1) ? options.node_count : 1;
    node_id = (options.node_min > 0) ? options.node_min : 1;
    if (node_id > node_count) node_id = node_count;

    pair_exhausted = malloc(maxlength * sizeof(int *));
	for (int L = 1; L <= maxlength; L++) {
		pair_exhausted[L-1] = calloc(tables.charset_sz, sizeof(int));
	}

    /* Allocation des structures de suffixe pour le stride du 2e caractère */
    for (int L = minlength; L <= maxlength; L++) {
        int num_first = tables.charset_sz;
        int tail_len = L - 2;  /* 0 pour L==2, <0 pour L==1 */
        per_len[L-1].tails = malloc(num_first * sizeof(uint8_t ***));
        per_len[L-1].exhausted = malloc(num_first * sizeof(int **));
        for (int f = 0; f < num_first; f++) {
            per_len[L-1].tails[f] = malloc(tables.charset_sz * sizeof(uint8_t *));
            per_len[L-1].exhausted[f] = calloc(tables.charset_sz, sizeof(int));
            for (int sc = 0; sc < tables.charset_sz; sc++) {
                if (tail_len > 0) {
                    per_len[L-1].tails[f][sc] = calloc(tail_len, sizeof(uint8_t));
                } else {
                    per_len[L-1].tails[f][sc] = NULL;
                }
            }
        }
    }

    status_init(get_progress, 0);
    rec_restore_mode(restore_state);
    rec_init(db, save_state);

    if (john_main_process) {
        log_event("Proceeding with inc2 mode (weighted first & second char stride)");
        log_event("Lengths: %d-%d, charset: %d, node %d/%d",
                  minlength, maxlength, tables.charset_sz, node_id, node_count);
        if (rec_restored) fprintf(stderr, "Proceeding with inc2 mode (resumed)\n");
    }

    crk_init(db, NULL, NULL);

    if (state_restored)
        state_restored = 0;

	double log_charset_sz = log(tables.charset_sz);

    int work_done;
    do {
        work_done = 0;
        for (int L = minlength; L <= maxlength; L++) {
            int tail_len = L - 2;
			double diff = (double)(L - minlength) / maxlength;
			// diff * diff squares the ratio, making the drop-off much sharper
			double len_factor = 1.0 / (1.0 + 5.0 * (diff * diff));
            for (int first = 0; first < tables.charset_sz; first++) {
                int pair_idx = (L - minlength) * tables.charset_sz + first;
                if (pair_idx % node_count != (node_id - 1))
                    continue;

                if (pair_exhausted[L-1][first])
                    continue;

                /* Longueur 1 : un seul candidat */
                if (L == 1) {
                    build_word(1, first, NULL, word);
                    set++;
                    if (options.flags & FLG_MASK_CHK) {
                        if (do_mask_crack(word)) goto out;
                    } else {
                        if (crk_process_key(word)) goto out;
                    }
                    pair_exhausted[0][first] = 1;
                    work_done = 1;
                    continue;
                }

                /* Weighted total stride for this (L,first) pair */
				double diff1 = (double)first / tables.charset_sz;
				double factor1 = 1.0 / (1.0 + 5.0 * (diff1 * diff1));
				int total_stride = (int)(INTERLEAVE_STRIDE * factor1 * len_factor);
				if (total_stride < 1) total_stride = 1;

				int any_work = 0;
				int generated = 0;   // candidates already produced for this pair

				for (int sc = 0; sc < tables.charset_sz && !pair_exhausted[L-1][first]; sc++) {
					if (per_len[L-1].exhausted[first][sc])
						continue;

					/* Weighted sub-stride for the second character */
					double factor2 = log_charset_sz;//1.0 / (1.0 + (double)sc / (double)tables.charset_sz);
					int stride2 = (int)(INTERLEAVE_SUB_STRIDE * factor2);
					if (stride2 < 1) stride2 = 1;

					/* Don't exceed the total budget for the first character */
					if (generated + stride2 > total_stride)
						stride2 = total_stride - generated;
					if (stride2 <= 0) break;

					for (int k = 0; k < stride2; k++) {
						if (event_abort) goto out;
						if (per_len[L-1].exhausted[first][sc])
							break;

						uint8_t suffix[PLAINTEXT_BUFFER_SIZE];
						suffix[0] = sc;
						if (tail_len > 0)
							memcpy(&suffix[1], per_len[L-1].tails[first][sc], tail_len);
						build_word(L, first, suffix, word);
						set++;

						if (options.flags & FLG_MASK_CHK) {
							if (do_mask_crack(word)) goto out;
						} else {
							if (crk_process_key(word)) goto out;
						}

						if (tail_len > 0) {
							if (suffix_add_one(per_len[L-1].tails[first][sc], tail_len, tables.charset_sz))
								per_len[L-1].exhausted[first][sc] = 1;
						} else {
							per_len[L-1].exhausted[first][sc] = 1;   // L=2
						}
						any_work = 1;
						generated++;
					}
				}
				if (any_work)
					work_done = 1;

				// Mark pair exhausted when all second characters are done
				int all_exhausted = 1;
				for (int sc = 0; sc < tables.charset_sz; sc++) {
					if (!per_len[L-1].exhausted[first][sc]) {
						all_exhausted = 0;
						break;
					}
				}
				if (all_exhausted)
					pair_exhausted[L-1][first] = 1;
			}
        }
        if (work_done)
            rec_save();
    } while (work_done && !event_abort);
out:
    crk_done();
    rec_done(event_abort);      // save_state est appelé dedans, donc per_len doit encore exister

    /* Maintenant on peut libérer per_len */
    for (int L = minlength; L <= maxlength; L++) {
        for (int first = 0; first < tables.charset_sz; first++) {
            for (int sc = 0; sc < tables.charset_sz; sc++) {
                free(per_len[L-1].tails[first][sc]);
            }
            free(per_len[L-1].tails[first]);
            free(per_len[L-1].exhausted[first]);
        }
        free(per_len[L-1].tails);
        free(per_len[L-1].exhausted);
    }

    /* Libération de pair_exhausted */
    for (int L = 1; L <= maxlength; L++)
        free(pair_exhausted[L-1]);
    free(pair_exhausted);

    return 0;
}
