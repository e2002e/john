/*
 * inc2.c – Weighted‑stride incremental mode (full key‑space, probability‑ordered).
 *
 * ... (original header preserved) ...
 */
#include "os.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
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

int inc2_cur_len;                     /* exported for status line */

#if JTR_HAVE_INT128
typedef uint128_t uint_big;
#define UINT_BIG_MAX UINT128_MAX
#else
typedef uint64_t uint_big;
#define UINT_BIG_MAX UINT64_MAX
#endif

#define MAX_CAND_LENGTH PLAINTEXT_BUFFER_SIZE
#define DEFAULT_MAX_LEN  16
#define INTERLEAVE_STRIDE  1000000       /* base stride per (L,first) pair per iteration */

#define ALPHA  0.5
#define MIN_EXTENSION_PROBABILITY 0.1

char word[PLAINTEXT_BUFFER_SIZE];

/* ------------------------------------------------------------------------- */
/*  Frequency table structures (unchanged)                                   */
/* ------------------------------------------------------------------------- */
struct inc2_tables {
    int charset_sz;
    int maxlength;
    int num_special;
    char *special_letters;
    char **base_order;
    char ***chainFreq;
    char ***counterChainFreq;
    char ***perm;
    int char_to_index[256];
    char index_to_char[256];
};

static struct inc2_tables tables;

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

/* ------------------------------------------------------------------------- */
/*  State structures                                                         */
/* ------------------------------------------------------------------------- */
struct len_state {
    int L;
    uint8_t **digits;
    int *exhausted;
};

/* Global tracking */
static uint_big total_generated = 0;
static uint_big total_implicit = 0;
static uint_big set = 0;
static int node_id = 1, node_count = 1;

static struct len_state *states;
static int minlength, maxlength;
static int use_dynamic_length = 0;


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
/*  Free frequency tables                                                    */
/* ------------------------------------------------------------------------- */
static void free_tables(struct inc2_tables *t)
{
    int i, j;
    if (!t) return;
    if (t->base_order) {
        for (i = 0; i < t->maxlength; i++) free(t->base_order[i]);
        free(t->base_order);
    }
    if (t->chainFreq) {
        for (i = 0; i < t->maxlength - 1; i++) {
            for (j = 0; j < t->num_special; j++) free(t->chainFreq[i][j]);
            free(t->chainFreq[i]);
        }
        free(t->chainFreq);
    }
    if (t->counterChainFreq) {
        for (i = 0; i < t->maxlength - 1; i++) {
            for (j = 0; j < t->num_special; j++) free(t->counterChainFreq[i][j]);
            free(t->counterChainFreq[i]);
        }
        free(t->counterChainFreq);
    }
    if (t->perm) {
        for (i = 0; i < t->maxlength; i++) {
            for (j = 0; j < t->charset_sz; j++) free(t->perm[i][j]);
            free(t->perm[i]);
        }
        free(t->perm);
    }
    free(t->special_letters);
    memset(t, 0, sizeof(*t));
}

/* ------------------------------------------------------------------------- */
/*  Build character mapping                                                  */
/* ------------------------------------------------------------------------- */
static void build_charmap(struct inc2_tables *t, const char *charset_str)
{
    memset(t->char_to_index, -1, sizeof(t->char_to_index));
    for (int i = 0; i < t->charset_sz; i++) {
        unsigned char c = (unsigned char)charset_str[i];
        t->char_to_index[c] = i;
        t->index_to_char[i] = c;
    }
}

/* ------------------------------------------------------------------------- */
/*  Load frequency file – identical to original                               */
/* ------------------------------------------------------------------------- */
static int load_freq_from_file(const char *fname)
{
    FILE *fp = fopen(fname, "r");
    if (!fp) {
        fprintf(stderr, "inc2: cannot open '%s', using defaults\n", fname);
        return 0;
    }
    int file_maxlen, i, L, sp, charset_sz;
    char line[8192];
    if (!fgets(line, sizeof(line), fp) || sscanf(line, "%d", &file_maxlen) != 1) {
        fprintf(stderr, "inc2: bad maxlen in freq file\n"); fclose(fp); return 0;
    }
    if (file_maxlen < 1 || file_maxlen > MAX_CAND_LENGTH) {
        fprintf(stderr, "inc2: maxlen %d out of range, using defaults\n", file_maxlen);
        fclose(fp); return 0;
    }
    if (!fgets(line, sizeof(line), fp)) { fclose(fp); return 0; }
    line[strcspn(line, "\r\n")] = 0;
    int num_special = strlen(line);
    if (num_special < 1) { fclose(fp); return 0; }
    if (!fgets(line, sizeof(line), fp)) { fclose(fp); return 0; }
    line[strcspn(line, "\r\n")] = 0;
    charset_sz = strlen(line);
    if (charset_sz < 1 || charset_sz > 256) { fclose(fp); return 0; }

    struct inc2_tables new_tables;
    memset(&new_tables, 0, sizeof(new_tables));
    new_tables.maxlength = file_maxlen;
    new_tables.charset_sz = charset_sz;
    new_tables.num_special = num_special;
    new_tables.special_letters = my_strdup(line);
    new_tables.base_order = malloc(file_maxlen * sizeof(char *));
    for (i = 0; i < file_maxlen; i++) new_tables.base_order[i] = malloc(charset_sz + 1);
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
    fseek(fp, 0, SEEK_SET);
    fgets(line, sizeof(line), fp);
    fgets(line, sizeof(line), fp);
    line[strcspn(line, "\r\n")] = 0;
    free(new_tables.special_letters);
    new_tables.special_letters = my_strdup(line);
    fclose(fp);
    fp = fopen(fname, "r");
    fgets(line, sizeof(line), fp);
    fgets(line, sizeof(line), fp);
    fgets(line, sizeof(line), fp);
    line[strcspn(line, "\r\n")] = 0;
    build_charmap(&new_tables, line);
    strcpy(new_tables.base_order[0], line);
    for (i = 1; i < file_maxlen; i++) {
        if (!fgets(line, sizeof(line), fp)) goto fail;
        line[strcspn(line, "\r\n")] = 0;
        if ((int)strlen(line) != charset_sz) goto fail;
        strcpy(new_tables.base_order[i], line);
    }
    for (L = 1; L < file_maxlen; L++) {
        for (sp = 0; sp < num_special; sp++) {
            int chain_len;
            if (!fgets(line, sizeof(line), fp) || sscanf(line, "%d", &chain_len) != 1) goto fail;
            if (chain_len > 0) {
                if (!fgets(line, sizeof(line), fp)) goto fail;
                line[strcspn(line, "\r\n")] = 0;
                if ((int)strlen(line) != chain_len) goto fail;
                new_tables.chainFreq[L-1][sp] = my_strdup(line);
            } else {
                if (!fgets(line, sizeof(line), fp)) goto fail;
                new_tables.chainFreq[L-1][sp] = my_strdup("");
            }
            if (!fgets(line, sizeof(line), fp)) goto fail;
            line[strcspn(line, "\r\n")] = 0;
            int expected_counter_len = charset_sz - chain_len;
            if ((int)strlen(line) != expected_counter_len) goto fail;
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
    t.base_order = malloc(maxlength * sizeof(char *));
    for (int i = 0; i < maxlength; i++) {
        t.base_order[i] = malloc(27);
        const char *src = (i < 15) ? default_base[i] : default_base[14];
        strcpy(t.base_order[i], src);
    }
    t.chainFreq = malloc((maxlength - 1) * sizeof(char **));
    t.counterChainFreq = malloc((maxlength - 1) * sizeof(char **));
    for (int i = 1; i < maxlength; i++) {
        int row = i - 1;
        if (row >= (int)(sizeof(chainFreq_default)/sizeof(chainFreq_default[0])))
            row = (int)(sizeof(chainFreq_default)/sizeof(chainFreq_default[0])) - 1;
        t.chainFreq[i-1] = malloc(t.num_special * sizeof(char *));
        t.counterChainFreq[i-1] = malloc(t.num_special * sizeof(char *));
        for (int sp = 0; sp < t.num_special; sp++) {
            t.chainFreq[i-1][sp] = my_strdup(chainFreq_default[row][sp]);
            t.counterChainFreq[i-1][sp] = my_strdup(counterChainFreq_default[row][sp]);
        }
    }
    free_tables(&tables);
    memcpy(&tables, &t, sizeof(tables));
}

/* ------------------------------------------------------------------------- */
/*  Build permutation for position 'pos' given previous character 'prev'     */
/* ------------------------------------------------------------------------- */
static void build_permutation(int pos, int prev_idx, char *out)
{
    int class = -1;
    char prev_char = tables.index_to_char[prev_idx];
    for (int sp = 0; sp < tables.num_special; sp++) {
        if (tables.special_letters[sp] == prev_char) { class = sp; break; }
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

static void init_permutations(void)
{
    tables.perm = malloc(tables.maxlength * sizeof(char **));
    for (int i = 0; i < tables.maxlength; i++) {
        tables.perm[i] = malloc(tables.charset_sz * sizeof(char *));
        for (int c = 0; c < tables.charset_sz; c++) {
            tables.perm[i][c] = malloc(tables.charset_sz + 1);
            if (i == 0) strcpy(tables.perm[i][c], tables.base_order[0]);
            else build_permutation(i, c, tables.perm[i][c]);
        }
    }
}

static int suffix_add_one(uint8_t *digits, int d, int base)
{
    for (int i = d - 1; i >= 0; i--) {
        if (digits[i] < base - 1) { digits[i]++; return 0; }
        digits[i] = 0;
    }
    return 1; /* wrapped -> exhausted */
}

static void build_word(int L, int first, const uint8_t *suffix, char *out)
{
    out[0] = tables.base_order[0][first];
    for (int i = 1; i < L; i++) {
        int prev = tables.char_to_index[(unsigned char)out[i-1]];
        out[i] = tables.perm[i][prev][ suffix[i-1] ];
    }
    out[L] = '\0';
}

static double compute_weight(const uint8_t *digits, int d)
{
    double w = 1.0;
    for (int i = 0; i < d; i++)
        w *= 1.0 / (1.0 + ALPHA * (double)digits[i]);
    return w;
}
static double get_progress(void)
{
    static double node_total = -1.0;
    if (node_total < 0.0) {
        node_total = 0.0;
        for (int L = minlength; L <= maxlength; L++) {
            int pairs = 0;
            for (int first = 0; first < tables.charset_sz; first++) {
                int pair_idx = (L - minlength) * tables.charset_sz + first;
                if (pair_idx % node_count == (node_id - 1))
                    pairs++;
            }
            if (pairs == 0) continue;
            double space = (L == 1) ? 1.0 : pow((double)tables.charset_sz, (double)(L-1));
            node_total += pairs * space;
        }
    }
    if (node_total == 0.0) return -1.0;
    double done = (double)(total_generated + total_implicit) / node_total;
    return (done > 1.0) ? 1.0 : done;
}

/* ------------------------------------------------------------------------- */
/*  Save / restore (supports both fixed and dynamic modes)                   */
/* ------------------------------------------------------------------------- */
static void save_state(FILE *file)
{
    fprintf(file, "%d\n%d\n%d\n%d\n", minlength, maxlength, tables.charset_sz, use_dynamic_length ? 1 : 0);
    fprintf(file, "%llu\n%llu\n%llu\n",
            (unsigned long long)set,
            (unsigned long long)total_generated,
            (unsigned long long)total_implicit);

    /* Save every non‑exhausted (L, first) pair, with its suffix digits */
    for (int L = minlength; L <= maxlength; L++) {
        struct len_state *st = &states[L - minlength];
        int d = L - 1;
        for (int first = 0; first < tables.charset_sz; first++) {
            int pair_idx = (L - minlength) * tables.charset_sz + first;
            if (pair_idx % node_count != (node_id - 1)) continue;
            if (st->exhausted[first]) continue;
            fprintf(file, "%d %d", L, first);
            if (d > 0) {
                for (int i = 0; i < d; i++)
                    fprintf(file, " %02x", st->digits[first][i]);
            }
            fprintf(file, "\n");
        }
    }
}

static int restore_state(FILE *file)
{
    int mn, mx, cs, is_dyn;
    unsigned long long st, gen, imp;
    if (fscanf(file, "%d\n%d\n%d\n%d\n", &mn, &mx, &cs, &is_dyn) != 4) return 1;
    if (fscanf(file, "%llu\n%llu\n%llu\n", &st, &gen, &imp) != 3) return 1;

    if (cs != tables.charset_sz) {
        fprintf(stderr, "inc2: charset size mismatch in recovery file\n");
        return 1;
    }

    set = st;
    total_generated = gen;
    total_implicit = imp;

    /* Mark everything exhausted, then re‑enable pairs that are in the file */
    for (int L = minlength; L <= maxlength; L++) {
        struct len_state *st = &states[L - minlength];
        for (int first = 0; first < tables.charset_sz; first++)
            st->exhausted[first] = 1;
        /* Length‑1 pairs are re‑enabled below (if they appear in the file) */
    }

    char line[4096], *p;
    while (fgets(line, sizeof(line), file)) {
        int fileL, filefirst;
        if (sscanf(line, "%d %d", &fileL, &filefirst) < 2) continue;
        if (fileL < minlength || fileL > maxlength) continue;
        int pair_idx = (fileL - minlength) * tables.charset_sz + filefirst;
        if (pair_idx % node_count != (node_id - 1)) continue;

        struct len_state *st = &states[fileL - minlength];
        int d = fileL - 1;
        p = strchr(line, ' ') + 1;   /* skip L */
        p = strchr(p, ' ') + 1;      /* skip first */
        if (d > 0) {
            for (int i = 0; i < d; i++) {
                unsigned int byte;
                if (sscanf(p, "%2x", &byte) != 1) break;
                st->digits[filefirst][i] = (uint8_t)byte;
                p += 2;
                if (*p == ' ') p++;
            }
        }
        st->exhausted[filefirst] = 0;
    }
    return 0;
}

/* ------------------------------------------------------------------------- */
/*  Main cracking loop - recursive version                                   */
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

    use_dynamic_length = (minlength != maxlength);

    /* ---- load frequency tables ---- */
    if (freq_file && load_freq_from_file(freq_file)) {
        /* loaded */
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

    /* ---- allocate one state per length (used by both modes) ---- */
    states = calloc(maxlength - minlength + 1, sizeof(struct len_state));
    for (int L = minlength; L <= maxlength; L++) {
        struct len_state *st = &states[L - minlength];
        st->L = L;
        st->exhausted = calloc(tables.charset_sz, sizeof(int));
        if (L > 1) {
            st->digits = malloc(tables.charset_sz * sizeof(uint8_t *));
            for (int f = 0; f < tables.charset_sz; f++)
                st->digits[f] = calloc(L - 1, sizeof(uint8_t));
        } else {
            st->digits = NULL;
        }
    }

    status_init(get_progress, 0);
    rec_restore_mode(restore_state);
    rec_init(db, save_state);

    if (john_main_process) {
        log_event("Proceeding with inc2 mode (alpha=%.2f)", ALPHA);
        log_event("Lengths: %d-%d%s, charset: %d, node %d/%d",
                  minlength, maxlength,
                  use_dynamic_length ? " (dynamic)" : "",
                  tables.charset_sz, node_id, node_count);
        if (rec_restored) fprintf(stderr, "Proceeding with inc2 mode (resumed)\n");
    }

    crk_init(db, NULL, NULL);

    if (use_dynamic_length) {
        /* --------------------------------------------------------------- */
        /*  Dynamic mode: probability‑guided round‑robin between lengths   */
        /* --------------------------------------------------------------- */
        #define BASE_CANDIDATES_PER_ROUND  1000

        int any_work;
        do {
            any_work = 0;

            /* First pass: sum priorities of all live pairs */
            double total_priority = 0.0;
            for (int L = minlength; L <= maxlength; L++) {
                struct len_state *st = &states[L - minlength];
                int d = L - 1;
                for (int first = 0; first < tables.charset_sz; first++) {
                    int pair_idx = (L - minlength) * tables.charset_sz + first;
                    if (pair_idx % node_count != (node_id - 1)) continue;
                    if (st->exhausted[first]) continue;

                    double priority;
                    if (L == 1) {
                        priority = 2.0;
                    } else {
                        /* Guard against corrupted state */
                        if (!st->digits || !st->digits[first]) {
                            st->exhausted[first] = 1;  // skip corrupted pair
                            continue;
                        }
                        double prob = compute_weight(st->digits[first], d);
                        double length_bonus = 1.0 / (double)L;
                        priority = prob * length_bonus;
                    }
                    total_priority += priority;
                }
            }

            if (total_priority == 0.0) break;

            /* Second pass: generate a proportional batch from each pair */
            for (int L = minlength; L <= maxlength; L++) {
                struct len_state *st = &states[L - minlength];
                int d = L - 1;
                for (int first = 0; first < tables.charset_sz; first++) {
                    int pair_idx = (L - minlength) * tables.charset_sz + first;
                    if (pair_idx % node_count != (node_id - 1)) continue;
                    if (st->exhausted[first]) continue;

                    double priority;
                    if (L == 1) {
                        priority = 2.0;
                    } else {
                        if (!st->digits || !st->digits[first]) {
                            st->exhausted[first] = 1;
                            continue;
                        }
                        priority = compute_weight(st->digits[first], d) * (1.0 / (double)L);
                    }

                    int budget = (int)(BASE_CANDIDATES_PER_ROUND * (priority / total_priority));
                    if (budget < 1) budget = 1;

                    uint8_t suffix[PLAINTEXT_BUFFER_SIZE];
                    if (L > 1) memcpy(suffix, st->digits[first], d);

                    for (int k = 0; k < budget; k++) {
                        if (event_abort) goto out;

                        if (L == 1) {
                            word[0] = tables.base_order[0][first];
                            word[1] = '\0';
                            set++;
                            total_generated++;
                            any_work = 1;

                            if (options.flags & FLG_MASK_CHK) {
                                if (do_mask_crack(word)) goto out;
                            } else {
                                if (crk_process_key(word)) goto out;
                            }
                            st->exhausted[first] = 1;
                            break;
                        } else {
                            build_word(L, first, suffix, word);
                            set++;
                            total_generated++;
                            any_work = 1;

                            if (options.flags & FLG_MASK_CHK) {
                                if (do_mask_crack(word)) goto out;
                            } else {
                                if (crk_process_key(word)) goto out;
                            }

                            if (suffix_add_one(suffix, d, tables.charset_sz)) {
                                st->exhausted[first] = 1;
                                memcpy(st->digits[first], suffix, d);
                                break;
                            }
                        }
                    }
                    if (!st->exhausted[first] && L > 1)
                        memcpy(st->digits[first], suffix, d);
                }
            }

            if (any_work && total_generated % 10000 == 0)
                rec_save();

        } while (any_work && !event_abort);
    } else {
        /* --------------------------------------------------------------- */
        /*  Fixed‑length mode: original weighted‑stride batch generation   */
        /* --------------------------------------------------------------- */
        int work_done;
        do {
            work_done = 0;
            for (int L = minlength; L <= maxlength; L++) {
                struct len_state *st = &states[L - minlength];
                int d = L - 1;
                for (int first = 0; first < tables.charset_sz; first++) {
                    int pair_idx = (L - minlength) * tables.charset_sz + first;
                    if (pair_idx % node_count != (node_id - 1)) continue;
                    if (st->exhausted[first]) continue;

                    if (L == 1) {
                        word[0] = tables.base_order[0][first];
                        word[1] = '\0';
                        set++;
                        total_generated++;
                        work_done = 1;
                        if (options.flags & FLG_MASK_CHK) {
                            if (do_mask_crack(word)) goto out;
                        } else {
                            if (crk_process_key(word)) goto out;
                        }
                        st->exhausted[first] = 1;
                        continue;
                    }

                    uint8_t suffix[PLAINTEXT_BUFFER_SIZE];
                    memcpy(suffix, st->digits[first], d);

                    double weight = compute_weight(suffix, d);
                    int stride = (int)(INTERLEAVE_STRIDE * weight);
                    if (stride < 1) stride = 1;

                    for (int k = 0; k < stride; k++) {
                        if (event_abort) goto out;
                        build_word(L, first, suffix, word);
                        set++;
                        total_generated++;
                        work_done = 1;
                        if (options.flags & FLG_MASK_CHK) {
                            if (do_mask_crack(word)) goto out;
                        } else {
                            if (crk_process_key(word)) goto out;
                        }
                        if (suffix_add_one(suffix, d, tables.charset_sz)) {
                            st->exhausted[first] = 1;
                            break;
                        }
                    }
                    memcpy(st->digits[first], suffix, d);
                }
            }
            if (work_done) rec_save();
        } while (work_done && !event_abort);
    }

out:
    crk_done();
    rec_done(event_abort);

    /* ---- free state ---- */
    for (int L = minlength; L <= maxlength; L++) {
        struct len_state *st = &states[L - minlength];
        if (L > 1) {
            for (int f = 0; f < tables.charset_sz; f++)
                free(st->digits[f]);
            free(st->digits);
        }
        free(st->exhausted);
    }
    free(states);

    return 0;
}
