/*
 * inc2.c – Block‑interleaved lengths & first characters.
 *          First character rotates every BLOCK_SIZE words.
 *
 * Full version: dynamic character set, arbitrary special letters,
 *               frequency tables loaded from a file.
 *
 * Parallelisation: (length, first_char) pairs are distributed
 * round‑robin across nodes. Within a node, a small stride of
 * candidates is generated from each assigned pair per iteration,
 * ensuring early words from *all* lengths are tried quickly.
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

/* ------------------------------------------------------------------------- */
/*  Odometer data – dynamically allocated                                    */
/* ------------------------------------------------------------------------- */
static uint8_t ***suffix_digits;      /* [L-1][first_idx][i] for i = 0..L-2 */
static int **suffix_exhausted;        /* [L-1][first_idx] */

/* Power table for progress (charset_sz^L) */
static uint_big *pow_charset;

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
/*  Free suffix digit arrays                                                 */
/* ------------------------------------------------------------------------- */
static void free_suffix_arrays(void)
{
    int L, c;
    if (suffix_digits) {
        for (L = 1; L <= maxlength; L++) {
            for (c = 0; c < tables.charset_sz; c++)
                free(suffix_digits[L-1][c]);
            free(suffix_digits[L-1]);
        }
        free(suffix_digits);
        suffix_digits = NULL;
    }
    if (suffix_exhausted) {
        for (L = 1; L <= maxlength; L++)
            free(suffix_exhausted[L-1]);
        free(suffix_exhausted);
        suffix_exhausted = NULL;
    }
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

/* ------------------------------------------------------------------------- */
/*  Power tables (charset_sz^L) for progress reporting                       */
/* ------------------------------------------------------------------------- */
static void init_powers(void)
{
    int i;
    pow_charset = malloc((maxlength + 2) * sizeof(uint_big));
    pow_charset[0] = 1;
    for (i = 1; i <= maxlength + 1; i++) {
        if (pow_charset[i-1] > UINT_BIG_MAX / tables.charset_sz)
            pow_charset[i] = 0;
        else
            pow_charset[i] = pow_charset[i-1] * tables.charset_sz;
    }
}

/* ------------------------------------------------------------------------- */
/*  Build candidate word from odometer digits                                */
/* ------------------------------------------------------------------------- */
static void build_word_from_digits(int len, int first_idx, uint8_t *digits, char *out)
{
    int i;
    out[0] = tables.base_order[0][first_idx];
    for (i = 1; i < len; i++) {
        int prev = tables.char_to_index[(unsigned char)out[i-1]];
        out[i] = tables.perm[i][prev][ digits[len - 1 - i] ];
    }
    out[len] = '\0';
}

/* ------------------------------------------------------------------------- */
/*  Advance suffix odometer (base charset_sz)                                */
/* ------------------------------------------------------------------------- */
static int advance_suffix(int L, int first)
{
    int i;
    for (i = 0; i < L - 1; i++) {
        if (suffix_digits[L-1][first][i] < tables.charset_sz - 1) {
            suffix_digits[L-1][first][i]++;
            return 1;
        }
        suffix_digits[L-1][first][i] = 0;
    }
    suffix_exhausted[L-1][first] = 1;
    return 0;
}

/* ------------------------------------------------------------------------- */
/*  Progress – node's fraction of total candidates                           */
/* ------------------------------------------------------------------------- */
static double get_progress(void)
{
	int L;
    static double node_total = -1.0;
    if (node_total < 0.0) {
        node_total = 0.0;
        for (L = minlength; L <= maxlength; L++) {
            if (pow_charset[L] > 0)
                node_total += (double)pow_charset[L];
        }
        /* Each node's share is total/node_count */
        node_total /= node_count;
    }
    if (node_total == 0.0) return -1.0;
    double prog = (double)set / node_total;
    return prog > 1.0 ? 1.0 : prog;
}

/* ------------------------------------------------------------------------- */
/*  Recovery – save/restore only the node's assigned pairs                   */
/* ------------------------------------------------------------------------- */
static void save_state(FILE *file)
{
    fprintf(file, "%d\n%d\n%d\n", minlength, maxlength, tables.charset_sz);
    fprintf(file, "%llu\n", (unsigned long long)set);

    int L, c, i;
    for (L = minlength; L <= maxlength; L++) {
        for (c = 0; c < tables.charset_sz; c++) {
            int pair_idx = (L - minlength) * tables.charset_sz + c;
            if (pair_idx % node_count != (node_id - 1))
                continue;   /* not our pair */
            /* write L and first_char index as a header for clarity */
            fprintf(file, "%d %d\n", L, c);
            for (i = 0; i < L - 1; i++)
                fprintf(file, "%d ", suffix_digits[L-1][c][i]);
            fprintf(file, "\n");
            fprintf(file, "%d\n", suffix_exhausted[L-1][c]);
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

    /* Reset all suffix states */
    int L, c, i;
    for (L = mn; L <= mx; L++) {
        for (c = 0; c < tables.charset_sz; c++) {
            memset(suffix_digits[L-1][c], 0, L-1);
            suffix_exhausted[L-1][c] = 0;
        }
    }

    /* Read data for our pairs (identified by L and c) */
    int fileL, filec;
    while (fscanf(file, "%d %d\n", &fileL, &filec) == 2) {
        int pair_idx = (fileL - mn) * tables.charset_sz + filec;
        if (pair_idx % node_count != (node_id - 1)) {
            /* Unexpected pair – skip? but we only saved ours, so should never happen */
            /* skip the data */
            char dummy[256];
            fgets(dummy, sizeof(dummy), file);  /* digits line */
            fgets(dummy, sizeof(dummy), file);  /* exhausted flag */
            continue;
        }
        for (i = 0; i < fileL - 1; i++) {
            int digit;
            if (fscanf(file, "%d ", &digit) != 1) return 1;
            suffix_digits[fileL-1][filec][i] = (uint8_t)(digit % tables.charset_sz);
        }
        int newline = fgetc(file);
        if (newline != '\n') ungetc(newline, file);
        int exhausted;
        if (fscanf(file, "%d\n", &exhausted) != 1) return 1;
        suffix_exhausted[fileL-1][filec] = exhausted;
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
    init_powers();

    /* ---- node splitting (round-robin) ---- */
    node_count = (options.node_count > 1) ? options.node_count : 1;
    node_id = (options.node_min > 0) ? options.node_min : 1;
    if (node_id > node_count) node_id = node_count;

    total_pairs = (maxlength - minlength + 1) * tables.charset_sz;

    /* ---- allocate suffix digit arrays ---- */
    free_suffix_arrays();
    suffix_digits = malloc(maxlength * sizeof(uint8_t **));
    suffix_exhausted = malloc(maxlength * sizeof(int *));
    int L, c;
    for (L = 1; L <= maxlength; L++) {
        suffix_digits[L-1] = malloc(tables.charset_sz * sizeof(uint8_t *));
        suffix_exhausted[L-1] = malloc(tables.charset_sz * sizeof(int));
        for (c = 0; c < tables.charset_sz; c++) {
            suffix_digits[L-1][c] = malloc((L - 1) * sizeof(uint8_t));
            memset(suffix_digits[L-1][c], 0, (L - 1));
            suffix_exhausted[L-1][c] = 0;
        }
    }

    status_init(get_progress, 0);
    rec_restore_mode(restore_state);
    rec_init(db, save_state);

    if (john_main_process) {
        log_event("Proceeding with inc2 mode (round-robin pairs, interleaved strides)");
        log_event("Lengths: %d-%d, charset: %d characters, special: %d",
                  minlength, maxlength, tables.charset_sz, tables.num_special);
        log_event("Node %d/%d of %d total pairs", node_id, node_count, total_pairs);
        if (rec_restored) fprintf(stderr, "Proceeding with inc2 mode (resumed)\n");
    }

    crk_init(db, NULL, NULL);

    if (state_restored)
        state_restored = 0;

    /* ---- main generation loop (interleaved strides) ---- */
    int work_done;
    do {
		work_done = 0;
		/* Walk all possible pair indices; only process those belonging to this node */
		int pair_idx;
		for (pair_idx = 0; pair_idx < total_pairs; pair_idx++) {
			if (pair_idx % node_count != (node_id - 1))
				continue;

			int first = pair_idx / (maxlength - minlength + 1);
			int L = minlength + (pair_idx % (maxlength - minlength + 1));

            if (suffix_exhausted[L-1][first])
                continue;

            int first_char_rank = first;   // 0-based
			double factor = 1.0 / (1.0 + (double)first_char_rank / (double)tables.charset_sz);
			int stride = (int)(INTERLEAVE_STRIDE * factor);
			if (stride < 1) stride = 1;

			uint_big stride_remain = stride;
			while (stride_remain > 0 &&
                   !suffix_exhausted[L-1][first] &&
                   !event_abort) {

                build_word_from_digits(L, first, suffix_digits[L-1][first], word);
                set++;

                if (options.flags & FLG_MASK_CHK) {
                    if (do_mask_crack(word)) goto out;
                } else {
                    if (crk_process_key(word)) goto out;
                }

                stride_remain--;
                if (!advance_suffix(L, first))
                    break;
            }
            work_done = 1;
        }
        /* Save state after each full pass (all pairs) for safe recovery */
        if (work_done)
            rec_save();
    } while (work_done && !event_abort);

out:
    crk_done();
    rec_done(event_abort);
    return 0;
}
