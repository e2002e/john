/*
 * This file is part of John the Ripper password cracker,
 * Copyright (c) 2013-2018 by magnum
 * Copyright (c) 2014 by Sayantan Datta
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted.
 *
 * There's ABSOLUTELY NO WARRANTY, express or implied.
 */

#include <stdio.h> /* for fprintf(stderr, ...) */
#include <string.h>
#include <ctype.h>
#include <math.h>
#include "arch.h"
#include "misc.h" /* for error() */
#include "logger.h"
#include "recovery.h"
#include "os.h"
#include "signals.h"
#include "status.h"
#include "options.h"
#include "config.h"
#include "external.h"
#include "cracker.h"
#include "john.h"
#include "mask.h"
#include "unicode.h"
#include "encoding_data.h"
#include "mask_ext.h"
#include "markov_tables.h"

/* Fallbacks in case an older markov_tables.h (without quantized levels) is in
 * use. The magnitude-threshold feature additionally needs markov_start_level[]
 * and markov_table_level[][] which only a regenerated header provides. */
#ifndef MARKOV_MAXLEVEL
#define MARKOV_MAXLEVEL 255
#endif
#ifndef MARKOV_LEVEL_SCALE
#define MARKOV_LEVEL_SCALE 16
#endif

//#define MASK_DEBUG

extern void wordlist_hybrid_fix_state(void);
extern void mkv_hybrid_fix_state(void);
extern void inc_hybrid_fix_state(void);
extern void pp_hybrid_fix_state(void);
extern void ext_hybrid_fix_state(void);

#define INTERLEAVE_STRIDE_DEFAULT (1024 * 64)
/* Round-robin chunk granularity: the most-frequent active length advances this
 * many candidates per round (rarer lengths proportionally less). Smaller = the
 * crack stream alternates lengths more finely (more rounds/segments); larger =
 * coarser length runs but a deeper interleaved front before the NSEG_MAX tail.
 * Tunable via JOHN_INTERLEAVE for sweeping (0/unset = default). */
static int interleave_stride_cached = -1;
static int get_interleave_stride(void)
{
	if (interleave_stride_cached < 0) {
		const char *e = getenv("JOHN_INTERLEAVE");
		int v = e ? atoi(e) : 0;
		interleave_stride_cached = (v > 0) ? v : INTERLEAVE_STRIDE_DEFAULT;
	}
	return interleave_stride_cached;
}
#define INTERLEAVE_STRIDE get_interleave_stride()

// Filtered tables for O(1) lookup in the hot loop
static unsigned char pos_markov_table[MAX_NUM_MASK_PLHDR][256][256];
static unsigned char pos_markov_start[MAX_NUM_MASK_PLHDR][256];
/* Valid transitions per (range, prev) never exceeds the range's char count
 * (<= 255), so a byte is enough; quarters the footprint of this table. */
static unsigned char pos_markov_row_counts[MAX_NUM_MASK_PLHDR][256];

/* Thresholded simplex wheel ceiling per position (see JOHN_GEN_MAXLEVEL /
 * JOHN_GEN_MINP). Without a threshold both equal the range char count, so the
 * simplex enumerates the full keyspace exactly as before. With a threshold the
 * ceiling shrinks to the largest number of above-cutoff chars any context keeps
 * at that position, which is what actually contracts the keyspace and the
 * advance scan. _start applies to the leftmost (start-node) position, _trans to
 * interior (transition) positions. */
static unsigned char pos_markov_start_ceiling[MAX_NUM_MASK_PLHDR];
static unsigned char pos_markov_trans_ceiling[MAX_NUM_MASK_PLHDR];

/*
 * Per-length-loop hoisted ("struct of arrays") view of the active ranges.
 * Rebuilt once whenever the generator (re)enters a length-loop, so the
 * per-candidate hot loops below never index the fat mask_range structs
 * (~390 B each, scattered by absolute range id) nor recompute invariant
 * write offsets / table base pointers on every key.
 */
static int             lc_kp[MAX_NUM_MASK_PLHDR + 1];     /* template_key write index */
static unsigned char   lc_start[MAX_NUM_MASK_PLHDR + 1];  /* range start (0 => Markov)  */
static unsigned char   lc_count[MAX_NUM_MASK_PLHDR + 1];  /* range char count           */
static unsigned char   lc_chars0[MAX_NUM_MASK_PLHDR + 1]; /* first char (0-choice fallback) */
static unsigned char  *lc_iter[MAX_NUM_MASK_PLHDR + 1];   /* &range.iter[loop]          */
static unsigned char (*lc_table[MAX_NUM_MASK_PLHDR + 1])[256]; /* pos_markov_table[ri]  */
static unsigned char  *lc_start_tab[MAX_NUM_MASK_PLHDR + 1];   /* pos_markov_start[ri]  */
static unsigned char  *lc_rowcnt[MAX_NUM_MASK_PLHDR + 1]; /* pos_markov_row_counts[ri]  */

static mask_parsed_ctx parsed_mask;
static mask_cpu_context cpu_mask_ctx, rec_ctx, restored_ctx;
static int *template_key_offsets;
static char *mask = NULL, *template_key;
static int old_extern_key_len;
static int max_keylen, rec_len, restored_len, restored;
static uint64_t rec_cl, cand_length;
static struct fmt_main *mask_fmt;
struct db_main *mask_db;
static int mask_bench_index;
static int parent_fix_state_pending;
static unsigned int int_mask_sum, format_cannot_reset;
static int using_default_mask;

int mask_add_len, mask_num_qw, mask_cur_len, mask_iter_warn;
int mask_increments_len;

/*
 * This keeps track of whether we have any 8-bit in our non-hybrid mask.
 * If we do not, we can skip expensive encoding conversions
 */
static int mask_has_8bit;

/*
 * cand and rec_cand is the number of remaining candidates.
 * So, its value decreases as cracking progress.
 */
static uint64_t cand, rec_cand;
/* Per-length-loop candidate budget for this node (block distribution).
 * Only meaningful when options.node_count > 1 and not a stacked/hybrid run. */
static uint64_t node_loop_cand[MASK_MAX_INC_LEN], rec_node_loop_cand[MASK_MAX_INC_LEN];

uint64_t mask_tot_cand;
uint64_t mask_parent_keys;

#define BUILT_IN_CHARSET "ludsaLUDSAbhBH123456789"

#define store_op(k, i) \
	parsed_mask->stack_op_br[k] = i;

#define store_cl(k, i) \
	parsed_mask->stack_cl_br[k] = i;

#define load_op(i) \
	parsed_mask->stack_op_br[i]

#define load_cl(i) \
	parsed_mask->stack_cl_br[i]

#define load_qtn(i) \
	parsed_mask->stack_qtn[i]

/*
 * Converts \xHH notation to characters. The original buffer is modified -
 * we are guaranteed the new string is shorter or same length.
 *
 * This function must pass escaped characters on, as-is (still escaped),
 * including "\\" which may escape "\\xHH" from being parsed as \xHH.
 */
static char* parse_hex(char *string)
{
	static int warned;
	unsigned char *s = (unsigned char*)string;
	unsigned char *d = s;

	if (!string || !*string)
		return string;

	while (*s)
	if (*s == '\\' && s[1] != 'x') {
		*d++ = *s++;
		*d++ = *s++;
	} else if (*s == '\\' && s[1] == 'x' &&
	    atoi16[s[2]] != 0x7f && atoi16[s[3]] != 0x7f) {
		char c = (atoi16[s[2]] << 4) + atoi16[s[3]];
		if (!c && !warned++ && john_main_process)
			fprintf(stderr, "Warning: \\x00 in mask terminates the string\n");
		if (strchr("\\[]?-", c))
			*d++ = '\\';
		*d++ = c;
		s += 4;
	} else
		*d++ = *s++;

	*d = 0;

	return string;
}

/*
 * Expands custom placeholders in string and returns a new resulting string.
 * with -1=?u?l, "A?1abc[3-6]" will expand to "A[?u?l]abc[3-6]"
 *
 * This function must pass any escaped characters on, as-is (still escaped).
 * This function must ignore ? inside square brackets as unchanged [a-c1?2] is [a-c1?2]
 */
static char* expand_cplhdr(char *string, int *conv_err)
{
	static char out[0x8000];
	unsigned char *s = (unsigned char*)string;
	char *d = out;
	int in_brackets = 0, esc=0;

	if (!string || !*string)
		return string;

	while (*s && d < &out[sizeof(out) - 2]) {
		if (s[0] == '?' && s[1] == '?') {
			*d++ = '\\';
			*d++ = *s++;
			s++;
		} else
		if (*s == '\\') {
			*d++ = *s++;
			esc = 1;
		} else
		if (!in_brackets && *s == '?' && s[1] >= '1' && s[1] <= '9') {
			int ab = 0;
			int pidx = s[1] - '1';
			char *cs = options.custom_mask[pidx];

			if (conv_err[pidx]) {
				if (john_main_process)
					fprintf(stderr,
					        "Error: Selected internal codepage can't hold all chars of mask placeholder ?%d\n",
					        pidx + 1);
				error();
			}
			if (*cs == 0) {
				if (john_main_process)
					fprintf(stderr, "Error: Custom mask placeholder ?%d not defined\n", pidx + 1);
				error();
			}
			if (*cs != '[') {
				*d++ = '[';
				ab = 1;
			}
			while (*cs && d < &out[sizeof(out) - 2])
				*d++ = *cs++;
			if (ab)
				*d++ = ']';
			s += 2;
		} else {
			if (!esc) {
				if (*s == '[') {
					++in_brackets;
					if (s[1] == ']') {
						if (john_main_process)
							fprintf(stderr, "Error: Invalid mask: Empty group []\n");
						error();
					}
				}
				else if (*s == ']')
					--in_brackets;
			} else
				esc = 0;
			*d++ = *s++;
		}
	}
	*d = '\0';

	return out;
}

#define add_range(a, b)	for (j = a; j <= b; j++) *o++ = j
#define add_string(str)	for (s = (char*)str; *s; s++) *o++ = *s

/*
 * Convert a single placeholder like ?l (given as 'l' char arg.) to a string.
 * plhdr2string('d', n) will return "0123456789"
 *
 * This function never has to deal with escapes (would not be called).
 */
static char* plhdr2string(char p, int fmt_case)
{
	static char out[256];
	char *s, *o = out;
	int j;

	/*
	 * Force lowercase for case insignificant formats. Dupes will
	 * be removed, so e.g. ?l?u == ?l.
	 */
	if (!fmt_case) {
		if (p == 'u')
			p = 'l';
		if (p == 'U')
			p = 'L';
	}

	if ((options.internal_cp == ENC_RAW || options.internal_cp == UTF_8) &&
	    (p == 'L' || p == 'U' || p == 'D' || p == 'S')) {
		if (john_main_process)
			fprintf(stderr,
			        "Error: Can't use ?%c placeholder without setting an 8-bit legacy codepage with\n"
			        "       --internal-codepage%s.\n", p,
			        (options.internal_cp == UTF_8) ? " (UTF-8 is not a codepage)" : "");
		error();
	}

	switch(p) {

	case 'l': /* lower-case letters */
		/* Rockyou character frequency */
		add_string("aeionrlstmcdyhubkgpjvfwzxq");
		break;

	case 'L': /* lower-case letters, non-ASCII only */
		switch (options.internal_cp) {
		case CP437:
			add_string(CHARS_LOWER_CP437
			           CHARS_LOW_ONLY_CP437
			           CHARS_NOCASE_CP437);
			break;
		case CP720:
			add_string(CHARS_LOWER_CP720
			           CHARS_LOW_ONLY_CP720
			           CHARS_NOCASE_CP720);
			break;
		case CP737:
			add_string(CHARS_LOWER_CP737
			           CHARS_LOW_ONLY_CP737
			           CHARS_NOCASE_CP737);
			break;
		case CP850:
			add_string(CHARS_LOWER_CP850
			           CHARS_LOW_ONLY_CP850
			           CHARS_NOCASE_CP850);
			break;
		case CP852:
			add_string(CHARS_LOWER_CP852
			           CHARS_LOW_ONLY_CP852
			           CHARS_NOCASE_CP852);
			break;
		case CP858:
			add_string(CHARS_LOWER_CP858
			           CHARS_LOW_ONLY_CP858
			           CHARS_NOCASE_CP858);
			break;
		case CP866:
			add_string(CHARS_LOWER_CP866
			           CHARS_LOW_ONLY_CP866
			           CHARS_NOCASE_CP866);
			break;
		case CP868:
			add_string(CHARS_LOWER_CP868
			           CHARS_LOW_ONLY_CP868
			           CHARS_NOCASE_CP868);
			break;
		case CP1250:
			add_string(CHARS_LOWER_CP1250
			           CHARS_LOW_ONLY_CP1250
			           CHARS_NOCASE_CP1250);
			break;
		case CP1251:
			add_string(CHARS_LOWER_CP1251
			           CHARS_LOW_ONLY_CP1251
			           CHARS_NOCASE_CP1251);
			break;
		case CP1252:
			add_string(CHARS_LOWER_CP1252
			           CHARS_LOW_ONLY_CP1252
			           CHARS_NOCASE_CP1252);
			break;
		case CP1253:
			add_string(CHARS_LOWER_CP1253
			           CHARS_LOW_ONLY_CP1253
			           CHARS_NOCASE_CP1253);
			break;
		case CP1254:
			add_string(CHARS_LOWER_CP1254
			           CHARS_LOW_ONLY_CP1254
			           CHARS_NOCASE_CP1254);
			break;
		case CP1255:
			add_string(CHARS_LOWER_CP1255
			           CHARS_LOW_ONLY_CP1255
			           CHARS_NOCASE_CP1255);
			break;
		case CP1256:
			add_string(CHARS_LOWER_CP1256
			           CHARS_LOW_ONLY_CP1256
			           CHARS_NOCASE_CP1256);
			break;
		case ISO_8859_1:
			add_string(CHARS_LOWER_ISO_8859_1
			           CHARS_LOW_ONLY_ISO_8859_1
			           CHARS_NOCASE_ISO_8859_1);
			break;
		case ISO_8859_2:
			add_string(CHARS_LOWER_ISO_8859_2
			           CHARS_LOW_ONLY_ISO_8859_2
			           CHARS_NOCASE_ISO_8859_2);
			break;
		case ISO_8859_7:
			add_string(CHARS_LOWER_ISO_8859_7
			           CHARS_LOW_ONLY_ISO_8859_7
			           CHARS_NOCASE_ISO_8859_7);
			break;
		case ISO_8859_15:
			add_string(CHARS_LOWER_ISO_8859_15
			           CHARS_LOW_ONLY_ISO_8859_15
			           CHARS_NOCASE_ISO_8859_15);
			break;
		case KOI8_R:
			add_string(CHARS_LOWER_KOI8_R
			           CHARS_LOW_ONLY_KOI8_R
			           CHARS_NOCASE_KOI8_R);
			break;
		}
		break;

	case 'u': /* upper-case letters */
		/* Rockyou character frequency */
		add_string("AEIOLRNSTMCDBYHUPKGJVFWZXQ");
		break;

	case 'U': /* upper-case letters, non-ASCII only */
		switch (options.internal_cp) {
		case CP437:
			add_string(CHARS_UPPER_CP437
			           CHARS_UP_ONLY_CP437
			           CHARS_NOCASE_CP437);
			break;
		case CP720:
			add_string(CHARS_UPPER_CP720
			           CHARS_UP_ONLY_CP720
			           CHARS_NOCASE_CP720);
			break;
		case CP737:
			add_string(CHARS_UPPER_CP737
			           CHARS_UP_ONLY_CP737
			           CHARS_NOCASE_CP737);
			break;
		case CP850:
			add_string(CHARS_UPPER_CP850
			           CHARS_UP_ONLY_CP850
			           CHARS_NOCASE_CP850);
			break;
		case CP852:
			add_string(CHARS_UPPER_CP852
			           CHARS_UP_ONLY_CP852
			           CHARS_NOCASE_CP852);
			break;
		case CP858:
			add_string(CHARS_UPPER_CP858
			           CHARS_UP_ONLY_CP858
			           CHARS_NOCASE_CP858);
			break;
		case CP866:
			add_string(CHARS_UPPER_CP866
			           CHARS_UP_ONLY_CP866
			           CHARS_NOCASE_CP866);
			break;
		case CP868:
			add_string(CHARS_UPPER_CP868
			           CHARS_UP_ONLY_CP868
			           CHARS_NOCASE_CP868);
			break;
		case CP1250:
			add_string(CHARS_UPPER_CP1250
			           CHARS_UP_ONLY_CP1250
			           CHARS_NOCASE_CP1250);
			break;
		case CP1251:
			add_string(CHARS_UPPER_CP1251
			           CHARS_UP_ONLY_CP1251
			           CHARS_NOCASE_CP1251);
			break;
		case CP1252:
			add_string(CHARS_UPPER_CP1252
			           CHARS_UP_ONLY_CP1252
			           CHARS_NOCASE_CP1252);
			break;
		case CP1253:
			add_string(CHARS_UPPER_CP1253
			           CHARS_UP_ONLY_CP1253
			           CHARS_NOCASE_CP1253);
			break;
		case CP1254:
			add_string(CHARS_UPPER_CP1254
			           CHARS_UP_ONLY_CP1254
			           CHARS_NOCASE_CP1254);
			break;
		case CP1255:
			add_string(CHARS_UPPER_CP1255
			           CHARS_UP_ONLY_CP1255
			           CHARS_NOCASE_CP1255);
			break;
		case CP1256:
			add_string(CHARS_UPPER_CP1256
			           CHARS_UP_ONLY_CP1256
			           CHARS_NOCASE_CP1256);
			break;
		case ISO_8859_1:
			add_string(CHARS_UPPER_ISO_8859_1
			           CHARS_UP_ONLY_ISO_8859_1
			           CHARS_NOCASE_ISO_8859_1);
			break;
		case ISO_8859_2:
			add_string(CHARS_UPPER_ISO_8859_2
			           CHARS_UP_ONLY_ISO_8859_2
			           CHARS_NOCASE_ISO_8859_2);
			break;
		case ISO_8859_7:
			add_string(CHARS_UPPER_ISO_8859_7
			           CHARS_UP_ONLY_ISO_8859_7
			           CHARS_NOCASE_ISO_8859_7);
			break;
		case ISO_8859_15:
			add_string(CHARS_UPPER_ISO_8859_15
			           CHARS_UP_ONLY_ISO_8859_15
			           CHARS_NOCASE_ISO_8859_15);
			break;
		case KOI8_R:
			add_string(CHARS_UPPER_KOI8_R
			           CHARS_UP_ONLY_KOI8_R
			           CHARS_NOCASE_KOI8_R);
			break;
		}
		break;

	case 'd': /* digits */
		/* Rockyou character frequency */
		add_string("1023985467");
		break;

	case 'D': /* digits, non-ASCII only */
		switch (options.internal_cp) {
		case CP437:
			add_string(CHARS_DIGITS_CP437);
			break;
		case CP720:
			add_string(CHARS_DIGITS_CP720);
			break;
		case CP737:
			add_string(CHARS_DIGITS_CP737);
			break;
		case CP850:
			add_string(CHARS_DIGITS_CP850);
			break;
		case CP852:
			add_string(CHARS_DIGITS_CP852);
			break;
		case CP858:
			add_string(CHARS_DIGITS_CP858);
			break;
		case CP866:
			add_string(CHARS_DIGITS_CP866);
			break;
		case CP868:
			add_string(CHARS_DIGITS_CP868);
			break;
		case CP1250:
			add_string(CHARS_DIGITS_CP1250);
			break;
		case CP1251:
			add_string(CHARS_DIGITS_CP1251);
			break;
		case CP1252:
			add_string(CHARS_DIGITS_CP1252);
			break;
		case CP1253:
			add_string(CHARS_DIGITS_CP1253);
			break;
		case CP1254:
			add_string(CHARS_DIGITS_CP1254);
			break;
		case CP1255:
			add_string(CHARS_DIGITS_CP1255);
			break;
		case CP1256:
			add_string(CHARS_DIGITS_CP1256);
			break;
		case ISO_8859_1:
			add_string(CHARS_DIGITS_ISO_8859_1);
			break;
		case ISO_8859_2:
			add_string(CHARS_DIGITS_ISO_8859_2);
			break;
		case ISO_8859_7:
			add_string(CHARS_DIGITS_ISO_8859_7);
			break;
		case ISO_8859_15:
			add_string(CHARS_DIGITS_ISO_8859_15);
			break;
		case KOI8_R:
			add_string(CHARS_DIGITS_KOI8_R);
			break;
		}
		break;

	case 's': /* specials */
		/* Rockyou character frequency */
		add_string("._!-* @#/$,\\&+=?)(';<%\"]~:[^`>{}|");
		break;

	case 'S': /* specials, non-ASCII only */
		switch (options.internal_cp) {
		case CP437:
			add_string(CHARS_PUNCTUATION_CP437
			           CHARS_SPECIALS_CP437
			           CHARS_WHITESPACE_CP437);
			break;
		case CP720:
			add_string(CHARS_PUNCTUATION_CP720
			           CHARS_SPECIALS_CP720
			           CHARS_WHITESPACE_CP720);
			break;
		case CP737:
			add_string(CHARS_PUNCTUATION_CP737
			           CHARS_SPECIALS_CP737
			           CHARS_WHITESPACE_CP737);
			break;
		case CP850:
			add_string(CHARS_PUNCTUATION_CP850
			           CHARS_SPECIALS_CP850
			           CHARS_WHITESPACE_CP850);
			break;
		case CP852:
			add_string(CHARS_PUNCTUATION_CP852
			           CHARS_SPECIALS_CP852
			           CHARS_WHITESPACE_CP852);
			break;
		case CP858:
			add_string(CHARS_PUNCTUATION_CP858
			           CHARS_SPECIALS_CP858
			           CHARS_WHITESPACE_CP858);
			break;
		case CP866:
			add_string(CHARS_PUNCTUATION_CP866
			           CHARS_SPECIALS_CP866
			           CHARS_WHITESPACE_CP866);
			break;
		case CP868:
			add_string(CHARS_PUNCTUATION_CP868
			           CHARS_SPECIALS_CP868
			           CHARS_WHITESPACE_CP868);
			break;
		case CP1250:
			add_string(CHARS_PUNCTUATION_CP1250
			           CHARS_SPECIALS_CP1250
			           CHARS_WHITESPACE_CP1250);
			break;
		case CP1251:
			add_string(CHARS_PUNCTUATION_CP1251
			           CHARS_SPECIALS_CP1251
			           CHARS_WHITESPACE_CP1251);
			break;
		case CP1252:
			add_string(CHARS_PUNCTUATION_CP1252
			           CHARS_SPECIALS_CP1252
			           CHARS_WHITESPACE_CP1252);
			break;
		case CP1253:
			add_string(CHARS_PUNCTUATION_CP1253
			           CHARS_SPECIALS_CP1253
			           CHARS_WHITESPACE_CP1253);
			break;
		case CP1254:
			add_string(CHARS_PUNCTUATION_CP1254
			           CHARS_SPECIALS_CP1254
			           CHARS_WHITESPACE_CP1254);
			break;
		case CP1255:
			add_string(CHARS_PUNCTUATION_CP1255
			           CHARS_SPECIALS_CP1255
			           CHARS_WHITESPACE_CP1255);
			break;
		case CP1256:
			add_string(CHARS_PUNCTUATION_CP1256
			           CHARS_SPECIALS_CP1256
			           CHARS_WHITESPACE_CP1256);
			break;
		case ISO_8859_1:
			add_string(CHARS_PUNCTUATION_ISO_8859_1
			           CHARS_SPECIALS_ISO_8859_1
			           CHARS_WHITESPACE_ISO_8859_1);
			break;
		case ISO_8859_2:
			add_string(CHARS_PUNCTUATION_ISO_8859_2
			           CHARS_SPECIALS_ISO_8859_2
			           CHARS_WHITESPACE_ISO_8859_2);
			break;
		case ISO_8859_7:
			add_string(CHARS_PUNCTUATION_ISO_8859_7
			           CHARS_SPECIALS_ISO_8859_7
			           CHARS_WHITESPACE_ISO_8859_7);
			break;
		case ISO_8859_15:
			add_string(CHARS_PUNCTUATION_ISO_8859_15
			           CHARS_SPECIALS_ISO_8859_15
			           CHARS_WHITESPACE_ISO_8859_15);
			break;
		case KOI8_R:
			add_string(CHARS_PUNCTUATION_KOI8_R
			           CHARS_SPECIALS_KOI8_R
			           CHARS_WHITESPACE_KOI8_R);
			break;
		}
		break;

	case 'B': /* All high-bit */
		add_range(0x80, 0xff);
		break;

	case 'b': /* All (except NULL which we can't handle) */
		add_range(0x01, 0xff);
		break;

	case 'h': /* Lower-case hex */
		add_range('0', '9');
		add_range('a', 'f');
		break;

	case 'H': /* Upper-case hex */
		add_range('0', '9');
		add_range('A', 'F');
		break;

	case 'a': /* Printable ASCII */
		/* Rockyou ASCII character frequency */
		if (fmt_case)
			add_string("ae1ionrls02tm3c98dy54hu6b7kgpjvfwzAxEIOLRNSTMqC.DBYH_!UPKGJ-* @VFWZ#/X$,\\&+=Q?)(';<%\"]~:[^`>{}|");
		else
			add_string("ae1ionrls02tm3c98dy54hu6b7kgpjvfwzxq._!-* @#/$,\\&+=?)(';<%\"]~:[^`>{}|");
		break;

	case 'A': /* All valid chars in codepage (including ASCII) */
		/* Rockyou ASCII character frequency */
		if (fmt_case)
			add_string("ae1ionrls02tm3c98dy54hu6b7kgpjvfwzAxEIOLRNSTMqC.DBYH_!UPKGJ-* @VFWZ#/X$,\\&+=Q?)(';<%\"]~:[^`>{}|");
		else
			add_string("ae1ionrls02tm3c98dy54hu6b7kgpjvfwzxq._!-* @#/$,\\&+=?)(';<%\"]~:[^`>{}|");
		switch (options.internal_cp) {
		case CP437:
			if (fmt_case)
				add_string(CHARS_ALPHA_CP437);
			else
				add_string(CHARS_LOWER_CP437
				           CHARS_LOW_ONLY_CP437
				           CHARS_NOCASE_CP437);
			add_string(CHARS_DIGITS_CP437
			           CHARS_PUNCTUATION_CP437
			           CHARS_SPECIALS_CP437
			           CHARS_WHITESPACE_CP437);
			break;
		case CP720:
			if (fmt_case)
				add_string(CHARS_ALPHA_CP720);
			else
				add_string(CHARS_LOWER_CP720
				           CHARS_LOW_ONLY_CP720
				           CHARS_NOCASE_CP720);
			add_string(CHARS_DIGITS_CP720
			           CHARS_PUNCTUATION_CP720
			           CHARS_SPECIALS_CP720
			           CHARS_WHITESPACE_CP720);
			break;
		case CP737:
			if (fmt_case)
				add_string(CHARS_ALPHA_CP737);
			else
				add_string(CHARS_LOWER_CP737
				           CHARS_LOW_ONLY_CP737
				           CHARS_NOCASE_CP737);
			add_string(CHARS_DIGITS_CP737
			           CHARS_PUNCTUATION_CP737
			           CHARS_SPECIALS_CP737
			           CHARS_WHITESPACE_CP737);
			break;
		case CP850:
			if (fmt_case)
				add_string(CHARS_ALPHA_CP850);
			else
				add_string(CHARS_LOWER_CP850
				           CHARS_LOW_ONLY_CP850
				           CHARS_NOCASE_CP850);
			add_string(CHARS_DIGITS_CP850
			           CHARS_PUNCTUATION_CP850
			           CHARS_SPECIALS_CP850
			           CHARS_WHITESPACE_CP850);
			break;
		case CP852:
			if (fmt_case)
				add_string(CHARS_ALPHA_CP852);
			else
				add_string(CHARS_LOWER_CP852
				           CHARS_LOW_ONLY_CP852
				           CHARS_NOCASE_CP852);
			add_string(CHARS_DIGITS_CP852
			           CHARS_PUNCTUATION_CP852
			           CHARS_SPECIALS_CP852
			           CHARS_WHITESPACE_CP852);
			break;
		case CP858:
			if (fmt_case)
				add_string(CHARS_ALPHA_CP858);
			else
				add_string(CHARS_LOWER_CP858
				           CHARS_LOW_ONLY_CP858
				           CHARS_NOCASE_CP858);
			add_string(CHARS_DIGITS_CP858
			           CHARS_PUNCTUATION_CP858
			           CHARS_SPECIALS_CP858
			           CHARS_WHITESPACE_CP858);
			break;
		case CP866:
			if (fmt_case)
				add_string(CHARS_ALPHA_CP866);
			else
				add_string(CHARS_LOWER_CP866
				           CHARS_LOW_ONLY_CP866
				           CHARS_NOCASE_CP866);
			add_string(CHARS_DIGITS_CP866
			           CHARS_PUNCTUATION_CP866
			           CHARS_SPECIALS_CP866
			           CHARS_WHITESPACE_CP866);
			break;
		case CP868:
			if (fmt_case)
				add_string(CHARS_ALPHA_CP868);
			else
				add_string(CHARS_LOWER_CP868
				           CHARS_LOW_ONLY_CP868
				           CHARS_NOCASE_CP868);
			add_string(CHARS_DIGITS_CP868
			           CHARS_PUNCTUATION_CP868
			           CHARS_SPECIALS_CP868
			           CHARS_WHITESPACE_CP868);
			break;
		case CP1250:
			if (fmt_case)
				add_string(CHARS_ALPHA_CP1250);
			else
				add_string(CHARS_LOWER_CP1250
				           CHARS_LOW_ONLY_CP1250
				           CHARS_NOCASE_CP1250);
			add_string(CHARS_DIGITS_CP1250
			           CHARS_PUNCTUATION_CP1250
			           CHARS_SPECIALS_CP1250
			           CHARS_WHITESPACE_CP1250);
			break;
		case CP1251:
			if (fmt_case)
				add_string(CHARS_ALPHA_CP1251);
			else
				add_string(CHARS_LOWER_CP1251
				           CHARS_LOW_ONLY_CP1251
				           CHARS_NOCASE_CP1251);
			add_string(CHARS_DIGITS_CP1251
			           CHARS_PUNCTUATION_CP1251
			           CHARS_SPECIALS_CP1251
			           CHARS_WHITESPACE_CP1251);
			break;
		case CP1252:
			if (fmt_case)
				add_string(CHARS_ALPHA_CP1252);
			else
				add_string(CHARS_LOWER_CP1252
				           CHARS_LOW_ONLY_CP1252
				           CHARS_NOCASE_CP1252);
			add_string(CHARS_DIGITS_CP1252
			           CHARS_PUNCTUATION_CP1252
			           CHARS_SPECIALS_CP1252
			           CHARS_WHITESPACE_CP1252);
			break;
		case CP1253:
			if (fmt_case)
				add_string(CHARS_ALPHA_CP1253);
			else
				add_string(CHARS_LOWER_CP1253
				           CHARS_LOW_ONLY_CP1253
				           CHARS_NOCASE_CP1253);
			add_string(CHARS_DIGITS_CP1253
			           CHARS_PUNCTUATION_CP1253
			           CHARS_SPECIALS_CP1253
			           CHARS_WHITESPACE_CP1253);
			break;
		case CP1254:
			if (fmt_case)
				add_string(CHARS_ALPHA_CP1254);
			else
				add_string(CHARS_LOWER_CP1254
				           CHARS_LOW_ONLY_CP1254
				           CHARS_NOCASE_CP1254);
			add_string(CHARS_DIGITS_CP1254
			           CHARS_PUNCTUATION_CP1254
			           CHARS_SPECIALS_CP1254
			           CHARS_WHITESPACE_CP1254);
			break;
		case CP1255:
			if (fmt_case)
				add_string(CHARS_ALPHA_CP1255);
			else
				add_string(CHARS_LOWER_CP1255
				           CHARS_LOW_ONLY_CP1255
				           CHARS_NOCASE_CP1255);
			add_string(CHARS_DIGITS_CP1255
			           CHARS_PUNCTUATION_CP1255
			           CHARS_SPECIALS_CP1255
			           CHARS_WHITESPACE_CP1255);
			break;
		case CP1256:
			if (fmt_case)
				add_string(CHARS_ALPHA_CP1256);
			else
				add_string(CHARS_LOWER_CP1256
				           CHARS_LOW_ONLY_CP1256
				           CHARS_NOCASE_CP1256);
			add_string(CHARS_DIGITS_CP1256
			           CHARS_PUNCTUATION_CP1256
			           CHARS_SPECIALS_CP1256
			           CHARS_WHITESPACE_CP1256);
			break;
		case ISO_8859_1:
			if (fmt_case)
				add_string(CHARS_ALPHA_ISO_8859_1);
			else
				add_string(CHARS_LOWER_ISO_8859_1
				           CHARS_LOW_ONLY_ISO_8859_1
				           CHARS_NOCASE_ISO_8859_1);
			add_string(CHARS_DIGITS_ISO_8859_1
			           CHARS_PUNCTUATION_ISO_8859_1
			           CHARS_SPECIALS_ISO_8859_1
			           CHARS_WHITESPACE_ISO_8859_1);
			break;
		case ISO_8859_2:
			if (fmt_case)
				add_string(CHARS_ALPHA_ISO_8859_2);
			else
				add_string(CHARS_LOWER_ISO_8859_2
				           CHARS_LOW_ONLY_ISO_8859_2
				           CHARS_NOCASE_ISO_8859_2);
			add_string(CHARS_DIGITS_ISO_8859_2
			           CHARS_PUNCTUATION_ISO_8859_2
			           CHARS_SPECIALS_ISO_8859_2
			           CHARS_WHITESPACE_ISO_8859_2);
			break;
		case ISO_8859_7:
			if (fmt_case)
				add_string(CHARS_ALPHA_ISO_8859_7);
			else
				add_string(CHARS_LOWER_ISO_8859_7
				           CHARS_LOW_ONLY_ISO_8859_7
				           CHARS_NOCASE_ISO_8859_7);
			add_string(CHARS_DIGITS_ISO_8859_7
			           CHARS_PUNCTUATION_ISO_8859_7
			           CHARS_SPECIALS_ISO_8859_7
			           CHARS_WHITESPACE_ISO_8859_7);
			break;
		case ISO_8859_15:
			if (fmt_case)
				add_string(CHARS_ALPHA_ISO_8859_15);
			else
				add_string(CHARS_LOWER_ISO_8859_15
				           CHARS_LOW_ONLY_ISO_8859_15
				           CHARS_NOCASE_ISO_8859_15);
			add_string(CHARS_DIGITS_ISO_8859_15
			           CHARS_PUNCTUATION_ISO_8859_15
			           CHARS_SPECIALS_ISO_8859_15
			           CHARS_WHITESPACE_ISO_8859_15);
			break;
		case KOI8_R:
			if (fmt_case)
				add_string(CHARS_ALPHA_KOI8_R);
			else
				add_string(CHARS_LOWER_KOI8_R
				           CHARS_LOW_ONLY_KOI8_R
				           CHARS_NOCASE_KOI8_R);
			add_string(CHARS_DIGITS_KOI8_R
			           CHARS_PUNCTUATION_KOI8_R
			           CHARS_SPECIALS_KOI8_R
			           CHARS_WHITESPACE_KOI8_R);
			break;
		default:
			add_range(0x80, 0xff);
		}
		break;
/*
 * Note: To add more cases, also append the symbol to string BUILT_IN_CHARSET.
 */
	default:
		if (john_main_process)
			fprintf(stderr, "Error: Can't nest custom mask placeholder ?%c.\n", p);
		error();
	}

	*o = '\0';
	return out;
}
#undef add_string
#undef add_range

/*
 * Expands all non-custom placeholders in string and returns a new resulting
 * string. ?d is expanded to [0123456789] as opposed to [0-9]. If the outer
 * brackets are already given, as in [?d], output is still [0123456789]
 *
 * This function must pass any escaped characters on, as-is (still escaped).
 * It may also have to ADD escapes to ranges produced from e.g. ?s.
 */
static char* expand_plhdr(char *string, int fmt_case)
{
	static char out[0x8000];
	unsigned char *s = (unsigned char*)string;
	char *d = out;
	int ab = 0;

	if (!string || !*string)
		return string;

	if (*s != '[' || string[strlen(string) - 1] != ']') {
		*d++ = '[';
		ab = 1;
	}
	while (*s && d < &out[sizeof(out) - 1]) {
		if (s[0] == '?' && s[1] == '?') {
			*d++ = '\\';
			*d++ = *s++;
			s++;
		} else
		if (*s == '\\') {
			*d++ = *s++;
			*d++ = *s++;
		} else
		if (s[0] == ']' && s[1] == '[') {
			s += 2;
		} else
		if (*s == '?' && strchr(BUILT_IN_CHARSET, s[1])) {
			char *ps = plhdr2string(s[1], fmt_case);
			while (*ps && d < &out[sizeof(out) - 2]) {
				if (strchr("\\[]?-", ARCH_INDEX(*ps)))
					*d++ = '\\';
				*d++ = *ps++;
			}
			s += 2;
		} else
			*d++ = *s++;
	}
	if (ab)
		*d++ = ']';
	*d = '\0';

	return out;
}

/* Drop length-1-ranges eg. [a] -> a. Modifies string in-place. */
static char* drop1range(char *mask)
{
	char *s = mask, *d = mask;

	while ((*d = *s)) {
		if (*s == '\\')
			*++d = *++s;
		else if (*s == '[') {
			int len = 0;
			char *s1 = s;

			while (s1[1] && !(s1[1] == ']' && s1[0] != '\\')) {
				if (*s1++ != '\\')
					len++;
			}
			if (len && s1[1] == ']') {
				if (len == 1) {
					len = s1 - s;
					s++;
					while (len--)
						*d++ = *s++;
					s++;
				} else
					while (len--)
						*d++ = *s++;
				continue;
			}
		}
		d++;
		s++;
	}

	return mask;
}

/*
 * Return effective length of a mask. \xHH must already be handled.
 *
 * abc -> 3
 * abc?d -> 4
 * abc?l[0-9abcdef] -> 5
 * abc?w -> 3 (the parent-mode word placeholder does not count)
 */
static int mask_len(const char *mask)
{
	int len = 0;
	const char *p = mask;

	while (*p) {
		if (*p == '?') {
			if (p[1] == 'w' || p[1] == 'W') {
				p += 2;
			} else if (strchr(BUILT_IN_CHARSET "?", (int)p[1])) {
				len++;
				p += 2;
			} else {
				len++;
				p++;
			}
		} else if (*p == '\\') {
			len++;
			if (*(++p))
				p++;
		} else if (*p == '[') {
			char *q = strchr(++p, ']');
			const char *m = p;

			while (q && q > m && q[-1] == '\\') {
				m = q;
				q = strchr(++m, ']');
			}

			len++;
			if (q)
				p = q + 1;
		} else {
			len++;
			p++;
		}
	}

	return len;
}

/*
 * valid braces:
 * [abcd], [[[[[abcde], []]abcde]]], [[[ab]cdefr]]
 * invalid braces:
 * [[ab][c], parsed as two separate ranges [[ab] and [c] (no error)
 * [[ab][, error
 *
 * This function must pass any escaped characters on, as-is (still escaped).
 */
static void parse_braces(char *mask, mask_parsed_ctx *parsed_mask)
{
	int i, j ,k;
	int cl_br_enc;

	/* The last element is worst-case boundary for search_stack(). */
	for (i = 0; i <= MAX_NUM_MASK_PLHDR; i++) {
		store_cl(i, -1);
		store_op(i, -1);
	}

	j = k = 0;
	while (j < strlen(mask)) {

		for (i = j; i < strlen(mask); i++) {
			if (mask[i] == '\\')
				i++;
			else
			if (mask[i] == '[')
				break;
		}
		if (i < strlen(mask))
		/* store first opening brace for kth placeholder */
			store_op(k, i);

		cl_br_enc = 0;
		for (i++; i < strlen(mask); i++) {
			if (mask[i] == '\\') {
				i++;
				continue;
			}
			if (mask[i] == ']') {
			/* store last closing brace for kth placeholder */
				store_cl(k, i);
				cl_br_enc = 1;
			}
			if (mask[i] == '[' && cl_br_enc)
				break;
		}

		j = i;
		k++;
		if (k > MAX_NUM_MASK_PLHDR) {
			if (john_main_process)
				fprintf(stderr, "Error: Mask parsing unsuccessful, too many ranges / custom placeholders\n");
			error();
		}
	}

	for (i = 0; i < MAX_NUM_MASK_PLHDR; i++)
		if ((load_op(i) == -1) ^ (load_cl(i) == -1)) {
			if (john_main_process)
				fprintf(stderr, "Error: Mask parsing unsuccessful, missing closing bracket\n");
			error();
		}
}

/*
 * Stores the valid ? placeholders in a stack_qtn
 * valid:
 * -if outside [] braces and
 * -if ? is immediately followed by the identifier such as
 * ?a for all printable ASCII.
 *
 * This function must pass any escaped characters on, as-is (still escaped).
 */
static void parse_qtn(char *mask, mask_parsed_ctx *parsed_mask)
{
	int i, j, k;

	/* The last element is worst-case boundary for search_stack(). */
	for (i = 0; i <= MAX_NUM_MASK_PLHDR; i++)
		parsed_mask->stack_qtn[i] = -1;

	for (i = 0, k = 0; i < strlen(mask); i++) {
		if (mask[i] == '\\') {
			i++;
			continue;
		}
		else if (mask[i] == '?' && i + 1 < strlen(mask) &&
		         strchr(BUILT_IN_CHARSET, ARCH_INDEX(mask[i + 1]))) {
			j = 0;
			while (load_op(j) != -1 && load_cl(j) != -1) {
				if (i > load_op(j) && i < load_cl(j))
					goto cont;
				j++;
			}
			parsed_mask->stack_qtn[k++] = i;
			if (k > MAX_NUM_MASK_PLHDR) {
				if (john_main_process)
					fprintf(stderr, "Error: Mask parsing unsuccessful, too many placeholders\n");
				error();
			}
		}
cont:
		;
	}
}

static int search_stack(mask_parsed_ctx *parsed_mask, int loc)
{
	int t;

	for (t = 0; load_op(t) != -1; t++)
		if (load_op(t) <= loc && load_cl(t) >= loc)
			return load_cl(t);

	for (t = 0; load_qtn(t) != -1; t++)
		if (load_qtn(t) == loc)
			return loc + 1;
	return 0;
}

/*
 * Maps the position of a range in a mask to its actual postion in a key.
 * Offset for wordlist + mask is not taken into account.
 */
static int calc_pos_in_key(const char *mask, mask_parsed_ctx *parsed_mask,
                           int mask_loc)
{
	int i, ret_pos;

	i = ret_pos = 0;
	while (i < mask_loc) {
		int t;

		if (mask[i] == '\\') {
			i++;
			if (i < mask_loc && mask[i] == '\\') {
				i++;
				ret_pos++;
			}
			continue;
		}
		t = search_stack(parsed_mask, i);
#ifdef MASK_DEBUG
		fprintf(stderr, "t=%d\n", t);
#endif
		i = t ? t + 1 : i + 1;
		ret_pos++;
	}

	return ret_pos;
}

#define count(i) cpu_mask_ctx->ranges[i].count

#define fill_range()	  \
	if (a > b) {							\
		for (x = a; x >= b; x--)				\
			if (!memchr((const char*)cpu_mask_ctx->		\
			   ranges[i].chars, x, count(i)))		\
				cpu_mask_ctx->ranges[i].		\
				chars[count(i)++] = x;			\
	} else {							\
		for (x = a; x <= b; x++) 				\
			if (!memchr((const char*)cpu_mask_ctx->		\
			    ranges[i].chars, x, count(i)))		\
				cpu_mask_ctx->ranges[i].		\
				chars[count(i)++] = x;			\
	}

#define add_string(string)						\
	for (p = (char*)string; *p; p++)				\
		cpu_mask_ctx->ranges[i].chars[count(i)++] = *p

#define set_range_start()						\
	for (j = 0; j < count(i); j++)		\
			if (cpu_mask_ctx->ranges[i].chars[0] + j !=	\
			    cpu_mask_ctx->ranges[i].chars[j])		\
				break;					\
	if (j == count(i))				\
		cpu_mask_ctx->ranges[i].start =				\
			cpu_mask_ctx->ranges[i].chars[0]

#define check_n_insert 						\
	if (!memchr((const char*)cpu_mask_ctx->ranges[i].chars,	\
		(int)mask[j], count(i)))			\
		cpu_mask_ctx->ranges[i].chars[count(i)++] = mask[j]

/*
 * This function will finally remove any escape characters (after honoring
 * them of course, if they protected any of our specials).
 * Called by finalize_mask()
 */
static void init_cpu_mask(const char *mask, mask_parsed_ctx *parsed_mask,
                          mask_cpu_context *cpu_mask_ctx, int len)
{
	int i, j, qtn_ctr, op_ctr, cl_ctr;
	char *p;
	int fmt_case = (mask_fmt->params.flags & FMT_CASE);

#ifdef MASK_DEBUG
	fprintf(stderr, "%s(%s, %d) real_max = %dx%d+%d = %d\n", __FUNCTION__, mask, len, options.eff_maxlength, mask_num_qw, mask_add_len, options.eff_maxlength * mask_num_qw + mask_add_len);
#endif

	for (i = 0; i < MAX_NUM_MASK_PLHDR; i++) {
		cpu_mask_ctx->ranges[i].start =
		cpu_mask_ctx->ranges[i].count =
		cpu_mask_ctx->ranges[i].pos =
		cpu_mask_ctx->ranges[i].offset = 0;
		if (mask_cur_len == options.eff_minlength)
		for (j = 0; j <= options.eff_maxlength - options.eff_minlength; j++)
			cpu_mask_ctx->ranges[i].iter[j] = 0;
		cpu_mask_ctx->active_positions[i] = 0;
	}
	cpu_mask_ctx->cpu_count = 0;

	qtn_ctr = op_ctr = cl_ctr = 0;

	for (i = 0; i < MAX_NUM_MASK_PLHDR; i++) {
		int pos;

		if ((unsigned int)load_op(op_ctr) <
		    (unsigned int)load_qtn(qtn_ctr)) {

			pos = calc_pos_in_key(mask, parsed_mask, load_op(op_ctr));
#ifdef MASK_DEBUG
			fprintf(stderr, "load_op(%d) = %u\n", op_ctr, load_op(op_ctr));
			fprintf(stderr, "calc_pos_in_key(%s, %d) = %d\n", mask, load_op(op_ctr), pos);
#endif
			if (!(options.flags & FLG_MASK_STACKED) &&
			    pos >= len && !format_cannot_reset)
				break;
			cpu_mask_ctx->ranges[i].pos = pos;

			for (j = load_op(op_ctr) + 1; j < load_cl(cl_ctr);) {
				int a , b;

				if (mask[j] == '\\') {
					j++;
					if (j >= load_cl(cl_ctr))
						break;
					check_n_insert;
				}
				else if (mask[j] == '-' &&
				         j + 1 < load_cl(cl_ctr) &&
				         j - 1 > load_op(op_ctr) &&
					 mask[j + 1] != '\\') {
					int x;

					if (!memchr((const char*)cpu_mask_ctx->ranges[i].chars,
					            (int)mask[j - 1], count(i)))
						count(i)--;

					a = (unsigned char)mask[j - 1];
					b = (unsigned char)mask[j + 1];

					fill_range();

					j++;
				}
				else if (mask[j] == '-' &&
				         j + 2 < load_cl(cl_ctr) &&
				         j - 1 > load_op(op_ctr) &&
					 mask[j + 1] == '\\') {
					 int x;

					if (!memchr((const char*)cpu_mask_ctx->ranges[i].chars,
					            (int)mask[j - 1], count(i)))
						count(i)--;

					a = (unsigned char)mask[j - 1];
					b = (unsigned char)mask[j + 2];

					fill_range();

					j += 2;
				}
				else check_n_insert;

				j++;
			}

			set_range_start();

			op_ctr++;
			cl_ctr++;
			cpu_mask_ctx->active_positions[i] = 1;
		}
		else if ((unsigned int)load_op(op_ctr) >
		         (unsigned int)load_qtn(qtn_ctr))  {
			int j;

			pos = calc_pos_in_key(mask, parsed_mask, load_qtn(qtn_ctr));
#ifdef MASK_DEBUG
			fprintf(stderr, "load_qtn(%d) = %u\n", qtn_ctr, load_qtn(qtn_ctr));
			fprintf(stderr, "calc_pos_in_key(%s, %d) = %d\n", mask, load_qtn(qtn_ctr), pos);
#endif
			if (!(options.flags & FLG_MASK_STACKED) &&
			    pos >= len && !format_cannot_reset)
				break;
			cpu_mask_ctx->ranges[i].pos = pos;

			add_string(plhdr2string(mask[load_qtn(qtn_ctr) + 1],
			                        fmt_case));
			set_range_start();

			qtn_ctr++;
			cpu_mask_ctx->active_positions[i] = 1;
		}
	}

#ifdef MASK_DEBUG
	fprintf(stderr, "%s() count is %d\n", __FUNCTION__, cpu_mask_ctx->cpu_count);
#endif

	if (restored) {
		memcpy(cpu_mask_ctx->active_idx, restored_ctx.active_idx,
		       sizeof(cpu_mask_ctx->active_idx));
		cpu_mask_ctx->active_count = restored_ctx.active_count;
		cpu_mask_ctx->cpu_count = restored_ctx.cpu_count;
	}

	// Build the flat active index list
	cpu_mask_ctx->active_count = 0;
	for (i = 0; i < MAX_NUM_MASK_PLHDR; i++) {
		if (cpu_mask_ctx->active_positions[i]) {
			cpu_mask_ctx->active_idx[cpu_mask_ctx->active_count++] = i;
		}
	}
	cpu_mask_ctx->cpu_count = cpu_mask_ctx->active_count;

	/* Probability-magnitude threshold (OMEN-style level cutoff), applied as a
	 * UNIFORM per-position char count so the GPU gen simplex stays dup-free.
	 *
	 * Each Markov char carries a quantized -log2(P) "level" (low = probable).
	 * A naive per-CONTEXT cutoff (keep chars with level<=cutoff given prev_char)
	 * makes the fan-out jagged, which a fixed-wheel simplex can only cover by
	 * clamping the unused top ranks -> 50-70% duplicate keys (measured). So we
	 * instead derive a single per-position count T = how many chars clear the
	 * cutoff in the MARGINAL (position-0 / start) distribution, and cap every
	 * context's wheel to T (<= the smallest in-charset fan-out, so rank<rowcnt
	 * always -> no clamp -> zero duplicates). Within the kept top-T the
	 * conditional best-first order is preserved. This is the only magnitude
	 * threshold a product-space simplex can enumerate dup-free on-GPU; a true
	 * per-context (jagged) cutoff needs a variable-radix trie / capped unrank
	 * (~100x slower here) and is left to the host-banded B_EXACT path.
	 * Default = MARKOV_MAXLEVEL: no pruning, byte-identical to before.
	 *   JOHN_GEN_MAXLEVEL=<0..255>  direct level cutoff
	 *   JOHN_GEN_MINP=<prob>        marginal probability floor -> level */
	int gen_level_cutoff = MARKOV_MAXLEVEL;
	{
		const char *ml = getenv("JOHN_GEN_MAXLEVEL");
		const char *mp = getenv("JOHN_GEN_MINP");
		if (ml && *ml) {
			int v = atoi(ml);
			gen_level_cutoff = v < 0 ? 0 : (v > MARKOV_MAXLEVEL ? MARKOV_MAXLEVEL : v);
		} else if (mp && *mp) {
			double tau = atof(mp);
			if (tau > 0.0 && tau <= 1.0) {
				int v = (int)(-log2(tau) * MARKOV_LEVEL_SCALE + 0.5);
				gen_level_cutoff = v < 0 ? 0 : (v > MARKOV_MAXLEVEL ? MARKOV_MAXLEVEL : v);
			}
		}
	}

	/* Precompute filtered Markov tables for O(1) lookup in the hot loop */
	for (i = 0; i < cpu_mask_ctx->active_count; i++) {
		int ri = cpu_mask_ctx->active_idx[i];
		mask_range *r = &cpu_mask_ctx->ranges[ri];
		int valid_idx;
		int m, prev;

		// 1. Build filtered start nodes for this position (level-thresholded).
		//    valid_idx > 0 guard keeps the single most probable in-charset char
		//    even if it exceeds the cutoff, so a position is never left empty.
		valid_idx = 0;
		for (m = 0; m < 256; m++) {
			unsigned char cand = markov_start_nodes[m];
			if (!memchr((const char*)r->chars, cand, r->count))
				continue;
			if (markov_start_level[m] > gen_level_cutoff && valid_idx > 0)
				break;
			pos_markov_start[ri][valid_idx++] = cand;
		}
		pos_markov_start_ceiling[ri] = valid_idx;

		// 2. Build the FULL in-charset transition table (conditional order, NOT
		//    per-context pruned) and track the minimum in-charset fan-out over
		//    all contexts. Leaving the table full keeps every context's rowcnt
		//    uniform, so the uniform wheel cap below never clamps -> dup-free.
		int min_rowcnt = 256;
		for (prev = 0; prev < 256; prev++) {
			valid_idx = 0;
			for (m = 0; m < 256; m++) {
				unsigned char cand = markov_table[prev][m];
				if (memchr((const char*)r->chars, cand, r->count))
					pos_markov_table[ri][prev][valid_idx++] = cand;
			}
			pos_markov_row_counts[ri][prev] = valid_idx;
			if (valid_idx < min_rowcnt)
				min_rowcnt = valid_idx;
		}
		/* Uniform dup-free wheel ceiling: T = marginal-level kept count, capped
		 * to the smallest in-charset fan-out so rank < rowcnt in EVERY context
		 * (no clamp, no duplicates). T < 1 can't happen (start guard keeps >=1). */
		{
			int T = pos_markov_start_ceiling[ri];
			if (T > min_rowcnt) T = min_rowcnt;
			if (T < 1) T = 1;
			pos_markov_trans_ceiling[ri] = T;
		}
	}
}

#undef check_n_insert
#undef count
#undef fill_range

#define SAVE  0
#define RESTORE 1
static void save_restore(mask_cpu_context *cpu_mask_ctx, int range_idx, int ch)
{
	static int bckp_active_count;
	static int bckp_active_idx[MAX_NUM_MASK_PLHDR + 1];
	static int toggle;

	if (range_idx == -1) return;
	if (!ch) {                        /* save */
		bckp_active_count = cpu_mask_ctx->active_count;
		memcpy(bckp_active_idx, cpu_mask_ctx->active_idx, sizeof(bckp_active_idx));
		toggle = 1;
	} else if (toggle) {              /* restore */
		cpu_mask_ctx->active_count = bckp_active_count;
		cpu_mask_ctx->cpu_count  = bckp_active_count;
		memcpy(cpu_mask_ctx->active_idx, bckp_active_idx, sizeof(cpu_mask_ctx->active_idx));
		toggle = 0;
	}
}

/*
 * Truncates mask after range idx.  Called by generate_template_key()
 */
static void truncate_mask(mask_cpu_context *cpu_mask_ctx, int range_idx, int range_idx_end)
{
	int i;

#ifdef MASK_DEBUG
	fprintf(stderr, "%s(%d) max skip %d\n", __FUNCTION__, range_idx, mask_max_skip_loc);
#endif

	if (range_idx < mask_max_skip_loc && mask_max_skip_loc != -1) {
		if (john_main_process)
			fprintf(stderr,
			        "Error: Format's internal mask ranges (first %d positions) cannot be truncated!\n"
			        "       Increase min. length and use some other mode/format for the shorter.\n",
			        mask_max_skip_loc + 1);
		error();
	}

	mask_tot_cand = mask_int_cand.num_int_cand;

	if (range_idx == -1) {
		cpu_mask_ctx->active_count = 0;
		cpu_mask_ctx->cpu_count = 0;
		return;
	}

	// Keep only active_idx up to range_idx
	int new_count = 0;
	for (i = 0; i < cpu_mask_ctx->active_count; i++) {
		if (cpu_mask_ctx->active_idx[i] <= range_idx) {
			cpu_mask_ctx->active_idx[new_count++] = cpu_mask_ctx->active_idx[i];
		}
	}
	cpu_mask_ctx->active_count = new_count;
	cpu_mask_ctx->cpu_count = new_count;

	for (i = 0; i < new_count; i++) {
		mask_tot_cand *= cpu_mask_ctx->ranges[cpu_mask_ctx->active_idx[i]].count;
	}

	/* FIX: Removed the options.node_count division block from here.
	 * Scaling it here caused a double-division cascade downstream. */
}

/*
 * Returns the template of the keys corresponding to the mask.
 * Called by do_mask_crack()
 */
static char *generate_template_key(char *mask, const char *key, int key_len,
                                   mask_parsed_ctx *parsed_mask,
                                   mask_cpu_context *cpu_mask_ctx,
                                   int template_len)
{
	int i, k, t, j, l, offset;

#ifdef MASK_DEBUG
	fprintf(stderr, "%s(%s) ext key \"%s\" ext key_len %d tlen %d\n", __FUNCTION__, mask, key, key_len, template_len);
#endif

	i = 0, k = 0, j = 0, l = 0, offset = 0;

	while (template_key_offsets[l] != -1)
		template_key_offsets[l++] = -1;

	l = 0;
	while (i < strlen(mask)) {
		if ((t = search_stack(parsed_mask, i))) {
			template_key[k++] = '#';
			i = t + 1;
			cpu_mask_ctx->ranges[j++].offset = offset;
		} else if (mask[i] == '\\') {
			i++;
			if (i >= strlen(mask))
				break;
			template_key[k++] = mask[i++];
		} else if (key != NULL && (mask[i + 1] == 'w' ||
			mask[i + 1] == 'W') && mask[i] == '?') {
			template_key_offsets[l++] = ((unsigned char)mask[i + 1] << 16) | k;
			/* Subtract 2 to account for '?w' in mask */
			offset += (key_len - 2);
#ifdef MASK_DEBUG
			memset(&template_key[k], 'w', key_len);
#endif
			k += key_len;
			i += 2;
		} else
			template_key[k++] = mask[i++];

		if (!mask_increments_len && k >= (unsigned int)mask_cur_len) {
			truncate_mask(cpu_mask_ctx, j - 1, template_len - 1);
			k = mask_cur_len;
			break;
		}
	}

	/*
	 * Replace placeholders for any ranges that were handed off to the GPU.
	 * They must not stay as '#' – use the first character of that range.
	 */
	for (i = 0; i < MAX_NUM_MASK_PLHDR; i++) {
		if (cpu_mask_ctx->ranges[i].count > 0 &&
		    !cpu_mask_ctx->active_positions[i]) {
			int p = cpu_mask_ctx->ranges[i].pos +
			        cpu_mask_ctx->ranges[i].offset;
			if (p < max_keylen && template_key[p] == '#')
				template_key[p] = cpu_mask_ctx->ranges[i].chars[0];
		}
	}

	template_key[k] = '\0';

	if (!mask_has_8bit && !(options.flags & FLG_MASK_STACKED)) {
		for (i = 0; i < max_keylen; i++)
			if (template_key[i] & 0x80) {
				mask_has_8bit = 1;
				break;
			}

		for (i = 0; !mask_has_8bit && i < cpu_mask_ctx->active_count; i++) {
			int ri = cpu_mask_ctx->active_idx[i];
			if (cpu_mask_ctx->ranges[ri].pos < max_keylen) {
				for (j = 0; j < cpu_mask_ctx->ranges[ri].count; j++) {
					if (cpu_mask_ctx->ranges[ri].chars[j] & 0x80) {
						mask_has_8bit = 1;
						break;
					}
				}
			}
		}
	}
#ifdef MASK_DEBUG
	fprintf(stderr, "%s(): Template key: '%s'%s\n", __FUNCTION__, template_key, mask_has_8bit && !(options.flags & FLG_MASK_STACKED) ? " has 8-bit" : "");
#endif


	return template_key;
}

/* Handle internal encoding. */
static MAYBE_INLINE char* mask_cp_to_utf8(const char *in)
{
	static char out[PLAINTEXT_BUFFER_SIZE + 1];

	if (mask_has_8bit && options.internal_cp != UTF_8 && options.target_enc == UTF_8)
		return cp_to_utf8_r(in, out, sizeof(out) - 1);

	return (char*)in;
}

static MAYBE_INLINE char* mask_utf8_to_cp(const char *in)
{
	static char out[PLAINTEXT_BUFFER_SIZE + 1];

	if (mask_has_8bit && (options.flags & FLG_MASK_STACKED) && !(options.flags & FLG_RULES_CHK) &&
	    options.internal_cp != UTF_8 && options.target_enc == UTF_8)
		return utf8_to_cp_r(in, out, sizeof(out) - 1);

	return (char*)in;
}

/*
 * ----------------------------------------------------------------------------
 *  Markov-ordered mask enumeration via a simplex-lattice walk
 * ----------------------------------------------------------------------------
 *
 * Each active mask position behaves like a "wheel" whose characters have been
 * pre-sorted by Markov probability (most probable first) in the pos_markov_*
 * tables. A wheel's state is its rank iter[i]: iter[i] == 0 selects the most
 * probable character for that position, iter[i] == 1 the next, and so on, up to
 * count[i] - 1.
 *
 * A candidate's total "improbability" is the rank-sum
 *
 *     K = sum over i of iter[i].
 *
 * We emit candidates from most to least probable, i.e. in order of increasing
 * K. For a fixed K the set of valid rank vectors
 *
 *     { iter : 0 <= iter[i] <= count[i]-1,  sum iter[i] = K }
 *
 * is exactly the integer points of a simplex (the bounded compositions of K) --
 * the "K-layer". The enumeration therefore proceeds one layer at a time: every
 * point of the K=0 layer, then K=1, then K=2, ... Within a layer the points are
 * walked in descending-lexicographic order of iter[].
 *
 * Three routines implement this:
 *
 *   simplex_build_key()  - materialize template_key from the current iter[]
 *                          vector using the Markov tables. Incremental: only
 *                          rebuilds positions from the leftmost changed wheel.
 *   simplex_next_state() - advance iter[] to the next point of the current
 *                          K-layer; when the layer is exhausted, bump K and
 *                          reset to the first point of the next layer.
 *   divide_work()        - for --node distribution, unrank a global offset
 *                          straight into (K, iter[]) with a suffix DP instead
 *                          of stepping simplex_next_state() billions of times.
 * ----------------------------------------------------------------------------
 */

/* markov_table[prev][rank] -> char; markov_start_nodes[rank] -> char for the
 * first position. Both are sorted most-probable-first and filled in elsewhere
 * (when the Markov stats are loaded). */
extern unsigned char markov_table[256][256];
extern unsigned char markov_start_nodes[256];

/* Rebuild the hoisted per-loop cache for the 'limit' active positions of
 * length-loop 'loop'. Cheap (called once per ~INTERLEAVE_STRIDE candidates). */
static inline void prepare_loop_cache(mask_cpu_context * __restrict__ ctx, int loop, int limit) {
	mask_range * __restrict__ ranges = ctx->ranges;
	const int * __restrict__ active_idx = ctx->active_idx;
	for (int i = 0; i < limit; i++) {
		int ri = active_idx[i];
		mask_range *r = &ranges[ri];
		lc_kp[i]        = r->pos + r->offset;
		lc_start[i]     = r->start;
		/* Arithmetic ranges (r->start != 0) aren't Markov, so keep the full
		 * count. Markov positions use the level-thresholded wheel ceiling: the
		 * leftmost active position (i == 0) is materialized from the start-node
		 * table, the rest from the transition table — match each path's ceiling.
		 * With no threshold both ceilings equal r->count (identical behavior). */
		lc_count[i]     = r->start ? r->count
		                  : (i == 0 ? pos_markov_start_ceiling[ri]
		                            : pos_markov_trans_ceiling[ri]);
		lc_chars0[i]    = r->chars[0];
		lc_iter[i]      = &r->iter[loop];
		lc_table[i]     = pos_markov_table[ri];
		lc_start_tab[i] = pos_markov_start[ri];
		lc_rowcnt[i]    = pos_markov_row_counts[ri];
	}
}

/* Materialize template_key from the current iter[] rank vector for length-loop
 * 'loop'. 'start_from' is the leftmost wheel that changed since the previous
 * call (reported by simplex_next_state) so we only rebuild positions that can
 * have moved; everything to its left, including the prev_char feeding it, is
 * still correct in template_key. */
static inline void simplex_build_key(mask_cpu_context * __restrict__ ctx, int loop, int limit, int start_from) {
    if (limit == 0) {
        /* No host-iterated positions for this length: this is the single base
         * key that the GPU internal mask expands into mask_int_cand.num_int_cand
         * candidates (e.g. a 1-char length whose only position was assigned to
         * the GPU). Fill every position within the length - host-active ones and
         * the GPU internal-mask positions (which the GPU overwrites) - with a
         * placeholder first char so strlen(template_key) equals the length. */
        int L = mask_cur_len + loop;
        for (int i = 0; i < ctx->active_count; i++) {
            int ri = ctx->active_idx[i];
            mask_range *r = &ctx->ranges[ri];
            if (r->pos + r->offset < L)
                template_key[r->pos + r->offset] =
                    r->start ? r->start : pos_markov_start[ri][0];
        }
        if (mask_skip_ranges)
            for (int s = 0; s < MASK_FMT_INT_PLHDR && mask_skip_ranges[s] != -1; s++) {
                int ri = mask_skip_ranges[s];
                mask_range *r = &ctx->ranges[ri];
                if (r->pos + r->offset < L)
                    template_key[r->pos + r->offset] =
                        r->start ? r->start : pos_markov_start[ri][0];
            }
        template_key[L] = '\0';
        return;
    }

    int i;

    /* Fill the GPU internal-mask positions (which the GPU overwrites) with a
     * placeholder char. They're not host-iterated, so simplex never writes them,
     * yet template_key is shared across all length-loops: a shorter loop may have
     * left a '\0' at a GPU position, which would truncate this (longer) key right
     * at that position - the host would then hand the GPU a too-short key and the
     * real candidates for this length are never generated. Re-fill every call
     * (few positions) so the key is always the correct length regardless of what
     * a previous loop left behind, and so the Markov prev_char dependency feeding
     * any host position just past a GPU one reads a stable byte. */
    if (mask_skip_ranges) {
        int L = mask_cur_len + loop;
        for (int s = 0; s < MASK_FMT_INT_PLHDR && mask_skip_ranges[s] != -1; s++) {
            mask_range *r = &ctx->ranges[mask_skip_ranges[s]];
            if (r->pos + r->offset < L)
                template_key[r->pos + r->offset] =
                    r->start ? r->start : pos_markov_start[mask_skip_ranges[s]][0];
        }
    }

    /* Incremental rebuild: positions [0, start_from) are unchanged since the
     * previous key (simplex_next_state reported start_from as the leftmost wheel
     * it touched), so their bytes in template_key are already correct and so
     * is the prev_char dependency feeding position start_from. */
    if (start_from <= 0) {
        // CPU unconditionally sets its first handled position
        unsigned char s = lc_start[0], it = *lc_iter[0];
        template_key[lc_kp[0]] = s ? (s + it) : lc_start_tab[0][it];
        start_from = 1;
    }

    // CPU conditionally sets the rest
	for (i = start_from; i < limit; i++) {
		int kp = lc_kp[i];
		unsigned char prev_char = (unsigned char)template_key[kp - 1];
		unsigned char s = lc_start[i];

		if (s) {
			template_key[kp] = s + *lc_iter[i];
		} else {
			int available_choices = lc_rowcnt[i][prev_char];
			int target_idx = *lc_iter[i];

			if (available_choices > 0) {
				// Safety clamp: if the simplex rank exceeds available Markov transitions,
				// fall back to the last available (least probable) valid choice
				if (target_idx >= available_choices)
					target_idx = available_choices - 1;
				template_key[kp] = lc_table[i][prev_char][target_idx];
			} else {
				// Hard fallback: if this previous character has 0 valid transitions in the matrix,
				// fall back to the first character permitted globally by the mask
				template_key[kp] = lc_chars0[i];
			}
		}
	}

    template_key[mask_cur_len + loop] = '\0';
}

/* Advance iter[] to the next point of the current K-layer for length-loop
 * 'loop', writing the leftmost wheel that changed into *changed_from (so the
 * caller can rebuild template_key incrementally). When the layer is exhausted
 * we bump K and reset to the first point of the next layer. Returns 1 only when
 * even the maxed-out configuration cannot reach the new K, i.e. this loop's
 * whole keyspace is exhausted. */
static inline int simplex_next_state(mask_cpu_context * __restrict__ ctx, int loop, int limit, int *changed_from) {
	int i, j;
	int weight_to_redistribute = 0;

	/* Anti-diagonal (simplex-lattice) state advance: scan right-to-left for the
	 * first position where we can shift one unit of "rank weight" from a left
	 * wheel to its right neighbour, keeping the total sum K constant. */
	for (i = limit - 2; i >= 0; i--) {

		// Can we shift weight from position i to position i+1 for this specific loop?
		if (*lc_iter[i] > 0 && *lc_iter[i + 1] < lc_count[i + 1] - 1) {

			(*lc_iter[i])--;        // Decrement left wheel rank
			(*lc_iter[i + 1])++;    // Increment right wheel rank

			/* Collect all residual rank weights from wheels further to the right */
			for (j = i + 2; j < limit; j++) {
				weight_to_redistribute += *lc_iter[j];
				*lc_iter[j] = 0; // Reset right-side wheel back to baseline
			}

			/* Pour the collected weight back as far left as possible to start the next permutation */
			j = i + 1;
			while (weight_to_redistribute > 0 && j < limit) {
				int max_allowed = lc_count[j] - 1 - *lc_iter[j];
				int add = (weight_to_redistribute > max_allowed) ? max_allowed : weight_to_redistribute;

				*lc_iter[j] += add;
				weight_to_redistribute -= add;
				j++;
			}

			/* Leftmost wheel whose rank changed is the pivot i, so the key
			 * only needs rebuilding from position i onward. */
			*changed_from = i;
			return 0; // Successfully advanced to the next state within the current K-layer
		}
	}

	/* If the loop completes, the current Markov Rank-Sum layer (K) for this loop is completely exhausted.
	 * Advance to the next probability tier (K + 1). */
	ctx->current_k[loop]++;

	/* Reset the state arrays to the absolute first configuration of the new K layer.
	 * We pack the target sum into the leftmost wheels up to their individual 'count' capacities. */
	int remaining_k = ctx->current_k[loop];
	for (i = 0; i < limit; i++) {
		int max_idx = lc_count[i] - 1;

		if (remaining_k <= max_idx) {
			*lc_iter[i] = remaining_k;
			remaining_k = 0;
		} else {
			*lc_iter[i] = max_idx;
			remaining_k -= max_idx;
		}
	}

	/* If remaining_k is still greater than 0, even with every single active position maxed out,
	 * the entire keyspace for this interleaved loop length is completely exhausted. */
	if (remaining_k > 0) {
		return 1;
	}

	/* New K-layer resets every wheel, so the whole key must be rebuilt. */
	*changed_from = 0;
	return 0; // Successfully wrapped around to the beginning of the next K-layer
}

static int get_loop(mask_cpu_context *ctx, int loop) {
    int limit = 0;
    while (limit < ctx->active_count &&
           ctx->ranges[ctx->active_idx[limit]].pos < mask_cur_len + loop)
        limit++;
    return limit;
}

/* Weighted round-robin: give each length-loop a per-visit stride proportional
 * to P(length) from the Markov corpus, so the common lengths get more airtime
 * than the rare ones (passwords cluster at ~7-10 chars). loop l is word length
 * mask_cur_len + l; its weight is markov_len_count[len] (from markov_tables.h),
 * scaled so the most frequent active length gets the full INTERLEAVE_STRIDE and
 * the rest get proportionally smaller strides, floored at 1 so no length is
 * ever fully starved. An all-zero histogram (no length data) yields a flat
 * INTERLEAVE_STRIDE for every loop, i.e. plain round-robin. */
static void compute_loop_strides(int max_loop, int *loop_stride) {
    unsigned long long maxc = 0;
    for (int l = 0; l <= max_loop; l++) {
        int len = mask_cur_len + l;
        unsigned long long c = (len >= 0 && len < MARKOV_MAXLEN) ? markov_len_count[len] : 0;
        if (c > maxc) maxc = c;
    }
    for (int l = 0; l <= max_loop; l++) {
        if (maxc == 0) {
            loop_stride[l] = INTERLEAVE_STRIDE;
            continue;
        }
        int len = mask_cur_len + l;
        unsigned long long c = (len >= 0 && len < MARKOV_MAXLEN) ? markov_len_count[len] : 0;
        unsigned long long s = ((unsigned long long)INTERLEAVE_STRIDE * c) / maxc;
        loop_stride[l] = s < 1 ? 1 : (int)s;
    }
}

uint64_t global_idx = 0;   /* declared once at top of generate_keys */

/* A length-loop with no host-iterated positions (limit == 0) still has one host
 * "base" candidate (the empty product) that the GPU internal mask expands into
 * mask_int_cand.num_int_cand candidates - e.g. a 1-char length whose only
 * position was assigned to the GPU. Return 1 if that base key must be emitted:
 * there is a GPU internal mask and every GPU placeholder fits within the length
 * (otherwise the GPU would write past the key, so the length is genuinely
 * unenumerable and is skipped). */
static inline int loop_emits_gpu_base(mask_cpu_context *ctx, int loop) {
	int L = mask_cur_len + loop;
	int s, any = 0;

	if (L <= 0 || mask_int_cand.num_int_cand <= 1 || !mask_skip_ranges)
		return 0;
	for (s = 0; s < MASK_FMT_INT_PLHDR && mask_skip_ranges[s] != -1; s++) {
		any = 1;
		if (ctx->ranges[mask_skip_ranges[s]].pos +
		    ctx->ranges[mask_skip_ranges[s]].offset >= L)
			return 0;
	}
	return any;
}

static int generate_keys(mask_cpu_context *cpu_mask_ctx, uint64_t *my_candidates) {
	char key_e[PLAINTEXT_BUFFER_SIZE];
	char *key;
	int max_loop = options.eff_maxlength - mask_cur_len;
	int *loop_done = mem_calloc(max_loop + 1, sizeof(int));
	int n_active = max_loop + 1;
	int idx = 0;
	int loop_stride[MASK_MAX_INC_LEN];

	compute_loop_strides(max_loop, loop_stride);

	if (!options.node_count && !restored) {
		for (int l = 0; l <= max_loop; l++) {
			cpu_mask_ctx->current_k[l] = 0;
			for (int i = 0; i < cpu_mask_ctx->active_count; i++)
				cpu_mask_ctx->ranges[cpu_mask_ctx->active_idx[i]].iter[l] = 0;
		}
	}

#define process_key(key_i) \
		do { \
			key = key_i; \
			if (!f_filter || ext_filter_body(key_i, key = key_e)) \
				if (crk_process_key(mask_cp_to_utf8(key))) \
					return 1; \
		} while(0)

	while (n_active > 0) {
		while (loop_done[idx]) {
			idx++;
			if (idx > max_loop) idx = 0;
		}
		int loop = idx;
		int stride_count = 0;
		int limit = get_loop(cpu_mask_ctx, loop);
		/* Block distribution: this node has a finite per-loop budget. A loop
		 * with empty/exhausted budget is done. */
		int node_budgeted = options.node_count && !(options.flags & FLG_MASK_STACKED);
		if (limit == 0 || (node_budgeted && node_loop_cand[loop] == 0)) {
			/* A length whose only in-range positions are GPU-internal has no
			 * host work (limit == 0) but still has one base key to emit so the
			 * GPU can expand it (e.g. a 1-char length sitting entirely on the
			 * GPU). Emit it once, unless this node's block budget is spent. */
			if (limit == 0 && !(node_budgeted && node_loop_cand[loop] == 0) &&
			    loop_emits_gpu_base(cpu_mask_ctx, loop)) {
				simplex_build_key(cpu_mask_ctx, loop, 0, 0);
				process_key(template_key);
				global_idx++;
				if (node_budgeted)
					--node_loop_cand[loop];
			}
			loop_done[loop] = 1;
			n_active--;
			idx++;
			if (idx > max_loop) idx = 0;
			continue;
		}

		/* Hoist this loop's active-range data; first key of every stride needs
		 * a full rebuild because the previous stride may have been a different
		 * length-loop. */
		prepare_loop_cache(cpu_mask_ctx, loop, limit);
		int dirty_from = 0;

		while (stride_count < loop_stride[loop] && !loop_done[loop]) {
			simplex_build_key(cpu_mask_ctx, loop, limit, dirty_from);

			process_key(template_key);
			global_idx++;

			/* Stop this loop once this node's contiguous block is spent. */
			if (node_budgeted && --node_loop_cand[loop] == 0) {
				loop_done[loop] = 1;
				n_active--;
				break;
			}

			if (simplex_next_state(cpu_mask_ctx, loop, limit, &dirty_from)) {
				loop_done[loop] = 1;
				n_active--;
				break;
			}

			stride_count++;
		}

		/* Fast lookahead index increment replacing modulo operator */
		idx++;
		if (idx > max_loop) idx = 0;
	}

	MEM_FREE(loop_done);
	return 0;
#undef process_key
}

static int bench_generate_keys(mask_cpu_context *cpu_mask_ctx, uint64_t *my_candidates) {
	int max_loop = options.eff_maxlength - mask_cur_len;
	int *loop_done = mem_calloc(max_loop + 1, sizeof(int));
	int n_active = max_loop + 1;
	int idx = 0;
	int loop_stride[MASK_MAX_INC_LEN];

	compute_loop_strides(max_loop, loop_stride);

	/* FIX: Only wipe iterators if we aren't distributed across nodes and not restoring a checkpoint.
	 * This prevents destroying the unique offsets assigned to each fork. */
	if (!options.node_count && !restored) {
		for (int l = 0; l <= max_loop; l++) {
			cpu_mask_ctx->current_k[l] = 0;
			for (int i = 0; i < cpu_mask_ctx->active_count; i++)
				cpu_mask_ctx->ranges[cpu_mask_ctx->active_idx[i]].iter[l] = 0;
		}
	}

#define process_key(key)                                            \
		mask_fmt->methods.set_key(mask_cp_to_utf8(template_key),        \
								mask_bench_index++);                  \
		if (mask_bench_index >= mask_fmt->params.max_keys_per_crypt) {  \
			mask_bench_index = 0;                                       \
			return 1;                                                   \
		}

	while (n_active > 0) {
		while (loop_done[idx]) {
			idx++;
			if (idx > max_loop) idx = 0;
		}
		int loop = idx;
		int stride_count = 0;
		int limit = get_loop(cpu_mask_ctx, loop);
		if (limit == 0) {
			/* GPU-only length: emit its single base key (see generate_keys). */
			if (loop_emits_gpu_base(cpu_mask_ctx, loop)) {
				simplex_build_key(cpu_mask_ctx, loop, 0, 0);
				process_key(template_key);
			}
			loop_done[loop] = 1;
			n_active--;
			idx++;
			if (idx > max_loop) idx = 0;
			continue;
		}

		prepare_loop_cache(cpu_mask_ctx, loop, limit);
		int dirty_from = 0;

		while (stride_count < loop_stride[loop] && !loop_done[loop]) {
			simplex_build_key(cpu_mask_ctx, loop, limit, dirty_from);

			process_key(template_key);

			if (simplex_next_state(cpu_mask_ctx, loop, limit, &dirty_from)) {
				loop_done[loop] = 1;
				n_active--;
				break;
			}

			stride_count++;

			if (options.node_count && !(options.flags & FLG_MASK_STACKED) &&
			    !(*my_candidates)--) {
				goto done;
			}
		}

		/* Fast lookahead index increment replacing modulo operator */
		idx++;
		if (idx > max_loop) idx = 0;
	}

done:
	MEM_FREE(loop_done);
	return 0;
#undef process_key
}

/* Skips iteration for positions stored in arr (internal mask ranges). */
static void skip_position(mask_cpu_context *cpu_mask_ctx, int *arr)
{
	if (arr != NULL) {
		int k = 0;
		/* FIX: Check absolute index against MAX_NUM_MASK_PLHDR, not active_count */
		while (k < MASK_FMT_INT_PLHDR && arr[k] >= 0 && arr[k] < MAX_NUM_MASK_PLHDR) {
			int idx = arr[k];
			cpu_mask_ctx->active_positions[idx] = 0;
			k++;
		}
	}

	// Rebuild active_idx from active_positions
	cpu_mask_ctx->active_count = 0;
	for (int i = 0; i < MAX_NUM_MASK_PLHDR; i++) {
		if (cpu_mask_ctx->active_positions[i]) {
			cpu_mask_ctx->active_idx[cpu_mask_ctx->active_count++] = i;
		}
	}
	cpu_mask_ctx->cpu_count = cpu_mask_ctx->active_count;
}

/*
 * Divide a work between multiple nodes.  Called by finalize_mask()
 */
/*
 * Divide a work between multiple nodes.  Called by finalize_mask()
 * FIX: Replaced float distribution with pure integer math to prevent overlap.
 */
static uint64_t divide_work(mask_cpu_context *cpu_mask_ctx)
{
	uint64_t my_candidates = 0;
	int i, j;

#ifdef MASK_DEBUG
	fprintf(stderr, "%s()\n", __FUNCTION__);
#endif

	/* Block distribution: each length-loop has its own keyspace and is split
	 * independently into node_count contiguous blocks. This node gets block
	 * [node_min-1 .. node_max-1] of every loop, fast-forwarded to its start.
	 * Per-loop budgets are stored in node_loop_cand[] so generate_keys can
	 * stop each loop at its block boundary (not at full keyspace exhaustion). */
	int max_loop = options.eff_maxlength - mask_cur_len;
	for (j = 0; j <= max_loop; j++) {
		node_loop_cand[j] = 0;

		int limit = 0;
		while (limit < cpu_mask_ctx->active_count &&
		       cpu_mask_ctx->ranges[cpu_mask_ctx->active_idx[limit]].pos < mask_cur_len + j) {
			limit++;
		}

		if (limit > 0) {
			/* This loop's full keyspace and this node's contiguous slice. */
			uint64_t loop_total = 1;
			for (i = 0; i < limit; i++)
				loop_total *= cpu_mask_ctx->ranges[cpu_mask_ctx->active_idx[i]].count;

			uint64_t node_share = loop_total / options.node_count;
			uint64_t remainder  = loop_total % options.node_count;
			uint64_t start = node_share * (options.node_min - 1) +
			    ((uint64_t)(options.node_min - 1) < remainder ? (uint64_t)(options.node_min - 1) : remainder);
			uint64_t end   = node_share * options.node_max +
			    ((uint64_t)options.node_max < remainder ? (uint64_t)options.node_max : remainder);
			uint64_t offset = start;

			node_loop_cand[j] = end - start;
			my_candidates += end - start;

			if (node_loop_cand[j] == 0)
				continue;
			// Calculate the absolute maximum K (worst-case probability penalty) for this length
			int max_k = 0;
			for (i = 0; i < limit; i++) {
				max_k += cpu_mask_ctx->ranges[cpu_mask_ctx->active_idx[i]].count - 1;
			}

			/* Combinatorial tier-jump + unrank (O(limit*K), no linear fast-forward) */
			// 4a. Build a SUFFIX DP: suf[i][s] = number of valid iter[]
			// configurations of wheels i..limit-1 whose rank-sum equals s.
			// suf[0][k] is then the size of the whole K=k probability layer.
			int K_SIZE = max_k + 1;
			uint64_t *suf = mem_calloc((limit + 1) * K_SIZE, sizeof(uint64_t));
			suf[limit * K_SIZE + 0] = 1;
			for (i = limit - 1; i >= 0; i--) {
				int C = cpu_mask_ctx->ranges[cpu_mask_ctx->active_idx[i]].count;
				for (int s = 0; s <= max_k; s++) {
					uint64_t acc = 0;
					for (int v = 0; v < C && v <= s; v++)
						acc += suf[(i + 1) * K_SIZE + (s - v)];
					suf[i * K_SIZE + s] = acc;
				}
			}

			// 4b. Skip whole probability tiers until the offset lands inside one.
			uint64_t fw_offset = offset;
			int target_k = 0;
			while (target_k <= max_k && fw_offset >= suf[0 * K_SIZE + target_k]) {
				fw_offset -= suf[0 * K_SIZE + target_k];
				target_k++;
			}
			cpu_mask_ctx->current_k[j] = target_k;

			// 4c. Unrank fw_offset directly into the iter[] vector. simplex_next_state
			// walks each K-layer in *descending-lexicographic* order of iter[], so
			// at each wheel we try the largest value first and subtract the count
			// of suffix configurations it skips. This replaces a loop of up to
			// billions of simplex_next_state() calls with limit*K arithmetic.
			int remaining_k = target_k;
			uint64_t rank = fw_offset;
			for (i = 0; i < limit; i++) {
				int ri = cpu_mask_ctx->active_idx[i];
				int C = cpu_mask_ctx->ranges[ri].count;
				int vmax = remaining_k < (C - 1) ? remaining_k : (C - 1);
				int v;
				for (v = vmax; v >= 0; v--) {
					uint64_t cnt = suf[(i + 1) * K_SIZE + (remaining_k - v)];
					if (rank < cnt)
						break;
					rank -= cnt;
				}
				cpu_mask_ctx->ranges[ri].iter[j] = v;
				remaining_k -= v;
			}

			MEM_FREE(suf);
		}
	}

	if (!my_candidates && !mask_increments_len) {
		if (john_main_process)
			fprintf(stderr, "%u: Error: Insufficient work. Cannot distribute work among nodes!\n", options.node_min);
		error();
	}

	return my_candidates;
}

/* ----------------------------------------------------------------------------
 *  GPU K-ordered password generation (host side)
 * ----------------------------------------------------------------------------
 * Builds the upload-ready Markov tables (once per mask config) and the per-
 * length-loop suffix-DP (the same combinatorial unrank divide_work() uses), and
 * provides mask_gpu_unrank_key() - the exact host mirror of the kernel
 * generator, used by the format's get_key() to reconstruct cracked plaintext.
 */
int mask_gpu_gen = 0;
/* Hybrid head/tail materialize (JOHN_GEN_HEAD=H, mirror of the GPU kernel's
 * -D GEN_HEAD path; JOHN_GEN_MARGINAL is the H==0 alias). -1 = full conditional
 * (disabled). The iter[] simplex point is still K-ordered; positions i >= H are
 * materialized from their own probability-sorted charset (startv[i]) independent
 * of the previous char, while positions 1..H-1 keep the conditional bigram model.
 * Set in mask_gpu_build_tables(). */
static int mask_gpu_head = -1;
int mask_gpu_max_loop = -1;
/* Bumped whenever the upload-ready tables are rebuilt; the format re-uploads
 * when it sees a new value. */
unsigned mask_gpu_serial = 0;
/* Cursor the format reads to know which block to generate. */
int mask_gpu_cur_loop = 0;
uint64_t mask_gpu_cur_base = 0;
/* When set (MASK_GPU_CPU env), generate on the host via mask_gpu_unrank_key and
 * push through crk_process_key - validates the unrank/ordering against the
 * format's normal crypt path. Resolved eagerly in mask_init() (which runs before
 * the format's reset()), so the format can read it to disable its gen path. */
int mask_gpu_cpu_validate = 0;
static mask_gpu_tables gpu_tabs;
static mask_gpu_loop gpu_loops[MASK_MAX_INC_LEN];

const mask_gpu_tables *mask_gpu_get_tables(void) { return &gpu_tabs; }

const mask_gpu_loop *mask_gpu_get_loop(int loop)
{
	if (loop < 0 || loop > mask_gpu_max_loop || !gpu_loops[loop].valid)
		return NULL;
	return &gpu_loops[loop];
}

static mask_gpu_plan gpu_plan;
static mask_gpu_seg *gpu_segs;
static int gpu_segs_cap;

/* Cap on the number of round-robin segments, to bound the device segment buffer
 * and the per-candidate binary search. Once this many segments have been laid
 * down, each remaining length's tail is emitted as one big segment instead of
 * being sliced further: the finely interleaved front (tens of thousands of
 * rounds = billions of candidates, far past any session) is what matters; the
 * unreachable deep tail of the long lengths need not be interleaved. */
#define MASK_GPU_NSEG_MAX (1 << 18)

const mask_gpu_plan *mask_gpu_get_plan(void) { return &gpu_plan; }

static mask_gpu_seg *plan_push(int *nseg)
{
	if (*nseg >= gpu_segs_cap) {
		gpu_segs_cap = gpu_segs_cap ? gpu_segs_cap * 2 : 4096;
		gpu_segs = mem_realloc(gpu_segs,
		                       (size_t)gpu_segs_cap * sizeof(*gpu_segs));
	}
	return &gpu_segs[(*nseg)++];
}

/* Lay the active length-loops out in the CPU's weighted round-robin order: each
 * length is sliced into Markov-weighted chunks (compute_loop_strides) and the
 * chunks are concatenated round-robin, so the virtual space interleaves lengths
 * while each length advances in K order. Remaining length tails are emitted as
 * one big segment each once only a single length is left (nothing to interleave
 * with) or the segment budget MASK_GPU_NSEG_MAX is reached (the deep tail beyond
 * that point is unreachable in practice, so it need not be finely sliced). */
static void mask_gpu_build_plan(void)
{
	int max_loop = mask_gpu_max_loop, loop, nseg = 0, n_active = 0;
	uint64_t cur[MASK_MAX_INC_LEN + 1], end[MASK_MAX_INC_LEN + 1];
	uint64_t suf_off[MASK_MAX_INC_LEN + 1], stride[MASK_MAX_INC_LEN + 1];
	int active[MASK_MAX_INC_LEN + 1], loop_stride[MASK_MAX_INC_LEN];
	uint64_t vbase = 0, soff = 0;

	compute_loop_strides(max_loop, loop_stride);

	for (loop = 0; loop <= max_loop; loop++) {
		const mask_gpu_loop *gl = mask_gpu_get_loop(loop);
		uint64_t start = 0, e = 0;

		active[loop] = 0;
		cur[loop] = end[loop] = 0;
		suf_off[loop] = soff;
		stride[loop] = 1;

		if (!gl || gl->total == 0)
			continue;
		e = gl->total;
		if (options.node_count && !(options.flags & FLG_MASK_STACKED)) {
			uint64_t share = gl->total / options.node_count;
			uint64_t rem   = gl->total % options.node_count;
			uint64_t nmin1 = options.node_min - 1;
			start = share * nmin1 + (nmin1 < rem ? nmin1 : rem);
			e     = share * options.node_max +
			        ((uint64_t)options.node_max < rem ? options.node_max : rem);
		}
		if (e <= start)
			continue;

		cur[loop] = start;
		end[loop] = e;
		active[loop] = 1;
		n_active++;
		soff += (uint64_t)(gl->limit + 1) * gl->ksize;
		stride[loop] = loop_stride[loop] < 1 ? 1 : (uint64_t)loop_stride[loop];
	}
	gpu_plan.suf_total = soff;

	{
		int coarsen = 0;
		/* Geometric tail growth factor per round, applied once the segment budget
		 * is near. Tunable via JOHN_TAIL_GROW (default 2, min 2). */
		const char *ge = getenv("JOHN_TAIL_GROW");
		uint64_t grow = (ge && atoi(ge) >= 2) ? (uint64_t)atoi(ge) : 2;
		/* Geometric tail on by default; JOHN_GEOTAIL=0 restores the old behavior
		 * (flush each remaining length's whole tail as one single-length segment
		 * at the budget) for A/B comparison. */
		const char *gt = getenv("JOHN_GEOTAIL");
		int geotail = !(gt && atoi(gt) == 0);

	while (n_active > 0) {
		/* Only one length left (or geotail disabled and the segment budget is
		 * reached): emit each remaining length's whole tail as a single segment. */
		if (n_active == 1 || (!geotail && nseg + n_active > MASK_GPU_NSEG_MAX)) {
			for (loop = 0; loop <= max_loop; loop++) {
				const mask_gpu_loop *gl;
				mask_gpu_seg *sg;

				if (!active[loop])
					continue;
				gl = mask_gpu_get_loop(loop);
				sg = plan_push(&nseg);
				sg->vbase   = vbase;
				sg->vcnt    = end[loop] - cur[loop];
				sg->lstart  = cur[loop];
				sg->suf_off = suf_off[loop];
				sg->loop    = loop;
				sg->limit   = gl->limit;
				sg->len     = gl->len;
				sg->max_k   = gl->max_k;
				sg->ksize   = gl->ksize;
				vbase += end[loop] - cur[loop];
				cur[loop] = end[loop];
			}
			break;
		}

		/* Approaching the segment budget: switch the still-multi-length tail to
		 * geometric (exponentially growing) chunks instead of flushing each
		 * length as one giant single-length segment. This keeps every active
		 * length interleaved all the way down - long-length, high-probability
		 * candidates are no longer stranded behind a shorter length's entire
		 * tail - while covering each remaining keyspace in O(log) rounds, so
		 * nseg stays around MASK_GPU_NSEG_MAX. */
		if (geotail && !coarsen &&
		    nseg + n_active > MASK_GPU_NSEG_MAX - MASK_GPU_NSEG_MAX / 8)
			coarsen = 1;

		for (loop = 0; loop <= max_loop; loop++) {
			const mask_gpu_loop *gl;
			mask_gpu_seg *sg;
			uint64_t chunk;

			if (!active[loop])
				continue;
			chunk = end[loop] - cur[loop];
			if (chunk > stride[loop])
				chunk = stride[loop];

			gl = mask_gpu_get_loop(loop);
			sg = plan_push(&nseg);
			sg->vbase   = vbase;
			sg->vcnt    = chunk;
			sg->lstart  = cur[loop];
			sg->suf_off = suf_off[loop];
			sg->loop    = loop;
			sg->limit   = gl->limit;
			sg->len     = gl->len;
			sg->max_k   = gl->max_k;
			sg->ksize   = gl->ksize;

			vbase += chunk;
			cur[loop] += chunk;
			if (cur[loop] >= end[loop]) {
				active[loop] = 0;
				n_active--;
			}
		}

		/* Grow strides geometrically once coarsening so the deep tail is covered
		 * in logarithmically many still-interleaved rounds. */
		if (coarsen) {
			for (loop = 0; loop <= max_loop; loop++)
				if (active[loop] && stride[loop] <= UINT64_MAX / grow)
					stride[loop] *= grow;
		}
	}
	}

	gpu_plan.seg = gpu_segs;
	gpu_plan.nseg = nseg;
	gpu_plan.total = vbase;

	if (getenv("MASK_GPU_PLAN")) {
		int l, s, segcnt[MASK_MAX_INC_LEN + 1] = {0};
		uint64_t segspan[MASK_MAX_INC_LEN + 1] = {0};

		fprintf(stderr, "[PLAN] nseg=%d total=%"PRIu64" max_loop=%d strides:",
			nseg, gpu_plan.total, max_loop);
		for (l = 0; l <= max_loop; l++)
			fprintf(stderr, " L%d=%d", mask_cur_len + l, loop_stride[l]);
		fprintf(stderr, "\n[PLAN] first segs (loop:len vbase vcnt): ");
		for (s = 0; s < nseg && s < 24; s++)
			fprintf(stderr, "[%d:%d @%"PRIu64" x%"PRIu64"] ",
				gpu_segs[s].loop, gpu_segs[s].len,
				gpu_segs[s].vbase, gpu_segs[s].vcnt);
		for (s = 0; s < nseg; s++) {
			segcnt[gpu_segs[s].loop]++;
			segspan[gpu_segs[s].loop] += gpu_segs[s].vcnt;
		}
		fprintf(stderr, "\n[PLAN] per-length (len: nseg span):");
		for (l = 0; l <= max_loop; l++)
			fprintf(stderr, " %d:%d/%"PRIu64, mask_cur_len + l,
				segcnt[l], segspan[l]);
		fprintf(stderr, "\n[PLAN] last segs (loop:len vcnt): ");
		for (s = (nseg > 18 ? nseg - 18 : 0); s < nseg; s++)
			fprintf(stderr, "[%d:%d x%"PRIu64"] ",
				gpu_segs[s].loop, gpu_segs[s].len, gpu_segs[s].vcnt);
		fprintf(stderr, "\n");
	}
}

void mask_gpu_virt_to_loop(uint64_t v, int *loop, uint64_t *g)
{
	int lo = 0, hi = gpu_plan.nseg - 1, s = 0;

	/* Largest segment with vbase <= v (segments are in ascending vbase order). */
	while (lo <= hi) {
		int mid = (lo + hi) >> 1;

		if (gpu_plan.seg[mid].vbase <= v) {
			s = mid;
			lo = mid + 1;
		} else
			hi = mid - 1;
	}
	if (gpu_plan.nseg) {
		*loop = gpu_plan.seg[s].loop;
		*g = gpu_plan.seg[s].lstart + (v - gpu_plan.seg[s].vbase);
	} else {
		*loop = 0;
		*g = 0;
	}
}

/* Compact the per-position Markov tables of the current mask into flat,
 * upload-ready arrays indexed by generated-position (left-to-right key order).
 * template_key must already hold the finalized max-length template (literals in
 * place) when this is called. */
static void mask_gpu_build_tables(mask_cpu_context *ctx)
{
	int npos = ctx->active_count;
	int i, p, maxL = options.eff_maxlength;

	{
		const char *he = getenv("JOHN_GEN_HEAD");

		if (he)
			mask_gpu_head = atoi(he);    /* H >= 0 conditional positions */
		else if (getenv("JOHN_GEN_MARGINAL"))
			mask_gpu_head = 0;           /* pure marginal alias */
		else
			mask_gpu_head = -1;          /* full conditional */
	}

	MEM_FREE(gpu_tabs.table);
	MEM_FREE(gpu_tabs.uint_table);
	MEM_FREE(gpu_tabs.startv);
	MEM_FREE(gpu_tabs.rowcnt);
	MEM_FREE(gpu_tabs.littmpl);
	memset(&gpu_tabs, 0, sizeof(gpu_tabs));

	gpu_tabs.npos = npos;
	gpu_tabs.table  = mem_alloc((size_t)npos * 256 * 256);
	gpu_tabs.uint_table = mem_alloc((size_t)npos * 256 * 64 * sizeof(uint32_t));
	gpu_tabs.startv = mem_alloc((size_t)npos * 256);
	gpu_tabs.rowcnt = mem_alloc((size_t)npos * 256);

	/* Local-table eligibility, accumulated across positions (see mask.h). */
	int ltab_ok = (npos >= 1), ltab_base = -1, ltab_nc = -1;

	for (i = 0; i < npos; i++) {
		int ri = ctx->active_idx[i];
		mask_range *r = &ctx->ranges[ri];

		gpu_tabs.keypos[i] = r->pos + r->offset;
		gpu_tabs.count[i]  = r->count;
		gpu_tabs.cstart[i] = r->start;
		gpu_tabs.chars0[i] = r->chars[0];

		/*
		 * Eligible only on the Markov path (r->start==0, the one that reads the
		 * table) with a charset whose VALUES form a contiguous range, identical
		 * across every position, and with contiguous active positions so each
		 * Markov prev char is a charset byte. r->chars[] is probability-sorted,
		 * so derive the range from its min/max, not chars[0].
		 */
		{
			int cmin = 255, cmax = 0, c;

			for (c = 0; c < r->count; c++) {
				if (r->chars[c] < cmin) cmin = r->chars[c];
				if (r->chars[c] > cmax) cmax = r->chars[c];
			}
			if (r->start != 0 || cmax - cmin + 1 != r->count)
				ltab_ok = 0;
			if (i == 0) {
				ltab_base = cmin;
				ltab_nc = r->count;
			} else {
				if (cmin != ltab_base || r->count != ltab_nc)
					ltab_ok = 0;
				if (gpu_tabs.keypos[i] != gpu_tabs.keypos[i - 1] + 1)
					ltab_ok = 0;
			}
		}
		memcpy(gpu_tabs.table  + (size_t)i * 256 * 256,
		       pos_markov_table[ri], 256 * 256);
		memcpy(gpu_tabs.startv + (size_t)i * 256,
		       pos_markov_start[ri], 256);
		memcpy(gpu_tabs.rowcnt + (size_t)i * 256,
		       pos_markov_row_counts[ri], 256);

		/*
		 * Saturate each prev-row's tail so the GPU kernel can index the table
		 * directly by the simplex rank without a separate row-count read+clamp.
		 * The simplex ranks iter[] in [0, count-1], but a given prev may have
		 * fewer valid Markov transitions (rowcnt <= count). The kernel used to
		 * read gpu_tabs.rowcnt and clamp (rank -> rowcnt-1, or chars0 when the
		 * row is empty); pre-filling slots [rowcnt, 256) with that exact fallback
		 * value makes table[rank] correct for every rank, dropping one divergent
		 * read per Markov position in the hot materialize. CPU consumers still
		 * clamp first, so they read the same value and are unaffected.
		 */
		for (int prev = 0; prev < 256; prev++) {
			size_t row = ((size_t)i * 256 + prev) * 256;
			int rc = pos_markov_row_counts[ri][prev];
			unsigned char fill = (rc > 0)
			    ? gpu_tabs.table[row + (rc - 1)]
			    : r->chars[0];
			int t;

			for (t = rc; t < 256; t++)
				gpu_tabs.table[row + t] = fill;
		}

		/* Pack this position's [256][256] prev,rank->char table into
		 * [256][64] uint32s (4 chars/word) for coalesced GPU reads. */
		for (int prev = 0; prev < 256; prev++) {
			for (int ti = 0; ti < 256; ti += 4) {
				size_t src = ((size_t)i * 256 * 256) +
				             ((size_t)prev * 256) + ti;
				uint32_t packed =
				    ((uint32_t)gpu_tabs.table[src + 0] <<  0) |
				    ((uint32_t)gpu_tabs.table[src + 1] <<  8) |
				    ((uint32_t)gpu_tabs.table[src + 2] << 16) |
				    ((uint32_t)gpu_tabs.table[src + 3] << 24);

				size_t dst = ((size_t)i * 256 * 64) +
				             ((size_t)prev * 64) + (ti / 4);
				gpu_tabs.uint_table[dst] = packed;
			}
		}
	}

	gpu_tabs.ltab_ok   = ltab_ok;
	gpu_tabs.ltab_base = ltab_base;
	gpu_tabs.ltab_nc   = ltab_nc;

	gpu_tabs.littmpl_len = maxL;
	gpu_tabs.littmpl = mem_calloc(maxL > 0 ? maxL : 1, 1);
	for (p = 0; p < maxL; p++)
		gpu_tabs.littmpl[p] = (unsigned char)template_key[p];

	mask_gpu_serial++;
}

/* Build the suffix-DP unrank table for every length-loop of the current run. */
static void mask_gpu_build_loops(mask_cpu_context *ctx)
{
	int max_loop = options.eff_maxlength - mask_cur_len;
	int loop, i, s, v;

	if (max_loop < 0)
		max_loop = 0;
	mask_gpu_max_loop = max_loop;

	for (loop = 0; loop <= max_loop; loop++) {
		mask_gpu_loop *gl = &gpu_loops[loop];
		int len = mask_cur_len + loop;
		int limit = 0, max_k = 0, ksize;
		uint64_t *suf, total = 0;

		MEM_FREE(gl->suf);
		memset(gl, 0, sizeof(*gl));
		gl->len = len;

		while (limit < ctx->active_count &&
		       ctx->ranges[ctx->active_idx[limit]].pos < len)
			limit++;
		gl->limit = limit;

		if (limit == 0) {
			/* No generated positions: a literal-only key (1 candidate) if the
			 * length is non-zero, else empty. */
			gl->ksize = 1;
			gl->suf = mem_calloc(1, sizeof(uint64_t));
			gl->suf[0] = 1;
			gl->total = (len > 0) ? 1 : 0;
			gl->valid = 1;
			continue;
		}

		for (i = 0; i < limit; i++)
			max_k += ctx->ranges[ctx->active_idx[i]].count - 1;
		ksize = max_k + 1;

		suf = mem_calloc((size_t)(limit + 1) * ksize, sizeof(uint64_t));
		suf[limit * ksize + 0] = 1;
		for (i = limit - 1; i >= 0; i--) {
			int C = ctx->ranges[ctx->active_idx[i]].count;
			for (s = 0; s <= max_k; s++) {
				uint64_t acc = 0;
				for (v = 0; v < C && v <= s; v++)
					acc += suf[(i + 1) * ksize + (s - v)];
				suf[i * ksize + s] = acc;
			}
		}
		for (s = 0; s <= max_k; s++)
			total += suf[0 * ksize + s];

		gl->max_k = max_k;
		gl->ksize = ksize;
		gl->suf = suf;
		gl->total = total;
		gl->valid = 1;
	}
}

/* Rebuild all GPU generation state for the current finalized mask. Called from
 * do_mask_crack() once the max-length template is materialized. */
static void mask_gpu_build(mask_cpu_context *ctx)
{
	mask_gpu_build_tables(ctx);
	mask_gpu_build_loops(ctx);
}

void mask_gpu_unrank_key(int loop, uint64_t g, char *out, int *out_len)
{
	const mask_gpu_loop *gl = &gpu_loops[loop];
	int len = gl->len, limit = gl->limit, ksize = gl->ksize;
	const uint64_t *suf = gl->suf;
	int iter[MAX_NUM_MASK_PLHDR];
	int i, target_k, remaining_k;
	uint64_t fw, rank;

	for (i = 0; i < len; i++)
		out[i] = (char)gpu_tabs.littmpl[i];
	out[len] = 0;
	if (out_len)
		*out_len = len;
	if (limit == 0)
		return;

	/* Skip whole K-layers, then unrank within the target layer (descending-
	 * lexicographic, matching simplex_next_state). */
	fw = g;
	target_k = 0;
	while (target_k <= gl->max_k && fw >= suf[target_k]) {
		fw -= suf[target_k];
		target_k++;
	}
	remaining_k = target_k;
	rank = fw;
	for (i = 0; i < limit; i++) {
		int C = gpu_tabs.count[i];
		int vmax = remaining_k < (C - 1) ? remaining_k : (C - 1);
		int vv;
		for (vv = vmax; vv >= 0; vv--) {
			uint64_t cnt = suf[(i + 1) * ksize + (remaining_k - vv)];
			if (rank < cnt)
				break;
			rank -= cnt;
		}
		iter[i] = vv;
		remaining_k -= vv;
	}

	/* Materialize left-to-right through the Markov tables - the exact mirror of
	 * the kernel. In head/tail mode (mask_gpu_head >= 0) positions i >= head draw
	 * from their own probability-sorted charset (startv[i]) independent of the
	 * previous char; positions 1..head-1 keep the conditional bigram model. */
	for (i = 0; i < limit; i++) {
		int kp = gpu_tabs.keypos[i];
		unsigned char cs = gpu_tabs.cstart[i];

		if (cs) {
			out[kp] = (char)(cs + iter[i]);
		} else if (i == 0) {
			out[kp] = (char)gpu_tabs.startv[iter[i]];
		} else if (mask_gpu_head >= 0 && i >= mask_gpu_head) {
			out[kp] = (char)gpu_tabs.startv[(size_t)i * 256 + iter[i]];
		} else {
			unsigned char prev = (unsigned char)out[kp - 1];
			int avail = gpu_tabs.rowcnt[(size_t)i * 256 + prev];
			int ti = iter[i];

			if (avail > 0) {
				if (ti >= avail)
					ti = avail - 1;
				out[kp] = (char)gpu_tabs.table[((size_t)i * 256 + prev) * 256 + ti];
			} else {
				out[kp] = (char)gpu_tabs.chars0[i];
			}
		}
	}
}

static double get_progress(void)
{
	double total;

	emms();

	if (!mask_tot_cand)
		return -1;

	total = crk_stacked_rule_count * mask_tot_cand;

	if (cand_length)
		total += cand_length;

	return 100.0 * status.cands / total;
}

void mask_save_state(FILE *file)
{
	int i, j;

	fprintf(file, "%"PRIu64"\n", rec_cand + 1);
	fprintf(file, "%d\n", rec_ctx.active_count);
	if (mask_increments_len) {
		fprintf(file, "%d\n", rec_len);
		fprintf(file, "%"PRIu64"\n", cand_length + 1);
	}
	for (i = 0; i < rec_ctx.active_count; i++) {
		int ri = rec_ctx.active_idx[i];
		for (j = 0; j <= options.eff_maxlength - options.eff_minlength; j++)
			fprintf(file, "%u\n", (unsigned)rec_ctx.ranges[ri].iter[j]);
	}
	/* Per-loop Markov K-layer and this node's remaining block budget. */
	for (j = 0; j <= options.eff_maxlength - options.eff_minlength; j++)
		fprintf(file, "%d\n", rec_ctx.current_k[j]);
	for (j = 0; j <= options.eff_maxlength - options.eff_minlength; j++)
		fprintf(file, "%"PRIu64"\n", rec_node_loop_cand[j]);
}

int mask_restore_state(FILE *file)
{
	int i, j, d;
	unsigned cu;
	uint64_t ull;
	int fail = !(options.flags & FLG_MASK_STACKED);

	if (fscanf(file, "%"PRIu64"\n", &ull) == 1)
		cand = ull;
	else
		return fail;

	if (fscanf(file, "%d\n", &d) == 1)
		restored_ctx.active_count = cpu_mask_ctx.active_count = d;
	else
		return fail;

	if (mask_increments_len) {
		if (fscanf(file, "%d\n", &d) == 1)
			restored_len = d;
		else
			return fail;
		if (fscanf(file, "%"PRIu64"\n", &ull) == 1)
			rec_cl = ull;
		else
			return fail;
	}

	for (i = 0; i < cpu_mask_ctx.active_count; i++) {
		int ri = cpu_mask_ctx.active_idx[i];
		for (j = 0; j <= options.eff_maxlength - options.eff_minlength; j++)
			if (fscanf(file, "%u\n", &cu) == 1)
				restored_ctx.ranges[ri].iter[j] =
				cpu_mask_ctx.ranges[ri].iter[j] = cu;
			else
				return fail;
	}
	/* Per-loop Markov K-layer and this node's remaining block budget. */
	for (j = 0; j <= options.eff_maxlength - options.eff_minlength; j++)
		if (fscanf(file, "%d\n", &d) == 1)
			restored_ctx.current_k[j] = cpu_mask_ctx.current_k[j] = d;
		else
			return fail;
	for (j = 0; j <= options.eff_maxlength - options.eff_minlength; j++)
		if (fscanf(file, "%"PRIu64"\n", &ull) == 1)
			node_loop_cand[j] = ull;
		else
			return fail;
	restored = 1;
	return 0;
}

void mask_fix_state(void)
{
	int i, j;

	if (parent_fix_state_pending) {
		crk_fix_state();
		parent_fix_state_pending = 0;
	}
	rec_cand = cand;
	rec_ctx.active_count = cpu_mask_ctx.active_count;
	rec_len = mask_cur_len;
	memcpy(rec_ctx.active_idx, cpu_mask_ctx.active_idx, sizeof(rec_ctx.active_idx));
	for (i = 0; i < cpu_mask_ctx.active_count; i++) {
		int ri = cpu_mask_ctx.active_idx[i];
		for (j = 0; j <= options.eff_maxlength - options.eff_minlength; j++)
			rec_ctx.ranges[ri].iter[j] = cpu_mask_ctx.ranges[ri].iter[j];
	}
	/* Snapshot per-loop Markov K-layer and node budget so resume is exact. */
	for (j = 0; j <= options.eff_maxlength - options.eff_minlength; j++) {
		rec_ctx.current_k[j] = cpu_mask_ctx.current_k[j];
		rec_node_loop_cand[j] = node_loop_cand[j];
	}
}

void remove_slash(char *mask)
{
	int i = 0;
	while (i < strlen(mask)) {
		if (mask[i] == '\\') {
		    int j = i;
		    while(j < strlen(mask)) {
			  mask[j] = mask[j + 1];
			  j++;
		    }
		}
		i++;
	}
}

/*
 * Stretch mask to mask_cur_len. If iterating over lengths, that means
 * current length - otherwise it's our minimum length (eg. 8 for WPAPSK).
 * Called by finalize_mask() but never for a hybrid mask.
 *
 *  1. If last mask position is a range, we repeat that.
 *     word?d --> word?d?d
 *
 *  2. Otherwise if there is any range, we repeat the *first* one.
 *     ?dword --> ?d?dword
 *     pass?dword --> pass?d?dword
 *
 *  3. Last resort, we just repeat the last character.
 *     pass --> passs
 */
char *stretch_mask(char *mask, mask_parsed_ctx *parsed_mask)
{
	char *stretched_mask;
	int i, j, k;
	int first_pl = -1, last_cl = -1;

#ifdef MASK_DEBUG
	fprintf(stderr, "%s(%s) to len %d\n", __FUNCTION__, mask, mask_cur_len);
#endif

	j = strlen(mask);

	// Find last closing range bracket
	while (parsed_mask->stack_cl_br[last_cl + 1] != -1)
		last_cl++;

	// Find first valid placeholder (ignoring ?w)
	for (i = 0; i < j; i++) {
		if (mask[i] == '\\') {
			i++;
			continue;
		}
		if (mask[i] == '?' &&
		    strchr(BUILT_IN_CHARSET, ARCH_INDEX(mask[i + 1])))
			break;
	}
	if (i < j)
		first_pl = i;

	stretched_mask =
		mem_alloc_tiny((options.eff_maxlength + 2) * j, MEM_ALIGN_NONE);

	strcpy(stretched_mask, mask);
	k = mask_len(mask);

	while (k && k < options.eff_maxlength) {
		i = strlen(mask) - 1;
		if (mask[i] == '\\' && i - 1 >= 0) {
			i--;
			if (!k) j--;
		}
		if (mask[i] == '\\') {
			if (!k) j++;
			strnzcpy(stretched_mask + j, mask + i, 3);
			j += 2;
		}
		else if (last_cl >= 0 && i == parsed_mask->stack_cl_br[last_cl]) {
			/* Repeat a trailing range word[abc] -> word[abc][abc] */
			i = parsed_mask->stack_op_br[last_cl];
			strcpy(stretched_mask + j, mask + i);
			j += strlen(mask + i);
		}
		else if (strchr(BUILT_IN_CHARSET, ARCH_INDEX(mask[i])) &&
		         i - 1 >= 0 && mask[i - 1] == '?') {
			/* Repeat a trailing placeholder word?d -> word?d?d */
			strnzcpy(stretched_mask + j, mask + i - 1, 3);
			j += 2;
		}
		else if (!format_cannot_reset && parsed_mask->stack_op_br[0] >= 0 &&
		         parsed_mask->stack_op_br[0] < first_pl) {
			/* Repeat a leading range [abc]word -> [abc][abc]word */
			/* Or repeat first range wor[range]d -> wor[range][range]d */
			int beg = parsed_mask->stack_op_br[0];
			int end = parsed_mask->stack_cl_br[0] + 1;

			memmove(stretched_mask + end, stretched_mask + beg, j - beg + 1);
			j += end - beg;
		}
		else if (!format_cannot_reset && first_pl >= 0) {
			/* Repeat a leading placeholder ?dword -> ?d?dword */
			/* Or repeat first placeholder w?dord -> w?d?dord */
			memmove(stretched_mask + first_pl + 2,
			        stretched_mask + first_pl, j + 3);
			j += 2;
		}
		else if (j) {
			/*
			 * Last resort, just repeat last character. This is
			 * likely useless but OTOH it will finish very fast.
			 */
			stretched_mask[j] = stretched_mask[j - 1];
			j++;
		}
		k++;
	}
	stretched_mask[j] = '\0';

#ifdef MASK_DEBUG
	fprintf(stderr, "%s(): %s --> %s\n", __FUNCTION__, mask, stretched_mask);
#endif
	return stretched_mask;
}

static void finalize_mask(int len);

/*
 * Notes about escapes, lists and ranges:
 *
 * Parsing chain:
 * mask -> utf8_to_cp() -> expand_plhdr() -> parse_hex()
 *                      -> parse_braces() -> parse_qtn()
 *
 * "\x41" means literal "A". Hex escaped characters must be passed as-is until
 * parse_hex(). All other escapes should be passed as-is past parse_qtn().
 * Note that de-hex comes after UTF-8 conversion so any 8-bit hex escaped
 * characters will be parsed as the *internal* encoding.
 *
 * Hex characters *can* compose ranges, e.g. "\x80-\xff", but can not end up as
 * placeholders. Eg. "\x3fd" ("?d" after de-hex) must be parsed literally as
 * "?d" and not a digits range.
 *
 * Anything else escaped by "\" must be parsed as literal character,
 * including but not limited to:
 *    "\\" means literal "\" with no further meaning
 *    "\?" means literal "?" and must never be parsed as placeholder (but -"-)
 *    "\-" means literal "-" and must never be parsed as range
 *    "\[" means literal "[" and must never start a list range
 *    "\]" means literal "]" and must never end a list range
 *
 */
void mask_init(struct db_main *db, char *unprocessed_mask)
{
	int conv_err[MAX_NUM_CUST_PLHDR] = { 0 };
	int i;

	mask_db = db;
	mask_fmt = db->format;
	mask_bench_index = 0;

	/* CPU validation of the GPU K-ordered generator: force gen mode on so
	 * do_mask_crack() drives mask_gpu_unrank_key() on the host (the format need
	 * not support GPU generation). Lets us prove the unrank against a CPU format
	 * before the kernel path is wired up. */
	if (getenv("MASK_GPU_CPU")) {
		mask_gpu_gen = 1;
		mask_gpu_cpu_validate = 1;
	}

	/* These formats are too weird for magnum to get working */
#if defined(HAVE_OPENCL) || defined(HAVE_ZTEX)
	/* Disable internal mask */
	if (options.req_int_cand_target == 0) {
		if (mask_int_cand_target)
			log_event("- Format's internal mask generation disabled by command-line option");
		mask_fmt->params.flags &= ~FMT_MASK;
		mask_int_cand_target = 0;
	} else
#endif
	/* These formats are too wierd for magnum to get working */
	if (!strcasecmp(mask_fmt->params.label, "descrypt-opencl") ||
	    !strcasecmp(mask_fmt->params.label, "lm-opencl"))
		format_cannot_reset = 1;

	/* Using "--mask" alone will use default mask and iterate over length */
	if (!(options.flags & FLG_MASK_STACKED) && (options.flags & FLG_CRACKING_CHK) && !unprocessed_mask &&
	    options.req_minlength < 0 && !options.req_maxlength)
		mask_increments_len = 1;

	/* Specified length range given */
	if ((options.req_minlength >= 0 || options.req_maxlength) &&
	    (options.eff_minlength != options.eff_maxlength) &&
	    !(options.flags & FLG_MASK_STACKED))
		mask_increments_len = 1;

	max_keylen = options.rule_stack ? 125 : options.eff_maxlength;

	if ((options.flags & FLG_MASK_STACKED) && max_keylen < 2) {
		if (john_main_process)
			fprintf(stderr,
			        "Error: Too short max. length for hybrid mask\n");
		error();
	}

#ifdef MASK_DEBUG
	fprintf(stderr, "%s(%s) maxlen %d\n", __FUNCTION__, unprocessed_mask,
	        max_keylen);
#endif

	/* Load defaults from john.conf */
	if (!unprocessed_mask) {
		if (options.flags & FLG_TEST_CHK) {
			static char test_mask[PLAINTEXT_BUFFER_SIZE + 8];
			int bl = mask_fmt->params.benchmark_length & 0xff;

			strcpy(test_mask, "?a?a?l?u?d?d?s?s" "xxxxxxxxxxxxxxxxxxxxx"
			                  "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
			                  "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
			                  "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"); // l = 125

			if (bl <= 8)
				test_mask[2 * bl] = 0;
			else if (bl < (strlen(test_mask) - 8))
				test_mask[bl + 8] = 0;
			else
				fprintf(stderr,
				        "Warning: Format wanted length %d benchmark", bl);

			unprocessed_mask = test_mask;
		}
		else if (options.flags & FLG_MASK_STACKED)
			unprocessed_mask = (char*)cfg_get_param("Mask", NULL, "DefaultHybridMask");
		else
			unprocessed_mask = (char*)cfg_get_param("Mask", NULL, "DefaultMask");

		if (!unprocessed_mask)
			unprocessed_mask = "";

		if (2 * options.eff_maxlength < strlen(unprocessed_mask))
			unprocessed_mask[2 * options.eff_maxlength] = 0;

		using_default_mask = 1;
	}

	if ((options.flags & FLG_TEST_CHK) && options.verbosity >= VERB_MAX)
		fprintf(stderr, "\nTest mask: %s\n", unprocessed_mask);

	if (!(options.flags & FLG_MASK_STACKED) && john_main_process) {
		log_event("Proceeding with mask mode");

		if (rec_restored) {
			fprintf(stderr, "Proceeding with mask mode:%s", unprocessed_mask);
			if (options.rule_stack)
				fprintf(stderr, ", rules-stack:%s", options.rule_stack);
			if (options.req_minlength >= 0 || options.req_maxlength)
				fprintf(stderr, ", lengths: %d-%d",
				        options.eff_minlength, options.eff_maxlength);
			fprintf(stderr, "\n");
		}
	}

	if (using_default_mask && !(options.flags & FLG_TEST_CHK) &&
	    john_main_process)
		fprintf(stderr, "Using default mask: %s\n", unprocessed_mask);

	/* Load defaults for custom placeholders ?1..?9 from john.conf */
	for (i = 0; i < MAX_NUM_CUST_PLHDR; i++) {
		char pl[2] = { '1' + i, 0 };

		if (!options.custom_mask[i] &&
		    !(options.custom_mask[i] = (char*)cfg_get_param("Mask", NULL, pl)))
			options.custom_mask[i] = "";
	}

	mask = unprocessed_mask;
	template_key = mem_alloc(0x400);
	old_extern_key_len = -1;

	/* Handle command-line (or john.conf) masks given in UTF-8 */
	if (options.input_enc == UTF_8 && options.internal_cp != UTF_8) {
		if (valid_utf8((UTF8*)mask) > 1) {
			int u8_length = strlen8((UTF8*)mask);

			utf8_to_cp_r(mask, mask, strlen(mask));
			if (strlen(mask) != u8_length) {
				if (john_main_process)
					fprintf(stderr, "Error: The selected internal codepage can't hold all characters of mask\n");
				error();
			}
		}

		for (i = 0; i < MAX_NUM_CUST_PLHDR; i++) {
			if (valid_utf8((UTF8*)options.custom_mask[i]) > 1) {
				int u8_length = strlen8((UTF8*)options.custom_mask[i]);
				int size = strlen(options.custom_mask[i]);
				char *tmp_buf;

				tmp_buf = mem_alloc(size); /* result is guaranteed to be at least one byte smaller */
				utf8_to_cp_r(options.custom_mask[i], tmp_buf, size);
				if (strlen(tmp_buf) != u8_length)
					conv_err[i] = 1; /* Defer error until expand_cplhdr() - if placeholder is used */
				else
					strnzcpy(options.custom_mask[i], tmp_buf, size);
				MEM_FREE(tmp_buf);
			}
		}
	}

	/* Expand static placeholders within custom ones */
	for (i = 0; i < MAX_NUM_CUST_PLHDR; i++)
		if (*options.custom_mask[i])
			options.custom_mask[i] =
				str_alloc_copy(expand_plhdr(options.custom_mask[i],
				    mask_fmt->params.flags & FMT_CASE));

	/* Finally expand custom placeholders ?1 .. ?9 */
	mask = expand_cplhdr(mask, conv_err);

	/*
	 * UTF-8 is not supported in mask mode unless -internal-codepage is used.
	 */
	if (options.internal_cp == UTF_8 && valid_utf8((UTF8*)mask) > 1) {
		if (john_main_process)
			fprintf(stderr, "Error: Mask contains UTF-8 characters; You need to set a legacy codepage\n"
			        "       with --internal-codepage (UTF-8 is not a codepage).\n");
		error();
	}

	/* De-hexify mask and custom placeholders */
	parse_hex(mask);
	for (i = 0; i < MAX_NUM_CUST_PLHDR; i++)
		if (*options.custom_mask[i])
			parse_hex(options.custom_mask[i]);

	/* Drop braces around a single-character [z] -> z */
	options.eff_mask = mask = drop1range(mask);

	if (mask_increments_len && !using_default_mask) {
		int orig_len = mask_len(mask);

		if (options.req_minlength < 0 && orig_len > options.eff_minlength)
			options.eff_minlength = orig_len;
	}

	/*
	 * An explicit, fixed-length mask (non-default, non-stacked) with no
	 * requested length range is exactly mask_len(mask) characters long. Pin
	 * the effective lengths (and max_keylen) to it, otherwise eff_minlength
	 * stays 0 and the mask is stretched up to the format's maximum length,
	 * emitting wrong-length candidates (or none at all on GPU formats).
	 */
	if (!mask_increments_len && !using_default_mask &&
	    !(options.flags & FLG_MASK_STACKED) &&
	    options.req_minlength < 0 && !options.req_maxlength) {
		int fixed_len = mask_len(mask);

		options.eff_minlength = options.eff_maxlength = fixed_len;
		if (!options.rule_stack)
			max_keylen = fixed_len;
	}

	if (format_cannot_reset) {
		if (options.flags & FLG_MASK_STACKED)
			mask_cur_len = 0;
		else
			mask_cur_len = mask_increments_len ? options.eff_maxlength : options.eff_minlength;
		finalize_mask(max_keylen);
	} else if (!((mask_fmt->params.flags & FMT_MASK) && mask_increments_len)) {
		mask_cur_len = options.eff_minlength;
		finalize_mask(max_keylen);
	} else {
		/*
		 * Increment-length on a FMT_MASK (GPU) format. Finalize at the max
		 * length now so the GPU kernel - built in the format's reset(), before
		 * the first do_mask_crack() - is compiled for the correct internal-mask
		 * configuration (mask_skip_ranges / static_gpu_locations). Otherwise
		 * those are still unset at kernel-build time and get_key() reconstructs
		 * cracked plaintext with a stale -1 location (out-of-bounds write).
		 * do_mask_crack() re-finalizes per parent key, so this is idempotent.
		 */
		mask_cur_len = options.eff_maxlength;
		finalize_mask(max_keylen);
	}

	if (format_cannot_reset && mask_increments_len && mask_skip_ranges[0] != -1) {
		int inc_min =
			mask_int_cand.int_cpu_mask_ctx->ranges[mask_max_skip_loc].pos + 1;
		if (inc_min > options.eff_maxlength) {
			if (john_main_process)
				fprintf(stderr, "Error: %s cannot use internal mask under these premises,\n"
				        "try using --mask-internal-target=0 option.\n", mask_fmt->params.label);
			error();
		}
		if (options.eff_minlength < inc_min) {
			mask_iter_warn = inc_min;
			if (john_main_process)
				fprintf(stderr, "Note: %s format can't currently increment length from %d, using %d instead\n",
			        mask_fmt->params.label, options.eff_minlength, inc_min);
			options.eff_minlength = inc_min;
		}
	}
}

/*
 * Finalizes the mask for current length (stretching it to mask_cur_len if
 * applicable).  Sets up CPU-side mask and calls mask_ext for setting up
 * GPU-side mask.  Called by do_mask_crack() if iterating lengths, otherwise
 * from mask_init() above.
 */
static void finalize_mask(int len)
{
	int i, max_static_range;

#ifdef MASK_DEBUG
	fprintf(stderr, "\n%s(%d) mask %s\n", __FUNCTION__, len, mask);
#endif
	/* Reset things, in case we're iterating over lengths */

//	memset(&cpu_mask_ctx, 0, sizeof(cpu_mask_ctx));
	memset(&parsed_mask, 0, sizeof(parsed_mask));
	MEM_FREE(mask_skip_ranges);
	MEM_FREE(mask_int_cand.int_cand);
	MEM_FREE(template_key_offsets);

	/* Parse ranges */
	parse_braces(mask, &parsed_mask);

	if (!(options.flags & FLG_MASK_STACKED) &&
	    (options.eff_minlength > mask_len(mask) || mask_len(mask) < len)) {
		mask = stretch_mask(mask, &parsed_mask);
		parse_braces(mask, &parsed_mask);
	}
	parse_qtn(mask, &parsed_mask);

	i = 0; mask_add_len = 0; mask_num_qw = 0; max_static_range = 0;
	while (i < strlen(mask)) {
		int t;

		if ((t = search_stack(&parsed_mask, i))) {
			mask_add_len++;
			i = t + 1;
			if (!mask_num_qw)
				max_static_range++;
		} else if (mask[i] == '\\') {
			i += 2;
			mask_add_len++;
		} else if (i + 1 < strlen(mask) && mask[i] == '?' &&
		    (mask[i + 1] == 'w' || mask[i + 1] == 'W')) {
			mask_num_qw++;
			i += 2;
			if ((options.flags & FLG_MASK_STACKED) &&
			    mask_add_len >= (unsigned int)len &&
			    mask_num_qw == 1) {
				if (john_main_process)
				fprintf(stderr, "Error: Hybrid mask must contain ?w/?W after truncation for max. length\n");
				error();
			}
		} else {
			i++;
			mask_add_len++;
		}
	}
	if (options.flags & FLG_MASK_STACKED) {
		mask_has_8bit = 1; /* Parent mode's word might have 8-bit */
		if (mask_add_len > len - 1)
			mask_add_len = len - 1;

		if (mask_num_qw == 0) {
			if (john_main_process)
				fprintf(stderr, "Error: Hybrid mask must contain ?w or ?W\n");
			error();
		}
	} else {
		if (mask_num_qw && john_main_process)
			fprintf(stderr, "Warning: ?w has no special meaning unless running hybrid mask\n");
		if (mask_add_len > len)
			mask_add_len = len;
	}

	if ((mask_fmt->params.flags & FMT_MASK) && options.rule_stack) {
		mask_int_cand_target = 0;
		mask_fmt->params.flags &= ~FMT_MASK;
		format_cannot_reset = 0;
		if (john_main_process) {
			fprintf(stderr, "Note: Disabling internal mask due to stacked rules\n");
			log_event("- Disabling internal mask due to stacked rules");
		}
	}
#if defined(HAVE_OPENCL) || defined(HAVE_ZTEX)
	else if ((mask_fmt->params.flags & FMT_MASK) && options.req_int_cand_target > 0) {
		log_event("- Overriding format's target internal mask factor of %d with user requested %d",
		          mask_int_cand_target, options.req_int_cand_target);
		mask_int_cand_target = options.req_int_cand_target;
	}
#endif

#ifdef MASK_DEBUG
	fprintf(stderr, "%s() qw %d minlen %d maxlen %d max_key_len %d mask_add_len %d mask len %d\n", __FUNCTION__, mask_num_qw, options.eff_minlength, max_keylen, len, mask_add_len, mask_len(mask));
#endif
	/* We decrease these here instead of changing parent modes. */
	if (options.flags & FLG_MASK_STACKED) {
		options.eff_minlength = MAX(0, options.eff_minlength - mask_add_len);
		options.eff_maxlength = MAX(0, options.eff_maxlength - mask_add_len);
		if (mask_num_qw) {
			options.eff_minlength /= mask_num_qw;
			options.eff_maxlength /= mask_num_qw;
		}
#ifdef MASK_DEBUG
		fprintf(stderr, "%s(): effective minlen %d maxlen %d x %d + mask_add_len %d == %d\n",
		        __FUNCTION__,
		        options.eff_minlength, options.eff_maxlength, mask_num_qw, mask_add_len, options.eff_maxlength * mask_num_qw + mask_add_len);
#endif
		if (options.eff_maxlength == 0) {
			if (john_main_process)
				fprintf(stderr, "Error: Hybrid mask would truncate input to length 0!\n");
			error();
		}
	}

	template_key_offsets = mem_alloc((mask_num_qw + 1) * sizeof(int));

	for (i = 0; i < mask_num_qw + 1; i++)
		template_key_offsets[i] = -1;

#ifdef MASK_DEBUG
	fprintf(stderr, "%s(): masks expanded (this is 'mask' when passed to "
	        "init_cpu_mask()):\n%s\n", __FUNCTION__, mask);
#endif
	init_cpu_mask(mask, &parsed_mask, &cpu_mask_ctx, max_keylen);

	/* GPU K-ordered generation generates *every* position on the device, so the
	 * internal-mask split must be off: keep all placeholders host-active and
	 * num_int_cand == 1. */
	if (mask_gpu_gen)
		mask_int_cand_target = 0;

	/* On a GPU (FMT_MASK) format iterating over length, the internal-mask
	 * placeholder is written at a fixed key position by the kernel; for any
	 * length shorter than that position the write corrupts the MD5 padding and
	 * the candidates are lost. Cap the placeholder to a position that fits the
	 * shortest length in the run (length 0 is the empty-key special case, so the
	 * shortest enumerated length is at least 1). Other cases impose no cap. */
	if ((mask_fmt->params.flags & FMT_MASK) && mask_increments_len &&
	    !(options.flags & FLG_MASK_STACKED)) {
		int min_len = options.eff_minlength > 0 ? options.eff_minlength : 1;
		mask_int_max_pos = min_len - 1;
	} else
		mask_int_max_pos = -1;

	mask_ext_calc_combination(&cpu_mask_ctx, max_static_range);

#ifdef MASK_DEBUG
	fprintf(stderr, "%s() MASK_FMT_INT_PLHDRs: max static range %d: ",
	        __FUNCTION__, max_static_range);
	for (i = 0; i < MASK_FMT_INT_PLHDR && mask_skip_ranges; i++)
		fprintf(stderr, "%d ", mask_skip_ranges[i]);
	fprintf(stderr, "\n");
#endif
	int_mask_sum = 0;
	if (mask_skip_ranges) {
		for (i = 0; i < MASK_FMT_INT_PLHDR && mask_skip_ranges[i] >= 0; i++)
			int_mask_sum |= cpu_mask_ctx.ranges[mask_skip_ranges[i]].count << (8 * i);
	}

	skip_position(&cpu_mask_ctx, mask_skip_ranges);

	/* If running hybrid (stacked), we let the parent mode distribute */
	if (!restored) {
		if (options.node_count && !(options.flags & FLG_MASK_STACKED)) {
			cand = divide_work(&cpu_mask_ctx);
		} else {
			cand = 1;
			for (i = 0; i < cpu_mask_ctx.active_count; i++) {
				int ri = cpu_mask_ctx.active_idx[i];
				if (!(options.flags & FLG_MASK_STACKED) &&
					cpu_mask_ctx.ranges[ri].pos >= max_keylen && !format_cannot_reset)
					continue;
				/* for incremental lengths, skip placeholders that are beyond the current length */
				if (cpu_mask_ctx.ranges[ri].pos < mask_cur_len)
					cand *= cpu_mask_ctx.ranges[ri].count;
			}
		}
	}
	mask_tot_cand = cand * mask_int_cand.num_int_cand;

	if ((john_main_process || !cfg_get_bool(SECTION_OPTIONS, SUBSECTION_MPI, "MPIAllGPUsSame", 0)) &&
		mask_int_cand.num_int_cand > 1)
		log_event("- Requested internal mask factor: %d, actual now %d",
		          mask_int_cand_target, mask_int_cand.num_int_cand);
}

void mask_crk_init(struct db_main *db)
{
#ifdef MASK_DEBUG
	fprintf(stderr, "%s()\n", __FUNCTION__);
#endif
	if (!(options.flags & FLG_MASK_STACKED)) {
		status_init(get_progress, 0);

		rec_restore_mode(mask_restore_state);
		rec_init(db, mask_save_state);

		crk_init(db, mask_fix_state, NULL);
	}
}

void mask_done()
{
#ifdef MASK_DEBUG
	fprintf(stderr, "%s()\n", __FUNCTION__);
#endif

	if (!(options.flags & FLG_MASK_STACKED)) {
		/* For reporting DONE regardless of rounding errors */
		if (!event_abort) {
			mask_tot_cand = status.cands;
			cand_length = 0;
		}
		if (!(options.flags & FLG_TEST_CHK)) {
			crk_done();
			rec_done(event_abort);
		}
	}
}

// Mask unload objects event. To be call after mask_done()
void mask_destroy()
{
#ifdef MASK_DEBUG
	fprintf(stderr, "%s()\n", __FUNCTION__);
#endif

	if (using_default_mask) {
		options.mask = NULL;
		using_default_mask = 0;
	}

	MEM_FREE(template_key);
	MEM_FREE(template_key_offsets);
	MEM_FREE(mask_skip_ranges);
	MEM_FREE(mask_int_cand.int_cand);
	mask_int_cand.num_int_cand = 1;
	mask_int_cand_target = 0;
}

/* Emit one block [base, base+count) of length-loop 'loop'. On a real GPU gen
 * format this hands the format the cursor and runs a GPU-generated crypt batch;
 * in CPU-validation mode it materializes each candidate on the host. */
static int mask_gpu_emit_block(uint64_t base, int count)
{
	if (mask_gpu_cpu_validate) {
		char key[PLAINTEXT_BUFFER_SIZE];
		int i, kl, loop;
		uint64_t g;

		for (i = 0; i < count; i++) {
			mask_gpu_virt_to_loop(base + i, &loop, &g);
			mask_gpu_unrank_key(loop, g, key, &kl);
			if (crk_process_key(key))
				return 1;
		}
		return 0;
	}

	mask_gpu_cur_base = base;
	return crk_process_gen_block(count);
}

/* GPU K-ordered generation driver: concatenate all active length-loops into one
 * virtual index space (mask_gpu_build_plan) and hand the format (or the CPU
 * validator) contiguous virtual ranges. A single GPU launch then spans whatever
 * lengths its block covers, so all lengths are crunched together instead of one
 * length fully draining before the next starts. */
static int mask_gpu_do_crack(const char *extern_key, int extern_key_len)
{
	int min = options.eff_minlength, max = options.eff_maxlength;
	int block_max;
	uint64_t base;
	const mask_gpu_plan *plan;

	/* mask_gpu_cpu_validate resolved eagerly in mask_init(). */

	/* Length-0 (empty) candidate is enumerated outside the simplex. */
	if (min == 0 && john_main_process)
		if (crk_process_key(fmt_null_key))
			return 1;

	mask_cur_len = (min > 0) ? min : 1;
	finalize_mask(max);
	generate_template_key(mask, extern_key, extern_key_len, &parsed_mask,
	                      &cpu_mask_ctx, max);
	mask_gpu_build(&cpu_mask_ctx);
	mask_gpu_build_plan();
	plan = mask_gpu_get_plan();

	block_max = mask_fmt->params.max_keys_per_crypt;
	if (block_max < 1)
		block_max = 1;
	if (mask_gpu_cpu_validate && block_max > 4096)
		block_max = 4096;

	if (getenv("MASK_GPU_PLAN")) {
		uint64_t round0 = 0;
		int l;
		for (l = 0; l < plan->nseg && plan->seg[l].vbase < plan->seg[0].vbase + 1; l++)
			;
		/* round size = vbase of the seg where loop 0 repeats */
		for (l = 1; l < plan->nseg; l++)
			if (plan->seg[l].loop == plan->seg[0].loop) { round0 = plan->seg[l].vbase; break; }
		fprintf(stderr, "[EMIT] block_max=%d round0_size=%"PRIu64" -> rounds/launch=%.1f "
			"(blocks to exhaust=%.1f)\n", block_max, round0,
			round0 ? (double)block_max / round0 : 0,
			(double)plan->total / block_max);
	}

	if (!restored) {
		mask_tot_cand = plan->total;
		cand = plan->total;
	}

	for (base = 0; base < plan->total; ) {
		uint64_t left = plan->total - base;
		int cnt = left > (uint64_t)block_max ? block_max : (int)left;

		if (mask_gpu_emit_block(base, cnt))
			return 1;
		base += cnt;
	}

	return event_abort;
}

int do_mask_crack(const char *extern_key)
{
	int extern_key_len = extern_key ? strlen(extern_key = mask_utf8_to_cp(extern_key)) : 0;
	int i;

	if (mask_gpu_gen)
		return mask_gpu_do_crack(extern_key, extern_key_len);

#ifdef MASK_DEBUG
	fprintf(stderr, "%s(\"%s\") (format %s internal mask)\n", __FUNCTION__, extern_key, mask_fmt->params.flags & FMT_MASK ? "has" : "doesn't have");
#endif

	mask_parent_keys++;

	if (mask_increments_len) {
		/* all lengths processed together, round‑robin */
		mask_cur_len = options.eff_minlength;
		if (mask_cur_len == 0) {
			if (john_main_process) {
				if (!format_cannot_reset && (mask_fmt->params.flags & FMT_MASK)) {
					finalize_mask(0);
					generate_template_key(mask, NULL, 0, &parsed_mask, &cpu_mask_ctx, 0);
				}
				if (crk_process_key(fmt_null_key))
					return 1;
			}
			mask_cur_len++;
		}

		/* finalize for the maximum length */
		finalize_mask(options.eff_maxlength);
		generate_template_key(mask, extern_key, extern_key_len, &parsed_mask, &cpu_mask_ctx, options.eff_maxlength);

		/* FIX: compute correct GLOBAL total candidates across all lengths cleanly */
		uint64_t global_tot_cand = 0;
		for (int loop = 0; loop <= options.eff_maxlength - mask_cur_len; loop++) {
			uint64_t len_cand = 1;
			for (int j = 0; j < cpu_mask_ctx.active_count; j++) {
				int ri = cpu_mask_ctx.active_idx[j];
				if (cpu_mask_ctx.ranges[ri].pos < mask_cur_len + loop)
					len_cand *= cpu_mask_ctx.ranges[ri].count;
			}
			global_tot_cand += len_cand * mask_int_cand.num_int_cand;
		}

		if (!restored && options.node_count && !(options.flags & FLG_MASK_STACKED)) {
			/* finalize_mask() -> divide_work() already partitioned the keyspace
			 * per length-loop, set this node's cand, node_loop_cand[] budgets,
			 * fast-forwarded the iterator state, and set mask_tot_cand to this
			 * node's share. Do not recompute it here. */
		} else {
			mask_tot_cand = global_tot_cand;
			if (!restored)
				cand = mask_tot_cand;
		}

		if (options.flags & FLG_TEST_CHK) {
			if (bench_generate_keys(&cpu_mask_ctx, &cand))
				return 1;
		} else {
			if (generate_keys(&cpu_mask_ctx, &cand))
				return 1;
		}
	} else {
		if (old_extern_key_len != extern_key_len) {
			save_restore(&cpu_mask_ctx, 0, RESTORE);
			generate_template_key(mask, extern_key, extern_key_len, &parsed_mask, &cpu_mask_ctx, max_keylen);
			old_extern_key_len = extern_key_len;
		}

		i = 0;
		while(template_key_offsets[i] != -1) {
			int offset = template_key_offsets[i] & 0xffff;
			unsigned char toggle =  (template_key_offsets[i++] >> 16) == 'W';
			int cpy_len = MIN(max_keylen - offset, extern_key_len);

			if (!toggle)
				memcpy(template_key + offset, extern_key, cpy_len);
			else {
				int z;
				for (z = 0; z < cpy_len; ++z) {
					if (enc_islower(extern_key[z]))
						template_key[offset + z] =
							enc_toupper(extern_key[z]);
					else
						template_key[offset + z] =
							enc_tolower(extern_key[z]);
				}
			}
		}
		if (options.flags & FLG_TEST_CHK) {
			if (bench_generate_keys(&cpu_mask_ctx, &cand))
				return 1;
		} else {
			if (generate_keys(&cpu_mask_ctx, &cand))
				return 1;
		}
	}

	if (options.flags & FLG_MASK_STACKED) {
		if (options.flags & FLG_WORDLIST_CHK)
			wordlist_hybrid_fix_state();
		else if (options.flags & FLG_MKV_CHK)
			mkv_hybrid_fix_state();
		else if (options.flags & FLG_INC_CHK)
			inc_hybrid_fix_state();
#if HAVE_LIBGMP || HAVE_INT128 || HAVE___INT128 || HAVE___INT128_T
		else if (options.flags & FLG_PRINCE_CHK)
			pp_hybrid_fix_state();
#endif
		else if (options.flags & FLG_EXTERNAL_CHK)
			ext_hybrid_fix_state();
		parent_fix_state_pending = 1;
	}

	return event_abort;
}

int mask_calc_len(const char *mask_in)
{
	char *mask = str_alloc_copy(mask_in);

	return mask_len(parse_hex(mask));
}
