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

/*
 * Mask mode cracker.
 */

#ifndef _JOHN_MASK_H
#define _JOHN_MASK_H

#include "loader.h"

// See also opencl_mask.h.
#define MASK_FMT_INT_PLHDR 4

// Maximum number of placeholders in a mask.
#define MAX_NUM_MASK_PLHDR 125

// Maximum number of length increments (must be >= options.eff_maxlength-options.eff_minlength)
#define MASK_MAX_INC_LEN   MAX_NUM_MASK_PLHDR

//#define MASK_DEBUG

typedef struct {
	/* store locations of op braces in mask */
	int stack_op_br[MAX_NUM_MASK_PLHDR + 1];
	/* store locations of cl braces in mask */
	int stack_cl_br[MAX_NUM_MASK_PLHDR + 1];
	/* store locations of valid ? in mask */
	int stack_qtn[MAX_NUM_MASK_PLHDR + 1];
} mask_parsed_ctx;

 /* Range of characters for a placeholder in the mask */
 /* Rearranging the structure could affect performance */
typedef struct {
	/* Characters in the range */
	unsigned char chars[0xFF];
	/* current position in chars[] while iterating for each length increment */
	unsigned char iter[MASK_MAX_INC_LEN];
	/* Number of characters in the range */
	unsigned char count;
	/*
	 * Set to zero when the characters in the range are not consecutive,
	 * otherwise start is set to the minimum value in range. Minimum
	 * value cannot be a null character which has a value zero.
	 */
	unsigned char start;
	/* Base position of the characters in key */
	int pos;
	/* offset when a key is inserted from other mode */
	int offset;
	/* REMOVED: unsigned char next;   (linked list is gone) */
} mask_range;

/* Processed mask structure for password generation on CPU */
typedef struct {
	/* Set of mask placeholders for generating password */
	mask_range ranges[MAX_NUM_MASK_PLHDR + 1];
	/* Positions in mask active for iteration on CPU (1 = active) */
	int active_positions[MAX_NUM_MASK_PLHDR + 1];
	/* Compact list of active placeholder indices (in order) */
	int active_idx[MAX_NUM_MASK_PLHDR + 1];
	/* Number of active placeholders */
	int active_count;
	/* Number of placeholders active for iteration on CPU (same as active_count) */
	int cpu_count;
	int current_k[MASK_MAX_INC_LEN];
} mask_cpu_context;

/*
 * Initialize mask mode cracker.
 */
extern void mask_init(struct db_main *db, char *unprocessed_mask);

/*
 * Initialize cracker database.
 */
extern void mask_crk_init(struct db_main *db);

/*
 * Runs the mask mode cracker.
 */
extern int do_mask_crack(const char *key);

extern void mask_done(void);
extern void mask_destroy(void);

/*
 * These are exported for stacked modes (eg. hybrid mask)
 */
extern void mask_fix_state(void);
extern void mask_save_state(FILE *file);
extern int mask_restore_state(FILE *file);

/* Evaluate mask_add_len from a given mask string without calling mask_init */
extern int mask_calc_len(const char *mask);

/*
 * Total number of candidates (per node) to begin with. Remains unchanged
 * throughout one call to do_mask_crack but may vary with hybrid parent key
 * length.  The number includes the part that is processed on GPU, and is
 * used as a multiplier in native mask mode's and parent modes' get_progress().
 */
extern uint64_t mask_tot_cand;

/* Hybrid mask's contribution to key length. Eg. for bc?l?d?w this will be 4. */
extern int mask_add_len;

/* Number of ?w in hybrid mask */
extern int mask_num_qw;

/* Number of times parent mode called hybrid mask. */
extern uint64_t mask_parent_keys;

/* Current length when pure mask mode iterates over lengths */
extern int mask_cur_len;

/* Incremental mask iteration started at this length (contrary to options) */
extern int mask_iter_warn;

/* Mask mode is incrementing mask length */
extern int mask_increments_len;

/* ----------------------------------------------------------------------------
 *  GPU K-ordered password generation
 * ----------------------------------------------------------------------------
 * When a FMT_MASK OpenCL format sets mask_gpu_gen, mask mode stops streaming
 * host-materialized keys and instead hands the format, per length-loop, a
 * contiguous range of global candidate indices [base, base+count). Each GPU
 * work-item unranks its index g into the (K, iter[]) simplex point and
 * materializes the password through the Markov tables - so candidates come out
 * in exact rank-sum (K) order within each length. The host uploads the tables
 * below once per mask config and the per-loop suffix-DP once per loop.
 */

/* Set to 1 by the format (in its reset/init) to request GPU generation. */
extern int mask_gpu_gen;

/* Set to 1 (from the MASK_GPU_CPU env) to validate the GPU K-ordered generator:
 * mask mode materializes each candidate on the host via mask_gpu_unrank_key() and
 * streams it through the format's NORMAL crypt path. A format that supports GPU
 * generation must therefore disable its gen kernel/crypt/get_key when this is set
 * (treat it as non-gen) even though mask_gpu_gen stays 1. */
extern int mask_gpu_cpu_validate;

/* Cursor the format's gen-crypt reads to know which block to generate:
 * candidates [mask_gpu_cur_base, +count) of length-loop mask_gpu_cur_loop. */
extern int mask_gpu_cur_loop;
extern uint64_t mask_gpu_cur_base;

/* Per-mask Markov materialization tables, flattened for GPU upload. Indexed by
 * generated-position i in [0, npos) in left-to-right key order. */
typedef struct {
	int npos;                              /* number of generated positions   */
	int keypos[MAX_NUM_MASK_PLHDR];        /* key byte offset of position i    */
	int count[MAX_NUM_MASK_PLHDR];         /* #chars at position i             */
	unsigned char cstart[MAX_NUM_MASK_PLHDR]; /* contiguous start char, 0=Markov */
	unsigned char chars0[MAX_NUM_MASK_PLHDR]; /* fallback char (range's chars[0]) */
	unsigned char *table;                  /* [npos][256][256] prev,rank->char */
	uint32_t *uint_table;
	unsigned char *startv;                 /* [npos][256] rank->char (pos 0)   */
	unsigned char *rowcnt;                 /* [npos][256] prev->#valid          */
	unsigned char *littmpl;                /* literal template (max length)     */
	int littmpl_len;                       /* == eff_maxlength template length  */
} mask_gpu_tables;

/* Per-length-loop unranking context (suffix-DP over the simplex). */
typedef struct {
	int valid;
	int limit;          /* generated positions active at this length */
	int len;            /* key length                                */
	int max_k;          /* maximum rank-sum                          */
	int ksize;          /* max_k + 1                                 */
	uint64_t total;     /* size of this length's keyspace            */
	uint64_t *suf;      /* [(limit+1) * ksize] suffix-DP table       */
} mask_gpu_loop;

extern int mask_gpu_max_loop;                       /* highest valid loop index   */
extern unsigned mask_gpu_serial;                    /* bumped when tables rebuilt */
extern const mask_gpu_tables *mask_gpu_get_tables(void);
extern const mask_gpu_loop   *mask_gpu_get_loop(int loop);
/* Reconstruct the plaintext for global index g of length-loop 'loop' (host-side
 * mirror of the kernel generator, used by the format's get_key). */
extern void mask_gpu_unrank_key(int loop, uint64_t g, char *out, int *out_len);

/*
 * Multi-length launch plan. The active length-loops are concatenated into one
 * node-local virtual candidate index space [0, total) so a SINGLE GPU launch can
 * span several lengths at once (restoring "all lengths tested together" that the
 * old internal-mask path got by packing mixed-length keys into one buffer).
 * Segment s covers virtual indices [vbase[s], vbase[s]+vcnt[s]) and maps to
 * candidate indices [lstart[s], lstart[s]+vcnt[s]) of length-loop loop[s]. The
 * format uploads a device mirror of this (plus the concatenated suf tables, with
 * per-segment element offset suf_off[]) and launches block-sized chunks of the
 * virtual space; each work-item locates its segment then unranks within it.
 */
/*
 * One virtual-space segment: a contiguous, K-ordered slice [lstart, lstart+vcnt)
 * of length-loop loop's keyspace, placed at virtual [vbase, vbase+vcnt). To
 * replicate the CPU's weighted round-robin over lengths, each length is sliced
 * into many small segments (a chunk per round-robin visit, sized by the Markov
 * length weight) and the segments are concatenated in round-robin order - so
 * walking the virtual space interleaves lengths while each length still advances
 * in increasing-K order. suf_off is this loop's element offset into the shared
 * concatenated suf buffer (segments of the same loop share it).
 */
typedef struct {
	uint64_t vbase;
	uint64_t vcnt;
	uint64_t lstart;
	uint64_t suf_off;
	int      loop;
	int      limit;
	int      len;
	int      max_k;
	int      ksize;
} mask_gpu_seg;

typedef struct {
	int nseg;
	uint64_t total;            /* node-local candidates, all lengths */
	uint64_t suf_total;        /* elements in the concatenated suf   */
	const mask_gpu_seg *seg;   /* nseg segments in virtual order      */
} mask_gpu_plan;

extern const mask_gpu_plan *mask_gpu_get_plan(void);
/* Map a virtual candidate index to its length-loop and the loop-local index g
 * (host mirror of the kernel's segment search, used by the format's get_key). */
extern void mask_gpu_virt_to_loop(uint64_t v, int *loop, uint64_t *g);

#endif
