/*
 * MD5 OpenCL kernel based on Solar Designer's MD5 algorithm implementation at:
 * https://openwall.info/wiki/people/solar/software/public-domain-source-code/md5
 *
 * This software is Copyright (c) 2010, Dhiru Kholia <dhiru.kholia at gmail.com>
 * and Copyright (c) 2012-2023, magnum
 * and Copyright (c) 2015, Sayantan Datta <std2048@gmail.com>
 * and it is hereby released to the general public under the following terms:
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted.
 *
 * Useful References:
 * 1. CUDA MD5 Hashing Experiments, http://majuric.org/software/cudamd5/
 * 2. oclcrack, http://sghctoma.extra.hu/index.php?p=entry&id=11
 * 3. http://people.eku.edu/styere/Encrypt/JS-MD5.html
 * 4. http://en.wikipedia.org/wiki/MD5#Algorithm
 */

#include "opencl_device_info.h"
#define AMD_PUTCHAR_NOCAST
#include "opencl_misc.h"
#include "opencl_mask.h"

#undef MD5_LUT3 /* No good for this format, just here for reference */

/* The basic MD5 functions */
#if MD5_LUT3
#define F(x, y, z)	lut3(x, y, z, 0xca)
#define G(x, y, z)	lut3(x, y, z, 0xe4)
#elif USE_BITSELECT
#define F(x, y, z)	bitselect((z), (y), (x))
#define G(x, y, z)	bitselect((y), (x), (z))
#else
#if HAVE_ANDNOT
#define F(x, y, z)	((x & y) ^ ((~x) & z))
#else
#define F(x, y, z)	(z ^ (x & (y ^ z)))
#endif
#define G(x, y, z)	((y) ^ ((z) & ((x) ^ (y))))
#endif

#if MD5_LUT3
#define H(x, y, z)	lut3(x, y, z, 0x96)
#define H2 H
#else
#define H(x, y, z)	(((x) ^ (y)) ^ (z))
#define H2(x, y, z)	((x) ^ ((y) ^ (z)))
#endif

#if MD5_LUT3
#define I(x, y, z)	lut3(x, y, z, 0x39)
#else
#define I(x, y, z)	((y) ^ ((x) | ~(z)))
#endif

/* The MD5 transformation for all four rounds. */
#define STEP(f, a, b, c, d, x, t, s)	  \
	(a) += f((b), (c), (d)) + (x) + (t); \
	    (a) = rotate((a), (uint)(s)); \
	    (a) += (b)

/* This handles an input of 0xffffffffU correctly */
#define BITMAP_SHIFT ((BITMAP_MASK >> 5) + 1)

INLINE void md5_encrypt(uint *hash, uint *W, uint len)
{
	hash[0] = 0x67452301;
	hash[1] = 0xefcdab89;
	hash[2] = 0x98badcfe;
	hash[3] = 0x10325476;

	/* Round 1 */
	STEP(F, hash[0], hash[1], hash[2], hash[3], W[0], 0xd76aa478, 7);
	STEP(F, hash[3], hash[0], hash[1], hash[2], W[1], 0xe8c7b756, 12);
	STEP(F, hash[2], hash[3], hash[0], hash[1], W[2], 0x242070db, 17);
	STEP(F, hash[1], hash[2], hash[3], hash[0], W[3], 0xc1bdceee, 22);
	STEP(F, hash[0], hash[1], hash[2], hash[3], W[4], 0xf57c0faf, 7);
	STEP(F, hash[3], hash[0], hash[1], hash[2], W[5], 0x4787c62a, 12);
	STEP(F, hash[2], hash[3], hash[0], hash[1], W[6], 0xa8304613, 17);
	STEP(F, hash[1], hash[2], hash[3], hash[0], W[7], 0xfd469501, 22);
	STEP(F, hash[0], hash[1], hash[2], hash[3], W[8], 0x698098d8, 7);
	STEP(F, hash[3], hash[0], hash[1], hash[2], W[9], 0x8b44f7af, 12);
	STEP(F, hash[2], hash[3], hash[0], hash[1], W[10], 0xffff5bb1, 17);
	STEP(F, hash[1], hash[2], hash[3], hash[0], W[11], 0x895cd7be, 22);
	STEP(F, hash[0], hash[1], hash[2], hash[3], W[12], 0x6b901122, 7);
	STEP(F, hash[3], hash[0], hash[1], hash[2], W[13], 0xfd987193, 12);
	STEP(F, hash[2], hash[3], hash[0], hash[1], W[14], 0xa679438e, 17);
	STEP(F, hash[1], hash[2], hash[3], hash[0], W[15], 0x49b40821, 22);

	/* Round 2 */
	STEP(G, hash[0], hash[1], hash[2], hash[3], W[1], 0xf61e2562, 5);
	STEP(G, hash[3], hash[0], hash[1], hash[2], W[6], 0xc040b340, 9);
	STEP(G, hash[2], hash[3], hash[0], hash[1], W[11], 0x265e5a51, 14);
	STEP(G, hash[1], hash[2], hash[3], hash[0], W[0], 0xe9b6c7aa, 20);
	STEP(G, hash[0], hash[1], hash[2], hash[3], W[5], 0xd62f105d, 5);
	STEP(G, hash[3], hash[0], hash[1], hash[2], W[10], 0x02441453, 9);
	STEP(G, hash[2], hash[3], hash[0], hash[1], W[15], 0xd8a1e681, 14);
	STEP(G, hash[1], hash[2], hash[3], hash[0], W[4], 0xe7d3fbc8, 20);
	STEP(G, hash[0], hash[1], hash[2], hash[3], W[9], 0x21e1cde6, 5);
	STEP(G, hash[3], hash[0], hash[1], hash[2], W[14], 0xc33707d6, 9);
	STEP(G, hash[2], hash[3], hash[0], hash[1], W[3], 0xf4d50d87, 14);
	STEP(G, hash[1], hash[2], hash[3], hash[0], W[8], 0x455a14ed, 20);
	STEP(G, hash[0], hash[1], hash[2], hash[3], W[13], 0xa9e3e905, 5);
	STEP(G, hash[3], hash[0], hash[1], hash[2], W[2], 0xfcefa3f8, 9);
	STEP(G, hash[2], hash[3], hash[0], hash[1], W[7], 0x676f02d9, 14);
	STEP(G, hash[1], hash[2], hash[3], hash[0], W[12], 0x8d2a4c8a, 20);

	/* Round 3 */
	STEP(H, hash[0], hash[1], hash[2], hash[3], W[5], 0xfffa3942, 4);
	STEP(H2, hash[3], hash[0], hash[1], hash[2], W[8], 0x8771f681, 11);
	STEP(H, hash[2], hash[3], hash[0], hash[1], W[11], 0x6d9d6122, 16);
	STEP(H2, hash[1], hash[2], hash[3], hash[0], W[14], 0xfde5380c, 23);
	STEP(H, hash[0], hash[1], hash[2], hash[3], W[1], 0xa4beea44, 4);
	STEP(H2, hash[3], hash[0], hash[1], hash[2], W[4], 0x4bdecfa9, 11);
	STEP(H, hash[2], hash[3], hash[0], hash[1], W[7], 0xf6bb4b60, 16);
	STEP(H2, hash[1], hash[2], hash[3], hash[0], W[10], 0xbebfbc70, 23);
	STEP(H, hash[0], hash[1], hash[2], hash[3], W[13], 0x289b7ec6, 4);
	STEP(H2, hash[3], hash[0], hash[1], hash[2], W[0], 0xeaa127fa, 11);
	STEP(H, hash[2], hash[3], hash[0], hash[1], W[3], 0xd4ef3085, 16);
	STEP(H2, hash[1], hash[2], hash[3], hash[0], W[6], 0x04881d05, 23);
	STEP(H, hash[0], hash[1], hash[2], hash[3], W[9], 0xd9d4d039, 4);
	STEP(H2, hash[3], hash[0], hash[1], hash[2], W[12], 0xe6db99e5, 11);
	STEP(H, hash[2], hash[3], hash[0], hash[1], W[15], 0x1fa27cf8, 16);
	STEP(H2, hash[1], hash[2], hash[3], hash[0], W[2], 0xc4ac5665, 23);

	/* Round 4 */
	STEP(I, hash[0], hash[1], hash[2], hash[3], W[0], 0xf4292244, 6);
	STEP(I, hash[3], hash[0], hash[1], hash[2], W[7], 0x432aff97, 10);
	STEP(I, hash[2], hash[3], hash[0], hash[1], W[14], 0xab9423a7, 15);
	STEP(I, hash[1], hash[2], hash[3], hash[0], W[5], 0xfc93a039, 21);
	STEP(I, hash[0], hash[1], hash[2], hash[3], W[12], 0x655b59c3, 6);
	STEP(I, hash[3], hash[0], hash[1], hash[2], W[3], 0x8f0ccc92, 10);
	STEP(I, hash[2], hash[3], hash[0], hash[1], W[10], 0xffeff47d, 15);
	STEP(I, hash[1], hash[2], hash[3], hash[0], W[1], 0x85845dd1, 21);
	STEP(I, hash[0], hash[1], hash[2], hash[3], W[8], 0x6fa87e4f, 6);
	STEP(I, hash[3], hash[0], hash[1], hash[2], W[15], 0xfe2ce6e0, 10);
	STEP(I, hash[2], hash[3], hash[0], hash[1], W[6], 0xa3014314, 15);
	STEP(I, hash[1], hash[2], hash[3], hash[0], W[13], 0x4e0811a1, 21);
	STEP(I, hash[0], hash[1], hash[2], hash[3], W[4], 0xf7537e82, 6);
	STEP(I, hash[3], hash[0], hash[1], hash[2], W[11], 0xbd3af235, 10);
	STEP(I, hash[2], hash[3], hash[0], hash[1], W[2], 0x2ad7d2bb, 15);
	STEP(I, hash[1], hash[2], hash[3], hash[0], W[9], 0xeb86d391, 21);
}

INLINE void cmp_final(uint gid,
		uint iter,
		uint *hash,
		__global uint *offset_table,
		__global uint *hash_table,
		__global uint *return_hashes,
		volatile __global uint *output,
		volatile __global uint *bitmap_dupe) {

	uint t, offset_table_index, hash_table_index;
	unsigned long LO, HI;
	unsigned long p;

	HI = ((unsigned long)hash[3] << 32) | (unsigned long)hash[2];
	LO = ((unsigned long)hash[1] << 32) | (unsigned long)hash[0];

	p = (HI % OFFSET_TABLE_SIZE) * SHIFT64_OT_SZ;
	p += LO % OFFSET_TABLE_SIZE;
	p %= OFFSET_TABLE_SIZE;
	offset_table_index = (unsigned int)p;

	//error: chances of overflow is extremely low.
	LO += (unsigned long)offset_table[offset_table_index];

	p = (HI % HASH_TABLE_SIZE) * SHIFT64_HT_SZ;
	p += LO % HASH_TABLE_SIZE;
	p %= HASH_TABLE_SIZE;
	hash_table_index = (unsigned int)p;

	if (hash_table[hash_table_index] == hash[0])
	if (hash_table[HASH_TABLE_SIZE + hash_table_index] == hash[1])
	{
/*
 * Prevent duplicate keys from cracking same hash
 */
		if (!(atomic_or(&bitmap_dupe[hash_table_index/32], (1U << (hash_table_index % 32))) & (1U << (hash_table_index % 32)))) {
			t = atomic_inc(&output[0]);
			output[1 + 3 * t] = gid;
			output[2 + 3 * t] = iter;
			output[3 + 3 * t] = hash_table_index;
			return_hashes[2 * t] = hash[2];
			return_hashes[2 * t + 1] = hash[3];
		}
	}
}

INLINE void cmp(uint gid,
		uint iter,
		uint *hash,
#if USE_LOCAL_BITMAPS
		__local
#else
		__global
#endif
		uint *bitmaps,
		__global uint *offset_table,
		__global uint *hash_table,
		__global uint *return_hashes,
		volatile __global uint *output,
		volatile __global uint *bitmap_dupe) {
	uint bitmap_index, tmp = 1;

	hash[0] += 0x67452301;
	hash[1] += 0xefcdab89;
	hash[2] += 0x98badcfe;
	hash[3] += 0x10325476;

#if SELECT_CMP_STEPS > 4
	bitmap_index = hash[0] & BITMAP_MASK;
	tmp &= (bitmaps[bitmap_index >> 5] >> (bitmap_index & 31)) & 1U;
	bitmap_index = (hash[0] >> 16) & BITMAP_MASK;
	tmp &= (bitmaps[BITMAP_SHIFT + (bitmap_index >> 5)] >> (bitmap_index & 31)) & 1U;
	bitmap_index = hash[1] & BITMAP_MASK;
	tmp &= (bitmaps[BITMAP_SHIFT * 2 + (bitmap_index >> 5)] >> (bitmap_index & 31)) & 1U;
	bitmap_index = (hash[1] >> 16) & BITMAP_MASK;
	tmp &= (bitmaps[BITMAP_SHIFT * 3 + (bitmap_index >> 5)] >> (bitmap_index & 31)) & 1U;
	bitmap_index = hash[2] & BITMAP_MASK;
	tmp &= (bitmaps[BITMAP_SHIFT * 4 + (bitmap_index >> 5)] >> (bitmap_index & 31)) & 1U;
	bitmap_index = (hash[2] >> 16) & BITMAP_MASK;
	tmp &= (bitmaps[BITMAP_SHIFT * 5 + (bitmap_index >> 5)] >> (bitmap_index & 31)) & 1U;
	bitmap_index = hash[3] & BITMAP_MASK;
	tmp &= (bitmaps[BITMAP_SHIFT * 6 + (bitmap_index >> 5)] >> (bitmap_index & 31)) & 1U;
	bitmap_index = (hash[3] >> 16) & BITMAP_MASK;
	tmp &= (bitmaps[BITMAP_SHIFT * 7 + (bitmap_index >> 5)] >> (bitmap_index & 31)) & 1U;
#elif SELECT_CMP_STEPS > 2
	bitmap_index = hash[3] & BITMAP_MASK;
	tmp &= (bitmaps[bitmap_index >> 5] >> (bitmap_index & 31)) & 1U;
	bitmap_index = hash[2] & BITMAP_MASK;
	tmp &= (bitmaps[BITMAP_SHIFT + (bitmap_index >> 5)] >> (bitmap_index & 31)) & 1U;
	bitmap_index = hash[1] & BITMAP_MASK;
	tmp &= (bitmaps[BITMAP_SHIFT * 2 + (bitmap_index >> 5)] >> (bitmap_index & 31)) & 1U;
	bitmap_index = hash[0] & BITMAP_MASK;
	tmp &= (bitmaps[BITMAP_SHIFT * 3 + (bitmap_index >> 5)] >> (bitmap_index & 31)) & 1U;
#elif SELECT_CMP_STEPS > 1
	bitmap_index = hash[3] & BITMAP_MASK;
	tmp &= (bitmaps[bitmap_index >> 5] >> (bitmap_index & 31)) & 1U;
	bitmap_index = hash[2] & BITMAP_MASK;
	tmp &= (bitmaps[BITMAP_SHIFT + (bitmap_index >> 5)] >> (bitmap_index & 31)) & 1U;
#else
	bitmap_index = hash[3] & BITMAP_MASK;
	tmp &= (bitmaps[bitmap_index >> 5] >> (bitmap_index & 31)) & 1U;
#endif

	if (tmp)
		cmp_final(gid, iter, hash, offset_table, hash_table, return_hashes, output, bitmap_dupe);
}

#define USE_CONST_CACHE \
	(CONST_CACHE_SIZE >= (NUM_INT_KEYS * 4))

/* OpenCL kernel entry point. Copy key to be hashed from
 * global to local (thread) memory. Break the key into 16 32-bit (uint)
 * words. MD5 hash of a key is 128 bit (uint4). */
__kernel void md5(__global uint *keys,
		  __global uint *index,
		  __global uint *int_key_loc,
#if USE_CONST_CACHE
		  constant
#else
		  __global
#endif
		  uint *int_keys,
		  __global uint *bitmaps,
		  __global uint *offset_table,
		  __global uint *hash_table,
		  __global uint *return_hashes,
		  volatile __global uint *out_hash_ids,
		  volatile __global uint *bitmap_dupe)
{
	uint i;
	uint gid = get_global_id(0);
	uint base = index[gid];
	uint W[16] = { 0 };
	uint len = base & 63;
	uint hash[4];

#if NUM_INT_KEYS > 1 && !IS_STATIC_GPU_MASK
	uint ikl = int_key_loc[gid];
	uint loc0 = ikl & 0xff;
#if MASK_FMT_INT_PLHDR > 1
#if LOC_1 >= 0
	uint loc1 = (ikl & 0xff00) >> 8;
#endif
#endif
#if MASK_FMT_INT_PLHDR > 2
#if LOC_2 >= 0
	uint loc2 = (ikl & 0xff0000) >> 16;
#endif
#endif
#if MASK_FMT_INT_PLHDR > 3
#if LOC_3 >= 0
	uint loc3 = (ikl & 0xff000000) >> 24;
#endif
#endif
#endif

#if !IS_STATIC_GPU_MASK
#define GPU_LOC_0 loc0
#define GPU_LOC_1 loc1
#define GPU_LOC_2 loc2
#define GPU_LOC_3 loc3
#else
#define GPU_LOC_0 LOC_0
#define GPU_LOC_1 LOC_1
#define GPU_LOC_2 LOC_2
#define GPU_LOC_3 LOC_3
#endif

#if USE_LOCAL_BITMAPS
	uint lid = get_local_id(0);
	uint lws = get_local_size(0);
	__local uint s_bitmaps[BITMAP_SHIFT * SELECT_CMP_STEPS];

	for (i = lid; i < BITMAP_SHIFT * SELECT_CMP_STEPS; i+= lws)
		s_bitmaps[i] = bitmaps[i];

	barrier(CLK_LOCAL_MEM_FENCE);
#endif

	keys += base >> 6;

	for (i = 0; i < (len+3)/4; i++)
		W[i] = *keys++;

	PUTCHAR(W, len, 0x80);
	W[14] = len << 3;

	for (i = 0; i < NUM_INT_KEYS; i++) {
#if NUM_INT_KEYS > 1
		PUTCHAR(W, GPU_LOC_0, (int_keys[i] & 0xff));
#if MASK_FMT_INT_PLHDR > 1
#if LOC_1 >= 0
		PUTCHAR(W, GPU_LOC_1, ((int_keys[i] & 0xff00) >> 8));
#endif
#endif
#if MASK_FMT_INT_PLHDR > 2
#if LOC_2 >= 0
		PUTCHAR(W, GPU_LOC_2, ((int_keys[i] & 0xff0000) >> 16));
#endif
#endif
#if MASK_FMT_INT_PLHDR > 3
#if LOC_3 >= 0
		PUTCHAR(W, GPU_LOC_3, ((int_keys[i] & 0xff000000) >> 24));
#endif
#endif
#endif
		md5_encrypt(hash, W, len);
		cmp(gid, i, hash,
#if USE_LOCAL_BITMAPS
		    s_bitmaps
#else
		    bitmaps
#endif
		    , offset_table, hash_table, return_hashes, out_hash_ids, bitmap_dupe);
	}
}

#ifdef GPU_GEN
/*
 * K-ordered GPU password generator.
 *
 * Instead of receiving host-materialized keys, each work-item is given a global
 * candidate index g = gbase + gid and unranks it into a simplex point
 * (rank-sum K, per-position ranks iter[]) using the suffix-DP table 'suf', then
 * materializes the password left-to-right through the Markov tables - the exact
 * mirror of mask.c:mask_gpu_unrank_key(). Candidates within a length come out in
 * increasing K (Markov probability) order.
 *
 * Args 0..9 match the layout opencl_hash_check_128 hardcodes (0..3 set by the
 * format, 4..9 the hash-check buffers); the generator inputs start at arg 10.
 */
/*
 * GEN_MAX_POS sizes the per-work-item key[]/iter[] arrays and GEN_NDW is the
 * number of 32-bit message words actually packed/hashed. The host overrides both
 * from the run's max length (-D GEN_MAX_POS=, -D GEN_NDW=) so a short mask (e.g.
 * ?d^7) packs 2 words instead of 14 and spills a handful of bytes instead of 128;
 * the defaults below cover the full 55-byte plaintext when unset.
 */
#ifndef GEN_MAX_POS
#define GEN_MAX_POS 64
#endif
#ifndef GEN_NDW
#define GEN_NDW 14
#endif

/*
 * Register-resident generator variant (build with -D GEN_REGS). The per-candidate
 * state machine (full-unrank, simplex advance, materialize) is rewritten as
 * fully-unrolled, compile-time-indexed sweeps of width GEN_REG_MAX, so iter[]
 * carries only literal indices and the compiler can promote it to registers (no
 * local-memory spill). The order is a bit-exact mirror of the runtime-indexed
 * path below (and of mask.c:simplex_next_state). The host guarantees every
 * length-loop's position count (glimit) is <= GEN_REG_MAX. key[] stays in local
 * memory regardless (it is indexed by the runtime g_keypos[] indirection).
 */
#ifndef GEN_REG_MAX
#define GEN_REG_MAX 16
#endif

/* Device mirror of one mask_gpu_plan segment (must match gen_seg in the format).
 * Four ulongs then four uints = 48 bytes, naturally 8-aligned. */
typedef struct {
	ulong vbase;
	ulong vcnt;
	ulong lstart;
	ulong suf_off;
	uint  limit;
	uint  len;
	uint  max_k;
	uint  ksize;
} gen_seg;

__kernel void md5_gen(__global uint *keys_unused,
		  __global uint *index_unused,
		  __global uint *int_key_loc_unused,
		  __global uint *int_keys_unused,
		  __global uint *bitmaps,
		  __global uint *offset_table,
		  __global uint *hash_table,
		  __global uint *return_hashes,
		  volatile __global uint *out_hash_ids,
		  volatile __global uint *bitmap_dupe,
		  __global ulong *suf,
		  __global uint *g_table_packed,
		  /*
		   * The small per-position tables are uniform across all work-items and
		   * read every candidate (g_count in the odometer; keypos/cstart/chars0
		   * and the Markov startv/rowcnt in materialization), so they live in
		   * __constant for broadcast via the hardware constant cache. suf and the
		   * big [npos][256][256] g_table can exceed the 64 KiB constant limit, so
		   * they stay __global; g_littmpl is written once per work-item.
		   */
		  __constant uchar *g_startv,
		  __constant uchar *g_rowcnt,
		  __global   uchar *g_littmpl,
		  __constant int   *g_keypos,
		  __constant int   *g_count,
		  __constant uchar *g_cstart,
		  __constant uchar *g_chars0,
		  __global gen_seg *segs,
		  uint nseg,
		  ulong gbase,
		  uint gcount,
		  uint gR)
{
	uint i, j;
	uint gid = get_global_id(0);

#if USE_LOCAL_BITMAPS
	uint lid = get_local_id(0);
	uint lws = get_local_size(0);
	__local uint s_bitmaps[BITMAP_SHIFT * SELECT_CMP_STEPS];

	for (i = lid; i < BITMAP_SHIFT * SELECT_CMP_STEPS; i += lws)
		s_bitmaps[i] = bitmaps[i];

	barrier(CLK_LOCAL_MEM_FENCE);
#endif

	{
		uint W[16] = { 0 };
		uint hash[4];
		uchar key[GEN_MAX_POS];
#ifdef GEN_REGS
		uchar iter[GEN_REG_MAX];
#else
		uchar iter[GEN_MAX_POS];
#endif
		ulong gl0 = (ulong)gid * gR;

		/*
		 * Each work-item handles gR contiguous candidates gl = gid*gR + j
		 * (j = 0..gR-1), virtual index V = gbase + gl. The virtual space
		 * concatenates all active length-loops (segs[]) so a single launch spans
		 * several lengths; the stable gid sort in ocl_hc keeps reported cracks in
		 * exact increasing-virtual order, and the sub-index j is carried in cmp's
		 * int_index slot so get_key can reconstruct V.
		 *
		 * Per candidate we first locate its segment. Segment index only grows
		 * across the j loop (V increases, segs are ascending in length), so the
		 * MD5 message length only grows - words written by a shorter length stay
		 * valid padding for the longer one. On a segment change we rebuild the
		 * length template + padding and full-unrank from the suffix-DP. Within a
		 * segment the iter[] vector advances incrementally (mirror of
		 * mask.c:simplex_next_state), rebuilding key[] from the leftmost changed
		 * position so md5_encrypt dominates the per-candidate cost.
		 */
		uint cur_seg = 0xffffffff;
		ulong seg_vbase = 0, seg_vend = 0, seg_lstart = 0, seg_suf_off = 0;
		uint glimit = 0, glen = 0, gmax_k = 0, gksize = 0;
		uint len = 0, cur_k = 0;

		for (j = 0; j < gR; j++) {
			ulong gl = gl0 + j;
			ulong V, g;
			uint mfrom;     /* leftmost iter[] position that changed */
			int full;

			if (gl >= gcount)
				break;
			V = gbase + gl;

			/* (Re)locate the segment when V leaves the cached one. Segments are
			 * in ascending vbase order, so binary-search the largest vbase <= V
			 * (the list is long under the round-robin interleave). The gR run is
			 * contiguous and usually stays in one segment, so this is paid about
			 * once per work-item. */
			if (cur_seg == 0xffffffff || V < seg_vbase || V >= seg_vend) {
				uint lo = 0, hi = nseg - 1, s = 0;

				while (lo <= hi) {
					uint mid = (lo + hi) >> 1;

					if (segs[mid].vbase <= V) {
						s = mid;
						lo = mid + 1;
					} else {
						if (mid == 0)
							break;
						hi = mid - 1;
					}
				}

				cur_seg     = s;
				seg_vbase   = segs[s].vbase;
				seg_vend    = seg_vbase + segs[s].vcnt;
				seg_lstart  = segs[s].lstart;
				seg_suf_off = segs[s].suf_off;
				glimit      = segs[s].limit;
				glen        = segs[s].len;
				gmax_k      = segs[s].max_k;
				gksize      = segs[s].ksize;

				/* Rebuild key[] template + folded MD5 padding for this length.
				 * Zero the whole packed range first (constant bound so it can
				 * unroll), lay down the literal template, then drop the single
				 * 0x80 terminator at position len. Everything from len+1..55 is
				 * already 0, so the per-candidate word pack below can use
				 * compile-time indices for ALL of W[0..13] - no per-length
				 * masking - which is what keeps W register-resident through
				 * md5_encrypt. W[14] (bit length) and W[15] are set here once per
				 * segment and never touched by the pack. */
				len = glen;
#pragma unroll
				for (i = 0; i < GEN_MAX_POS; i++)
					key[i] = (i < len) ? g_littmpl[i] : 0;
				key[len] = 0x80;
				W[14] = len << 3;
				W[15] = 0;
				full = 1;              /* must full-unrank on a new segment */
			} else {
				full = 0;
			}

			g = seg_lstart + (V - seg_vbase);   /* loop-local candidate index */

#ifdef GEN_REGS
			/*
			 * Register-resident state machine. Every iter[] index is a literal
			 * (the position loops are fully unrolled to the compile-time bound
			 * GEN_REG_MAX and masked by i<glimit), so iter[] lives in registers.
			 * Each block below is the exact order-equivalent of the runtime path
			 * in the #else branch.
			 */
			if (full) {
				/* Full unrank (mirror of #else full path): outer position loop
				 * unrolled with literal i; the inner descending-rank search is a
				 * scalar loop that never indexes iter[]. */
				ulong fw = g, rank;
				uint remaining_k;

				cur_k = 0;
				while (cur_k <= gmax_k && fw >= suf[seg_suf_off + cur_k]) {
					fw -= suf[seg_suf_off + cur_k];
					cur_k++;
				}
				remaining_k = cur_k;
				rank = fw;
#pragma unroll
				for (i = 0; i < GEN_REG_MAX; i++) {
					if (i < glimit) {
						int C = g_count[i];
						int vmax = (remaining_k < (uint)(C - 1)) ? (int)remaining_k : (C - 1);
						int vv;

						for (vv = vmax; vv >= 0; vv--) {
							ulong cnt = suf[seg_suf_off + (i + 1) * gksize + (remaining_k - vv)];
							if (rank < cnt)
								break;
							rank -= cnt;
						}
						iter[i] = (uchar)vv;
						remaining_k -= vv;
					}
				}
				mfrom = 0;
			} else {
				/*
				 * Incremental advance, branchless fixed sweeps (mirror of
				 * simplex_next_state). 1) forward sweep keeps the RIGHTMOST
				 * shiftable wheel as pivot (== the C scan's first hit from the
				 * right). 2) if found, a left-to-right sweep carrying the residual
				 * weight w applies the shift+pour. 3) else bump K and repack from
				 * the left.
				 */
				int pivot = -1;
				#pragma unroll
				for (i = 0; i < GEN_REG_MAX; i++) {
					int is_valid = ((int)i + 1 < (int)glimit) && (iter[i] > 0) && (iter[i + 1] < g_count[i + 1] - 1);
					pivot = is_valid ? (int)i : pivot;
				}

				/* * --- START OF PURE BRANCHLESS SIMPLEX TRANSFORMATION ---
				 * Instead of splitting the wavefront with if/else blocks, we use
				 * state predicates (0 or 1) to execute both pathways arithmetically.
				 */
				int has_pivot    = (pivot >= 0);
				int is_exhausted = !has_pivot;

				// 1. Unconditional K-layer advance (bypasses the master branch)
				cur_k += is_exhausted;

				// 2. Branchless residual weight (w) collection for the shift phase
				int w = 0;
#pragma unroll
				for (i = 0; i < GEN_REG_MAX; i++) {
					int mask_w = (i < glimit) && ((int)i >= pivot + 2);
					w += iter[i] * mask_w;
				}

				// 3. Unified Weight Register: holds 'w' if shifting, or 'cur_k' if expanding a new K-layer
				int r_weight = has_pivot ? w : (int)cur_k;

				// 4. The Unified, Divergence-Free State Mutation Sweep
#pragma unroll
				for (i = 0; i < GEN_REG_MAX; i++) {
					int cond_glimit = (i < glimit);
					int lookup_idx  = cond_glimit ? i : 0; // Safeguard constant buffer reads

					// Compute positional relationships relative to the pivot matrix
					int is_before_pivot = has_pivot && ((int)i < pivot);
					int is_pivot        = has_pivot && ((int)i == pivot);
					// If the layer is exhausted, every active wheel behaves as if it's "after the pivot"
					int is_after_pivot  = is_exhausted || ((int)i > pivot);

					// Evaluate candidate modifications for all execution paths simultaneously
					int val_before = iter[i];
					int val_pivot  = iter[i] - 1;

					// Path-A specific baseline character increment; evaluates to 0 during layer exhaustion
					int base = (has_pivot && ((int)i == pivot + 1)) ? (iter[i] + 1) : 0;
					int cap  = g_count[lookup_idx] - 1 - base;

					// Pour weight evenly using OpenCL's native, hardware-level branchless min()
					int add       = min(r_weight, cap);
					int val_after = base + add;

					// Apply loop-carried dependency to the weight pool using arithmetic subtraction
					r_weight -= add * (cond_glimit && is_after_pivot);

					// Combine distinct execution branches into a single arithmetic multiplexer
					int next_val = (is_before_pivot * val_before) +
					               (is_pivot        * val_pivot)  +
					               (is_after_pivot  * val_after);

					// Write back to private registers only if the loop index is valid for this segment
					iter[i] = cond_glimit ? (uchar)next_val : iter[i];
				}

				// 5. Branchless tracking assignment for the leftmost altered wheel.
				// The exhausted (no-pivot) layer advance is already folded into the
				// sweep above: cur_k was bumped (is_exhausted), r_weight seeded with
				// cur_k, and every wheel took the is_after_pivot fill path - the exact
				// left-to-right repack of the old else block. So mfrom is 0 there.
				mfrom = has_pivot ? (uint)pivot : 0;

				/* --- END OF PURE BRANCHLESS SIMPLEX TRANSFORMATION --- */
			}

			/* Materialize key[mfrom..glimit) - literal-indexed sweep, processed in
			 * increasing position order so the Markov key[kp-1] dependency holds. */
#pragma unroll
			for (i = 0; i < GEN_REG_MAX; i++) {
				if (i >= mfrom && i < glimit) {
					int kp = g_keypos[i];
					uchar cs = g_cstart[i];

					if (cs) {
						key[kp] = cs + iter[i];
					} else if (i == 0) {
						key[kp] = g_startv[iter[0]];
					} else {
						uchar prev = key[kp - 1];
						int avail = g_rowcnt[i * 256 + prev];
						int ti = iter[i];

						if (avail > 0) {
							if (ti >= avail)
								ti = avail - 1;
							/* Same coalesced packed-table read as the runtime
							 * path: fetch the uint32 holding 4 chars, extract
							 * the ti%4 byte. */
							int block_idx = (i * 256 + prev) * 64 + (ti >> 2);
							uint packed_chars = g_table_packed[block_idx];
							uint shift_amount = (ti & 3) << 3;
							key[kp] = (uchar)(packed_chars >> shift_amount);
						} else {
							key[kp] = g_chars0[i];
						}
					}
				}
			}
#else
			if (full) {
				/* Full unrank: skip whole K-layers, then unrank within the
				 * target layer in descending-lexicographic order (matches
				 * simplex_next_state). suf is shared, offset by seg_suf_off. */
				ulong fw = g, rank;
				uint remaining_k;

				cur_k = 0;
				while (cur_k <= gmax_k && fw >= suf[seg_suf_off + cur_k]) {
					fw -= suf[seg_suf_off + cur_k];
					cur_k++;
				}
				remaining_k = cur_k;
				rank = fw;
				for (i = 0; i < glimit; i++) {
					int C = g_count[i];
					int vmax = (remaining_k < (uint)(C - 1)) ? (int)remaining_k : (C - 1);
					int vv;

					for (vv = vmax; vv >= 0; vv--) {
						ulong cnt = suf[seg_suf_off + (i + 1) * gksize + (remaining_k - vv)];
						if (rank < cnt)
							break;
						rank -= cnt;
					}
					iter[i] = (uchar)vv;
					remaining_k -= vv;
				}
				mfrom = 0;
			} else {
				/* Incremental advance (mirror of simplex_next_state): scan
				 * right-to-left for a wheel that can shift one unit of rank
				 * weight to its right neighbour, keeping sum K constant; pour the
				 * residual right-side weight back as far left as possible. When
				 * no shift is possible the layer is exhausted, so bump cur_k and
				 * reset to the first point of the next layer. */
				int p, q, advanced = 0;

				for (p = (int)glimit - 2; p >= 0; p--) {
					if (iter[p] > 0 && iter[p + 1] < g_count[p + 1] - 1) {
						int w = 0;

						iter[p]--;
						iter[p + 1]++;
						for (q = p + 2; q < (int)glimit; q++) {
							w += iter[q];
							iter[q] = 0;
						}
						q = p + 1;
						while (w > 0 && q < (int)glimit) {
							int max_allowed = g_count[q] - 1 - iter[q];
							int add = (w > max_allowed) ? max_allowed : w;

							iter[q] += add;
							w -= add;
							q++;
						}
						mfrom = (uint)p;
						advanced = 1;
						break;
					}
				}
				if (!advanced) {
					int rem;

					cur_k++;
					rem = (int)cur_k;
					for (p = 0; p < (int)glimit; p++) {
						int mx = g_count[p] - 1;

						if (rem <= mx) {
							iter[p] = (uchar)rem;
							rem = 0;
						} else {
							iter[p] = (uchar)mx;
							rem -= mx;
						}
					}
					mfrom = 0;
				}
			}

			/* Materialize key[mfrom..glimit) through the Markov tables. Positions
			 * left of mfrom are unchanged, so key[kp-1] feeding mfrom is valid. */
			for (i = mfrom; i < glimit; i++) {
				int kp = g_keypos[i];
				uchar cs = g_cstart[i];

				if (cs) {
					key[kp] = cs + iter[i];
				} else if (i == 0) {
					key[kp] = g_startv[iter[0]];
				} else {
					uchar prev = key[kp - 1];
					int avail = g_rowcnt[i * 256 + prev];
					int ti = iter[i];

					if (avail > 0) {
						if (ti >= avail)
							ti = avail - 1;

						// 1. Calculate the 32-bit block index (ti / 4)
						int block_idx = (i * 256 + prev) * 64 + (ti >> 2);

						// 2. Fetch the 32-bit block (hardware aligned, highly coalesced)
						uint packed_chars = g_table_packed[block_idx];

						// 3. Extract the specific 8-bit character (ti % 4 * 8)
						uint shift_amount = (ti & 3) << 3;
						key[kp] = (uchar)(packed_chars >> shift_amount);

					} else {
						key[kp] = g_chars0[i];
					}
				}
			}
#endif /* GEN_REGS */

			/* Pack the message words with COMPILE-TIME indices so W[0..15] stay
			 * in registers across md5_encrypt. (A dynamic W[] index, e.g. a
			 * length-dependent loop bound, forces the whole array to local
			 * memory and md5 then reloads each word from local every round.)
			 * key[0..GEN_NDW*4) is fully defined - literals/Markov chars, the
			 * 0x80 terminator and zero padding; W[GEN_NDW..13] stay 0 from the
			 * W initializer and W[14]/W[15] were set on the segment change, so
			 * packing exactly GEN_NDW words covers every message of this run. */
#pragma unroll
			for (i = 0; i < GEN_NDW; i++)
				W[i] = (uint)key[4 * i] |
				       ((uint)key[4 * i + 1] << 8) |
				       ((uint)key[4 * i + 2] << 16) |
				       ((uint)key[4 * i + 3] << 24);

			md5_encrypt(hash, W, len);
			cmp(gid, j, hash,
#if USE_LOCAL_BITMAPS
			    s_bitmaps
#else
			    bitmaps
#endif
			    , offset_table, hash_table, return_hashes, out_hash_ids, bitmap_dupe);
		}
	}
}
#endif /* GPU_GEN */
