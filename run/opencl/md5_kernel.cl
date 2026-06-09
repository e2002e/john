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
#define GEN_MAX_POS 64

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
		  __global uchar *g_table,
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
		  uint glimit,
		  uint glen,
		  uint gmax_k,
		  uint gksize,
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
		uint len = glen;
		uint W[16] = { 0 };
		uint hash[4];
		uchar key[GEN_MAX_POS];
		uchar iter[GEN_MAX_POS];
		ulong gl0 = (ulong)gid * gR;

		for (i = 0; i < len; i++)
			key[i] = g_littmpl[i];

		/*
		 * Fold the MD5 padding into key[] once: the 0x80 terminator sits at the
		 * constant position len, and the up-to-three tail bytes that share its
		 * final word are zeroed. The per-candidate W build then packs key[]
		 * word-wise (ndw words, padding included) with no dynamic byte index into
		 * W, so W stays in registers and md5_encrypt runs register-resident -
		 * the byte-wise PUTCHAR(W, i, ...) loop it replaces spilled W to local
		 * memory, which the 64 md5 rounds then paid for on every candidate.
		 */
		key[len]     = 0x80;
		key[len + 1] = 0;
		key[len + 2] = 0;
		key[len + 3] = 0;
		W[14] = len << 3;
		uint ndw = (len + 4) / 4;   /* data words incl. the 0x80 pad byte */

		/*
		 * Each work-item handles gR contiguous candidates gl = gid*gR + j
		 * (j = 0..gR-1), global index g = gbase + gl. Contiguous layout plus
		 * the stable gid sort in ocl_hc keeps reported cracks in exact
		 * increasing-K order; the sub-index j is carried in cmp's int_index
		 * slot so get_key can reconstruct g. len is constant across the loop,
		 * so W's padding stays valid and only the data bytes are rewritten.
		 *
		 * Only the first candidate (j == 0) is unranked from scratch via the
		 * suffix-DP - an O(glimit * charset) walk over global suf[]. The rest
		 * advance the iter[] vector incrementally with the device mirror of
		 * mask.c:simplex_next_state(), which only touches a few wheels and lets
		 * us rebuild key[] from the leftmost changed position onward. This keeps
		 * the heavy global-memory unrank off the per-candidate hot path (gR-1 of
		 * every gR candidates), so md5_encrypt dominates again. cur_k carries the
		 * current K-layer across the j loop for the layer-rollover reset.
		 */
		uint cur_k = 0;

		for (j = 0; j < gR; j++) {
			ulong gl = gl0 + j;
			uint mfrom;     /* leftmost iter[] position that changed */

			if (gl >= gcount)
				break;

			if (j == 0) {
				/* Full unrank: skip whole K-layers, then unrank within the
				 * target layer in descending-lexicographic order (matches
				 * simplex_next_state). */
				ulong g = gbase + gl;
				ulong fw = g, rank;
				uint remaining_k;

				cur_k = 0;
				while (cur_k <= gmax_k && fw >= suf[cur_k]) {
					fw -= suf[cur_k];
					cur_k++;
				}
				remaining_k = cur_k;
				rank = fw;
				for (i = 0; i < glimit; i++) {
					int C = g_count[i];
					int vmax = (remaining_k < (uint)(C - 1)) ? (int)remaining_k : (C - 1);
					int vv;

					for (vv = vmax; vv >= 0; vv--) {
						ulong cnt = suf[(i + 1) * gksize + (remaining_k - vv)];
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
						key[kp] = g_table[(i * 256 + prev) * 256 + ti];
					} else {
						key[kp] = g_chars0[i];
					}
				}
			}

			/* Word-wise pack (key[] is padded so the last word carries the
			 * 0x80); linear W[i] writes keep W register-resident. */
			for (i = 0; i < ndw; i++)
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
