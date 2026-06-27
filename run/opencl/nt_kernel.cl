/*
 * NTLM kernel (OpenCL 1.2 conformant)
 *
 * Written by Alain Espinosa <alainesp at gmail.com> in 2010 and modified by
 * Samuele Giovanni Tonon in 2011. No copyright is claimed, and
 * the software is hereby placed in the public domain.
 * In case this attempt to disclaim copyright and place the software in the
 * public domain is deemed null and void, then the software is
 * Copyright (c) 2010 Alain Espinosa
 * Copyright (c) 2011 Samuele Giovanni Tonon
 * Copyright (c) 2015 Sayantan Datta <sdatta at openwall.com>
 * Copyright (c) 2015-2023 magnum
 * and it is hereby released to the general public under the following terms:
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted.
 *
 * There's ABSOLUTELY NO WARRANTY, express or implied.
 *
 * (This is a heavily cut-down "BSD license".)
 */

#define AMD_PUTCHAR_NOCAST
#include "opencl_misc.h"
#include "opencl_md4.h"
#include "opencl_unicode.h"
#include "opencl_mask.h"

//Init values
#define INIT_A 0x67452301
#define INIT_B 0xefcdab89
#define INIT_C 0x98badcfe
#define INIT_D 0x10325476

#define SQRT_2 0x5a827999
#define SQRT_3 0x6ed9eba1

/*
 * If enabled, will check bitmap after calculating just the
 * first 32 bits of 'b' (does not apply to nt-long-opencl).
 */
#define EARLY_REJECT	1

#if USE_LOCAL_BITMAPS
#define BITMAPS_TYPE	__local
#else
#define BITMAPS_TYPE	__global
#endif

#define USE_CONST_CACHE \
	(CONST_CACHE_SIZE >= (NUM_INT_KEYS * 4))

#if USE_CONST_CACHE
#define CACHE_TYPE	__constant
#else
#define CACHE_TYPE	__global
#endif

/* This handles an input of 0xffffffffU correctly */
#define BITMAP_SHIFT ((BITMAP_MASK >> 5) + 1)

INLINE int nt_crypt(uint *hash, uint *nt_buffer, uint md4_size, BITMAPS_TYPE uint *bitmaps)
{
	MD4_G_VARS

	/* Round 1 */
	hash[0] = 0xFFFFFFFF + nt_buffer[0] ; hash[0] = rotate(hash[0], 3u);
	hash[3] = INIT_D + (INIT_C ^ (hash[0] & 0x77777777)) + nt_buffer[1] ; hash[3] = rotate(hash[3], 7u );
	hash[2] = INIT_C + MD4_F(hash[3], hash[0], INIT_B)   + nt_buffer[2] ; hash[2] = rotate(hash[2], 11u);
	hash[1] = INIT_B + MD4_F(hash[2], hash[3], hash[0])  + nt_buffer[3] ; hash[1] = rotate(hash[1], 19u);

	hash[0] += MD4_F(hash[1], hash[2], hash[3]) + nt_buffer[4]  ; hash[0] = rotate(hash[0], 3u );
	hash[3] += MD4_F(hash[0], hash[1], hash[2]) + nt_buffer[5]  ; hash[3] = rotate(hash[3], 7u );
	hash[2] += MD4_F(hash[3], hash[0], hash[1]) + nt_buffer[6]  ; hash[2] = rotate(hash[2], 11u);
	hash[1] += MD4_F(hash[2], hash[3], hash[0]) + nt_buffer[7]  ; hash[1] = rotate(hash[1], 19u);

	hash[0] += MD4_F(hash[1], hash[2], hash[3]) + nt_buffer[8]  ; hash[0] = rotate(hash[0], 3u );
	hash[3] += MD4_F(hash[0], hash[1], hash[2]) + nt_buffer[9]  ; hash[3] = rotate(hash[3], 7u );
	hash[2] += MD4_F(hash[3], hash[0], hash[1]) + nt_buffer[10] ; hash[2] = rotate(hash[2], 11u);
	hash[1] += MD4_F(hash[2], hash[3], hash[0]) + nt_buffer[11] ; hash[1] = rotate(hash[1], 19u);

	hash[0] += MD4_F(hash[1], hash[2], hash[3]) + nt_buffer[12] ; hash[0] = rotate(hash[0], 3u );
	hash[3] += MD4_F(hash[0], hash[1], hash[2]) + nt_buffer[13] ; hash[3] = rotate(hash[3], 7u );
#if PLAINTEXT_LENGTH > 27
	hash[2] += MD4_F(hash[3], hash[0], hash[1]) + nt_buffer[14] ; hash[2] = rotate(hash[2], 11u);
	hash[1] += MD4_F(hash[2], hash[3], hash[0]) + nt_buffer[15] ; hash[1] = rotate(hash[1], 19u);
#else
	hash[2] += MD4_F(hash[3], hash[0], hash[1]) + md4_size      ; hash[2] = rotate(hash[2], 11u);
	hash[1] += MD4_F(hash[2], hash[3], hash[0])                 ; hash[1] = rotate(hash[1], 19u);
#endif

	MD4_G_CACHE_NT

	/* Round 2 */
	hash[0] += MD4_G(hash[1], hash[2], hash[3]) + nt_buffer[0]  + SQRT_2; hash[0] = rotate(hash[0], 3u );
	hash[3] += MD4_G(hash[0], hash[1], hash[2]) + nt_buffer[4]  + SQRT_2; hash[3] = rotate(hash[3], 5u );
	hash[2] += MD4_G(hash[3], hash[0], hash[1]) + nt_buffer[8]  + SQRT_2; hash[2] = rotate(hash[2], 9u );
	hash[1] += MD4_G(hash[2], hash[3], hash[0]) + nt_buffer[12] + SQRT_2; hash[1] = rotate(hash[1], 13u);

	hash[0] += MD4_G(hash[1], hash[2], hash[3]) + nt_buffer[1]  + SQRT_2; hash[0] = rotate(hash[0], 3u );
	hash[3] += MD4_G(hash[0], hash[1], hash[2]) + nt_buffer[5]  + SQRT_2; hash[3] = rotate(hash[3], 5u );
	hash[2] += MD4_G(hash[3], hash[0], hash[1]) + nt_buffer[9]  + SQRT_2; hash[2] = rotate(hash[2], 9u );
	hash[1] += MD4_G(hash[2], hash[3], hash[0]) + nt_buffer[13] + SQRT_2; hash[1] = rotate(hash[1], 13u);

	hash[0] += MD4_G(hash[1], hash[2], hash[3]) + nt_buffer[2]  + SQRT_2; hash[0] = rotate(hash[0], 3u );
	hash[3] += MD4_G(hash[0], hash[1], hash[2]) + nt_buffer[6]  + SQRT_2; hash[3] = rotate(hash[3], 5u );
	hash[2] += MD4_G(hash[3], hash[0], hash[1]) + nt_buffer[10] + SQRT_2; hash[2] = rotate(hash[2], 9u );
#if PLAINTEXT_LENGTH > 27
	hash[1] += MD4_G(hash[2], hash[3], hash[0]) + nt_buffer[14] + SQRT_2; hash[1] = rotate(hash[1], 13u);
#else
	hash[1] += MD4_G(hash[2], hash[3], hash[0]) + md4_size      + SQRT_2; hash[1] = rotate(hash[1], 13u);
#endif
	hash[0] += MD4_G(hash[1], hash[2], hash[3]) + nt_buffer[3]  + SQRT_2; hash[0] = rotate(hash[0], 3u );
	hash[3] += MD4_G(hash[0], hash[1], hash[2]) + nt_buffer[7]  + SQRT_2; hash[3] = rotate(hash[3], 5u );
	hash[2] += MD4_G(hash[3], hash[0], hash[1]) + nt_buffer[11] + SQRT_2; hash[2] = rotate(hash[2], 9u );
#if PLAINTEXT_LENGTH > 27
	hash[1] += MD4_G(hash[2], hash[3], hash[0]) + nt_buffer[15] + SQRT_2; hash[1] = rotate(hash[1], 13u);
#else
	hash[1] += MD4_G(hash[2], hash[3], hash[0])                 + SQRT_2; hash[1] = rotate(hash[1], 13u);
#endif

	/* Round 3 */
	hash[0] += MD4_H (hash[1], hash[2], hash[3]) + nt_buffer[0]  + SQRT_3; hash[0] = rotate(hash[0], 3u );
	hash[3] += MD4_H2(hash[0], hash[1], hash[2]) + nt_buffer[8]  + SQRT_3; hash[3] = rotate(hash[3], 9u );
	hash[2] += MD4_H (hash[3], hash[0], hash[1]) + nt_buffer[4]  + SQRT_3; hash[2] = rotate(hash[2], 11u);
	hash[1] += MD4_H2(hash[2], hash[3], hash[0]) + nt_buffer[12] + SQRT_3; hash[1] = rotate(hash[1], 15u);

	hash[0] += MD4_H (hash[1], hash[2], hash[3]) + nt_buffer[2]  + SQRT_3; hash[0] = rotate(hash[0], 3u );
	hash[3] += MD4_H2(hash[0], hash[1], hash[2]) + nt_buffer[10] + SQRT_3; hash[3] = rotate(hash[3], 9u );
	hash[2] += MD4_H (hash[3], hash[0], hash[1]) + nt_buffer[6]  + SQRT_3; hash[2] = rotate(hash[2], 11u);
#if PLAINTEXT_LENGTH > 27
	hash[1] += MD4_H2(hash[2], hash[3], hash[0]) + nt_buffer[14] + SQRT_3; hash[1] = rotate(hash[1], 15u);
#else
	hash[1] += MD4_H2(hash[2], hash[3], hash[0]) + md4_size      + SQRT_3; hash[1] = rotate(hash[1], 15u);
#endif
	hash[0] += MD4_H (hash[1], hash[2], hash[3]) + nt_buffer[1]  + SQRT_3; hash[0] = rotate(hash[0], 3u );
	hash[3] += MD4_H2(hash[0], hash[1], hash[2]) + nt_buffer[9]  + SQRT_3; hash[3] = rotate(hash[3], 9u );
	hash[2] += MD4_H (hash[3], hash[0], hash[1]) + nt_buffer[5]  + SQRT_3; hash[2] = rotate(hash[2], 11u);
	hash[1] += MD4_H2(hash[2], hash[3], hash[0]) + nt_buffer[13];

#if EARLY_REJECT && PLAINTEXT_LENGTH <= 27
	uint bitmap_index = hash[1] & BITMAP_MASK;
	uint tmp = (bitmaps[BITMAP_SHIFT * 0 + (bitmap_index >> 5)] >> (bitmap_index & 31)) & 1U;
#if SELECT_CMP_STEPS == 8
	bitmap_index = (hash[1] >> 8) & BITMAP_MASK;
	tmp &= (bitmaps[BITMAP_SHIFT * 1 + (bitmap_index >> 5)] >> (bitmap_index & 31)) & 1U;
	bitmap_index = (hash[1] >> 16) & BITMAP_MASK;
	tmp &= (bitmaps[BITMAP_SHIFT * 2 + (bitmap_index >> 5)] >> (bitmap_index & 31)) & 1U;
	bitmap_index = (hash[1] >> 24) & BITMAP_MASK;
	tmp &= (bitmaps[BITMAP_SHIFT * 3 + (bitmap_index >> 5)] >> (bitmap_index & 31)) & 1U;
#elif SELECT_CMP_STEPS == 4
	bitmap_index = (hash[1] >> 16) & BITMAP_MASK;
	tmp &= (bitmaps[BITMAP_SHIFT * 1 + (bitmap_index >> 5)] >> (bitmap_index & 31)) & 1U;
#endif	/* SELECT_CMP_STEPS == 8 */
	if (likely(!tmp))
		return 0;
#endif	/* EARLY_REJECT && PLAINTEXT_LENGTH <= 27 */

	uint hash1 = hash[1] + SQRT_3; hash1 = rotate(hash1, 15u);

	hash[0] += MD4_H (hash[3], hash[2], hash1  ) + nt_buffer[3]  + SQRT_3; hash[0] = rotate(hash[0], 3u );

#if PLAINTEXT_LENGTH > 27
	if (likely(md4_size <= (27 << 4)))
		return 1;

	/*
	 * Complete the first of a multi-block MD4 (reversing steps not possible).
	 */
	hash[3] +=        MD4_H2(hash[2], hash1,   hash[0]) + nt_buffer[11] + SQRT_3; hash[3] = rotate(hash[3], 9u );
	hash[2] +=        MD4_H (hash1,   hash[0], hash[3]) + nt_buffer[7]  + SQRT_3; hash[2] = rotate(hash[2], 11u);
	hash[1] = hash1 + MD4_H2(hash[2], hash[3], hash[0]) + nt_buffer[15] + SQRT_3; hash[1] = rotate(hash[1], 15u);
	hash[0] += INIT_A;
	hash[1] += INIT_B;
	hash[2] += INIT_C;
	hash[3] += INIT_D;

#if PLAINTEXT_LENGTH > 59
	uint blocks = ((md4_size >> 4) + 5 + 31) / 32;
	while (--blocks)
#endif
	{
		nt_buffer += 16;

		uint a = hash[0];
		uint b = hash[1];
		uint c = hash[2];
		uint d = hash[3];

		hash[0] += MD4_F(hash[1], hash[2], hash[3]) + nt_buffer[0]  ; hash[0] = rotate(hash[0], 3u );
		hash[3] += MD4_F(hash[0], hash[1], hash[2]) + nt_buffer[1]  ; hash[3] = rotate(hash[3], 7u );
		hash[2] += MD4_F(hash[3], hash[0], hash[1]) + nt_buffer[2]  ; hash[2] = rotate(hash[2], 11u);
		hash[1] += MD4_F(hash[2], hash[3], hash[0]) + nt_buffer[3]  ; hash[1] = rotate(hash[1], 19u);

		hash[0] += MD4_F(hash[1], hash[2], hash[3]) + nt_buffer[4]  ; hash[0] = rotate(hash[0], 3u );
		hash[3] += MD4_F(hash[0], hash[1], hash[2]) + nt_buffer[5]  ; hash[3] = rotate(hash[3], 7u );
		hash[2] += MD4_F(hash[3], hash[0], hash[1]) + nt_buffer[6]  ; hash[2] = rotate(hash[2], 11u);
		hash[1] += MD4_F(hash[2], hash[3], hash[0]) + nt_buffer[7]  ; hash[1] = rotate(hash[1], 19u);

		hash[0] += MD4_F(hash[1], hash[2], hash[3]) + nt_buffer[8]  ; hash[0] = rotate(hash[0], 3u );
		hash[3] += MD4_F(hash[0], hash[1], hash[2]) + nt_buffer[9]  ; hash[3] = rotate(hash[3], 7u );
		hash[2] += MD4_F(hash[3], hash[0], hash[1]) + nt_buffer[10] ; hash[2] = rotate(hash[2], 11u);
		hash[1] += MD4_F(hash[2], hash[3], hash[0]) + nt_buffer[11] ; hash[1] = rotate(hash[1], 19u);

		hash[0] += MD4_F(hash[1], hash[2], hash[3]) + nt_buffer[12] ; hash[0] = rotate(hash[0], 3u );
		hash[3] += MD4_F(hash[0], hash[1], hash[2]) + nt_buffer[13] ; hash[3] = rotate(hash[3], 7u );
		hash[2] += MD4_F(hash[3], hash[0], hash[1]) + nt_buffer[14] ; hash[2] = rotate(hash[2], 11u);
		hash[1] += MD4_F(hash[2], hash[3], hash[0]) + nt_buffer[15] ; hash[1] = rotate(hash[1], 19u);

		MD4_G_CACHE_NT

		/* Round 2 */
		hash[0] += MD4_G(hash[1], hash[2], hash[3]) + nt_buffer[0]  + SQRT_2; hash[0] = rotate(hash[0], 3u );
		hash[3] += MD4_G(hash[0], hash[1], hash[2]) + nt_buffer[4]  + SQRT_2; hash[3] = rotate(hash[3], 5u );
		hash[2] += MD4_G(hash[3], hash[0], hash[1]) + nt_buffer[8]  + SQRT_2; hash[2] = rotate(hash[2], 9u );
		hash[1] += MD4_G(hash[2], hash[3], hash[0]) + nt_buffer[12] + SQRT_2; hash[1] = rotate(hash[1], 13u);

		hash[0] += MD4_G(hash[1], hash[2], hash[3]) + nt_buffer[1]  + SQRT_2; hash[0] = rotate(hash[0], 3u );
		hash[3] += MD4_G(hash[0], hash[1], hash[2]) + nt_buffer[5]  + SQRT_2; hash[3] = rotate(hash[3], 5u );
		hash[2] += MD4_G(hash[3], hash[0], hash[1]) + nt_buffer[9]  + SQRT_2; hash[2] = rotate(hash[2], 9u );
		hash[1] += MD4_G(hash[2], hash[3], hash[0]) + nt_buffer[13] + SQRT_2; hash[1] = rotate(hash[1], 13u);

		hash[0] += MD4_G(hash[1], hash[2], hash[3]) + nt_buffer[2]  + SQRT_2; hash[0] = rotate(hash[0], 3u );
		hash[3] += MD4_G(hash[0], hash[1], hash[2]) + nt_buffer[6]  + SQRT_2; hash[3] = rotate(hash[3], 5u );
		hash[2] += MD4_G(hash[3], hash[0], hash[1]) + nt_buffer[10] + SQRT_2; hash[2] = rotate(hash[2], 9u );
		hash[1] += MD4_G(hash[2], hash[3], hash[0]) + nt_buffer[14] + SQRT_2; hash[1] = rotate(hash[1], 13u);

		hash[0] += MD4_G(hash[1], hash[2], hash[3]) + nt_buffer[3]  + SQRT_2; hash[0] = rotate(hash[0], 3u );
		hash[3] += MD4_G(hash[0], hash[1], hash[2]) + nt_buffer[7]  + SQRT_2; hash[3] = rotate(hash[3], 5u );
		hash[2] += MD4_G(hash[3], hash[0], hash[1]) + nt_buffer[11] + SQRT_2; hash[2] = rotate(hash[2], 9u );
		hash[1] += MD4_G(hash[2], hash[3], hash[0]) + nt_buffer[15] + SQRT_2; hash[1] = rotate(hash[1], 13u);

		/* Round 3 */
		hash[0] += MD4_H (hash[1], hash[2], hash[3]) + nt_buffer[0]  + SQRT_3; hash[0] = rotate(hash[0], 3u );
		hash[3] += MD4_H2(hash[0], hash[1], hash[2]) + nt_buffer[8]  + SQRT_3; hash[3] = rotate(hash[3], 9u );
		hash[2] += MD4_H (hash[3], hash[0], hash[1]) + nt_buffer[4]  + SQRT_3; hash[2] = rotate(hash[2], 11u);
		hash[1] += MD4_H2(hash[2], hash[3], hash[0]) + nt_buffer[12] + SQRT_3; hash[1] = rotate(hash[1], 15u);

		hash[0] += MD4_H (hash[1], hash[2], hash[3]) + nt_buffer[2]  + SQRT_3; hash[0] = rotate(hash[0], 3u );
		hash[3] += MD4_H2(hash[0], hash[1], hash[2]) + nt_buffer[10] + SQRT_3; hash[3] = rotate(hash[3], 9u );
		hash[2] += MD4_H (hash[3], hash[0], hash[1]) + nt_buffer[6]  + SQRT_3; hash[2] = rotate(hash[2], 11u);
		hash[1] += MD4_H2(hash[2], hash[3], hash[0]) + nt_buffer[14] + SQRT_3; hash[1] = rotate(hash[1], 15u);

		hash[0] += MD4_H (hash[1], hash[2], hash[3]) + nt_buffer[1]  + SQRT_3; hash[0] = rotate(hash[0], 3u );
		hash[3] += MD4_H2(hash[0], hash[1], hash[2]) + nt_buffer[9]  + SQRT_3; hash[3] = rotate(hash[3], 9u );
		hash[2] += MD4_H (hash[3], hash[0], hash[1]) + nt_buffer[5]  + SQRT_3; hash[2] = rotate(hash[2], 11u);
		hash[1] += MD4_H2(hash[2], hash[3], hash[0]) + nt_buffer[13] + SQRT_3; hash[1] = rotate(hash[1], 15u);

		hash[0] += MD4_H (hash[3], hash[2], hash[1]) + nt_buffer[3]  + SQRT_3; hash[0] = rotate(hash[0], 3u );
		hash[3] += MD4_H2(hash[2], hash[1], hash[0]) + nt_buffer[11] + SQRT_3; hash[3] = rotate(hash[3], 9u );
		hash[2] += MD4_H (hash[1], hash[0], hash[3]) + nt_buffer[7]  + SQRT_3; hash[2] = rotate(hash[2], 11u);
		hash[1] += MD4_H2(hash[2], hash[3], hash[0]) + nt_buffer[15] + SQRT_3; hash[1] = rotate(hash[1], 15u);

		hash[0] += a;
		hash[1] += b;
		hash[2] += c;
		hash[3] += d;
	}

	/*
	 * This bogus reverse adds a little work to long crypts instead
	 * of losing the real reverse for single block crypts.
	 */
	hash[3] -= INIT_D;
	hash[2] -= INIT_C;
	hash[1] -= INIT_B;
	hash[0] -= INIT_A;
	hash[1]  = (hash[1] >> 15) | (hash[1] << 17);
	hash[1] -= SQRT_3 + MD4_H2(hash[2], hash[3], hash[0]);
	hash[1]  = rotate(hash[1], -15u);
	hash[1] -= SQRT_3;
#endif
	return 1;
}

#if __OS_X__ && (cpu(DEVICE_INFO) || gpu_nvidia(DEVICE_INFO))
/* This is a workaround for driver/runtime bugs */
#define MAYBE_VOLATILE volatile
#else
#define MAYBE_VOLATILE
#endif

#if UTF_8

INLINE uint prepare_key(__global uint *key, uint length,
                        MAYBE_VOLATILE uint *nt_buffer)
{
	const __global UTF8 *source = (const __global UTF8*)key;
	const __global UTF8 *sourceEnd = &source[length];
	MAYBE_VOLATILE UTF16 *target = (UTF16*)nt_buffer;
	MAYBE_VOLATILE const UTF16 *targetEnd = &target[PLAINTEXT_LENGTH];
	UTF32 ch;
	uint extraBytesToRead;

	/* Input buffer is UTF-8 without zero-termination */
	while (source < sourceEnd) {
		if (*source < 0xC0) {
			*target++ = (UTF16)*source++;
			if (target >= targetEnd)
				break;
			continue;
		}
		ch = *source;
		// This point must not be reached with *source < 0xC0
		extraBytesToRead =
			opt_trailingBytesUTF8[ch & 0x3f];
		if (source + extraBytesToRead >= sourceEnd) {
			break;
		}
		switch (extraBytesToRead) {
		case 3:
			ch <<= 6;
			ch += *++source;
		case 2:
			ch <<= 6;
			ch += *++source;
		case 1:
			ch <<= 6;
			ch += *++source;
			++source;
			break;
		default:
			*target = UNI_REPLACEMENT_CHAR;
			break; // from switch
		}
		if (*target == UNI_REPLACEMENT_CHAR)
			break; // from while
		ch -= offsetsFromUTF8[extraBytesToRead];
#ifdef UCS_2
		/* UCS-2 only */
		*target++ = (UTF16)ch;
#else
		/* full UTF-16 with surrogate pairs */
		if (ch <= UNI_MAX_BMP) {  /* Target is a character <= 0xFFFF */
			*target++ = (UTF16)ch;
		} else {  /* target is a character in range 0xFFFF - 0x10FFFF. */
			if (target + 1 >= targetEnd)
				break;
			ch -= halfBase;
			*target++ = (UTF16)((ch >> halfShift) + UNI_SUR_HIGH_START);
			*target++ = (UTF16)((ch & halfMask) + UNI_SUR_LOW_START);
		}
#endif
		if (target >= targetEnd)
			break;
	}
	*target = 0x80;	// Terminate

	return (uint)(target - (UTF16*)nt_buffer);
}

#else

INLINE uint prepare_key(__global uint *key, uint length, uint *nt_buffer)
{
	uint i, nt_index, keychars;

	nt_index = 0;
	for (i = 0; i < (length + 3)/ 4; i++) {
		keychars = key[i];
		nt_buffer[nt_index++] = CP_LUT(keychars & 0x000000FF) | (CP_LUT((keychars & 0x0000FF00) >> 8) << 16);
		nt_buffer[nt_index++] = CP_LUT((keychars & 0x00FF0000) >> 16) | (CP_LUT(keychars >> 24) << 16);
	}
	nt_index = length >> 1;
	nt_buffer[nt_index] = (nt_buffer[nt_index] & 0xFFFF) | (0x80 << ((length & 1) << 4));

	return length;
}

#endif /* UTF_8 */

INLINE void cmp_final(uint gid,
                      uint iter,
                      uint *hash,
                      __global uint *offset_table,
                      __global uint *hash_table,
                      volatile __global uint *output,
                      volatile __global uint *bitmap_dupe)
{
	uint t, hash_table_index;
	ulong hash64;

	hash64 = ((ulong)hash[1] << 32) | (ulong)hash[0];
	hash64 += (ulong)offset_table[hash64 % OFFSET_TABLE_SIZE];
	hash_table_index = hash64 % HASH_TABLE_SIZE;

	if (hash_table[hash_table_index] == hash[0] &&
	    hash_table[hash_table_index + HASH_TABLE_SIZE] == hash[1]) {
		/*
		 * Prevent duplicate keys from cracking same hash
		 */
		if (!(atomic_or(&bitmap_dupe[hash_table_index / 32],
		                (1U << (hash_table_index % 32))) & (1U << (hash_table_index % 32)))) {
			t = atomic_inc(&output[0]);
			output[3 * t + 1] = gid;
			output[3 * t + 2] = iter;
			output[3 * t + 3] = hash_table_index;
		}
	}
}

INLINE void cmp(uint gid,
                uint iter,
                uint *hash,
                BITMAPS_TYPE uint *bitmaps,
                __global uint *offset_table,
                __global uint *hash_table,
                volatile __global uint *output,
                volatile __global uint *bitmap_dupe)
{
	uint bitmap_index, tmp = 1;

#if SELECT_CMP_STEPS == 8
#if !EARLY_REJECT || PLAINTEXT_LENGTH > 27
	bitmap_index = hash[1] & BITMAP_MASK;
	tmp &= (bitmaps[BITMAP_SHIFT * 0 + (bitmap_index >> 5)] >> (bitmap_index & 31)) & 1U;
	bitmap_index = (hash[1] >> 8) & BITMAP_MASK;
	tmp &= (bitmaps[BITMAP_SHIFT * 1 + (bitmap_index >> 5)] >> (bitmap_index & 31)) & 1U;
	bitmap_index = (hash[1] >> 16) & BITMAP_MASK;
	tmp &= (bitmaps[BITMAP_SHIFT * 2 + (bitmap_index >> 5)] >> (bitmap_index & 31)) & 1U;
	bitmap_index = (hash[1] >> 24) & BITMAP_MASK;
	tmp &= (bitmaps[BITMAP_SHIFT * 3 + (bitmap_index >> 5)] >> (bitmap_index & 31)) & 1U;
#endif
	bitmap_index = hash[0] & BITMAP_MASK;
	tmp &= (bitmaps[BITMAP_SHIFT * 4 + (bitmap_index >> 5)] >> (bitmap_index & 31)) & 1U;
	bitmap_index = (hash[0] >> 8) & BITMAP_MASK;
	tmp &= (bitmaps[BITMAP_SHIFT * 5 + (bitmap_index >> 5)] >> (bitmap_index & 31)) & 1U;
	bitmap_index = (hash[0] >> 16) & BITMAP_MASK;
	tmp &= (bitmaps[BITMAP_SHIFT * 6 + (bitmap_index >> 5)] >> (bitmap_index & 31)) & 1U;
	bitmap_index = (hash[0] >> 24) & BITMAP_MASK;
	tmp &= (bitmaps[BITMAP_SHIFT * 7 + (bitmap_index >> 5)] >> (bitmap_index & 31)) & 1U;

#elif SELECT_CMP_STEPS == 4
#if !EARLY_REJECT || PLAINTEXT_LENGTH > 27
	bitmap_index = hash[1] & BITMAP_MASK;
	tmp &= (bitmaps[BITMAP_SHIFT * 0 + (bitmap_index >> 5)] >> (bitmap_index & 31)) & 1U;
	bitmap_index = (hash[1] >> 16) & BITMAP_MASK;
	tmp &= (bitmaps[BITMAP_SHIFT * 1 + (bitmap_index >> 5)] >> (bitmap_index & 31)) & 1U;
#endif
	bitmap_index = hash[0] & BITMAP_MASK;
	tmp &= (bitmaps[BITMAP_SHIFT * 2 + (bitmap_index >> 5)] >> (bitmap_index & 31)) & 1U;
	bitmap_index = (hash[0] >> 16) & BITMAP_MASK;
	tmp &= (bitmaps[BITMAP_SHIFT * 3 + (bitmap_index >> 5)] >> (bitmap_index & 31)) & 1U;

#elif SELECT_CMP_STEPS == 2
#if !EARLY_REJECT || PLAINTEXT_LENGTH > 27
	bitmap_index = hash[1] & BITMAP_MASK;
	tmp &= (bitmaps[BITMAP_SHIFT * 0 + (bitmap_index >> 5)] >> (bitmap_index & 31)) & 1U;
#endif
	bitmap_index = hash[0] & BITMAP_MASK;
	tmp &= (bitmaps[BITMAP_SHIFT * 1 + (bitmap_index >> 5)] >> (bitmap_index & 31)) & 1U;

#elif !EARLY_REJECT || PLAINTEXT_LENGTH > 27 /* SELECT_CMP_STEPS == 1 */
	bitmap_index = hash[1] & BITMAP_MASK;
	tmp &= (bitmaps[BITMAP_SHIFT * 0 + (bitmap_index >> 5)] >> (bitmap_index & 31)) & 1U;

#endif	/* SELECT_CMP_STEPS == 8 */

	if (tmp)
		cmp_final(gid, iter, hash, offset_table, hash_table, output, bitmap_dupe);
}

/*
 * OpenCL kernel entry point. Break the key into 16 32-bit (uint)
 * words. MD4 hash of a key is 128 bits but we only do 64 bits here, and
 * reverse steps where possible.
 */
__kernel void nt(__global uint *keys,
                 __global uint *index,
                 __global uint *int_key_loc,
                 CACHE_TYPE uint *int_keys,
                 __global uint *bitmaps,
                 __global uint *offset_table,
                 __global uint *hash_table,
                 volatile __global uint *out_hash_ids,
                 volatile __global uint *bitmap_dupe)
{
	uint i;
	uint gid = get_global_id(0);
	uint base = index[gid];
	uint nt_buffer[(PLAINTEXT_LENGTH + 5 + 31) / 32 * 16] = { 0 };
	uint md4_size = base & 127;
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

#define BITMAPS	s_bitmaps
#else
#define BITMAPS	bitmaps
#endif

	keys += base >> 7;
	md4_size = prepare_key(keys, md4_size, nt_buffer);

	/* Put the length word in the correct place in buffer, outside the loop */
	uint size_idx = ((md4_size + 5 + 31) / 32 - 1) * 16 + 14;
	md4_size <<= 4;
	nt_buffer[size_idx] = md4_size;

	for (i = 0; i < NUM_INT_KEYS; i++) {
#if NUM_INT_KEYS > 1
		PUTSHORT(nt_buffer, GPU_LOC_0, CP_LUT(int_keys[i] & 0xff));
#if MASK_FMT_INT_PLHDR > 1
#if LOC_1 >= 0
		PUTSHORT(nt_buffer, GPU_LOC_1, CP_LUT((int_keys[i] & 0xff00) >> 8));
#endif
#endif
#if MASK_FMT_INT_PLHDR > 2
#if LOC_2 >= 0
		PUTSHORT(nt_buffer, GPU_LOC_2, CP_LUT((int_keys[i] & 0xff0000) >> 16));
#endif
#endif
#if MASK_FMT_INT_PLHDR > 3
#if LOC_3 >= 0
		PUTSHORT(nt_buffer, GPU_LOC_3, CP_LUT((int_keys[i] & 0xff000000) >> 24));
#endif
#endif
#endif
		if (nt_crypt(hash, nt_buffer, md4_size, BITMAPS))
			cmp(gid, i, hash, BITMAPS, offset_table, hash_table, out_hash_ids, bitmap_dupe);
	}
}

#ifdef GPU_GEN
/*
 * K-ordered GPU password generator for NTLM (single MD4 block, len <= 27).
 *
 * Direct port of md5_kernel.cl:md5_gen - the segment location, simplex unrank
 * and Markov materialization are hash-independent and bit-exact mirrors of
 * mask.c:mask_gpu_unrank_key()/simplex_next_state(). Two things differ from the
 * MD5 generator:
 *   1. Args. NTLM uses the 64-bit hash check (opencl_hash_check_64), which has
 *      no return_hashes buffer, so the hash-check args occupy slots 0..8 and the
 *      generator inputs start at arg 9 (MD5/128 starts at 10).
 *   2. Message packing. Instead of packing 4 raw key[] bytes per uint into W[],
 *      we build the UTF-16 nt_buffer: two CP_LUT-encoded chars per uint (mirror
 *      of prepare_key's codepage path), drop the 0x80 terminator at char index
 *      len, and place the bit length at nt_buffer[14]. key[] stays raw bytes so
 *      the Markov key[kp-1] dependency is unaffected.
 *
 * See md5_gen for the full commentary on the per-segment template, the
 * register-resident (GEN_REGS) variant and the virtual index space.
 */
#ifndef GEN_MAX_POS
#define GEN_MAX_POS 64
#endif
#ifndef GEN_NDW
#define GEN_NDW 14
#endif
#ifndef GEN_REG_MAX
#define GEN_REG_MAX 16
#endif

/* Device mirror of one mask_gpu_plan segment (must match gen_seg in the format). */
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

__kernel void nt_gen(__global uint *keys_unused,
		  __global uint *index_unused,
		  __global uint *int_key_loc_unused,
		  __global uint *int_keys_unused,
		  __global uint *bitmaps,
		  __global uint *offset_table,
		  __global uint *hash_table,
		  volatile __global uint *out_hash_ids,
		  volatile __global uint *bitmap_dupe,
		  __global ulong *suf,
		  __global uint *g_table_packed,
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
		  uint gR
#ifdef GEN_ODOMETER
		, __constant uchar *g_boxbounds  /* per-seg lo[limit]+radix[limit] */
#endif
		  )
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
		uint nt_buffer[(PLAINTEXT_LENGTH + 5 + 31) / 32 * 16] = { 0 };
		uint hash[4];
		uchar key[GEN_MAX_POS];
#ifdef GEN_REGS
		uchar iter[GEN_REG_MAX];
#else
		uchar iter[GEN_MAX_POS];
#endif
		ulong gl0 = (ulong)gid * gR;

		uint cur_seg = 0xffffffff;
		ulong seg_vbase = 0, seg_vend = 0, seg_lstart = 0, seg_suf_off = 0;
		uint glimit = 0, glen = 0, gmax_k = 0, gksize = 0;
		uint len = 0, cur_k = 0, md4_size = 0;

		for (j = 0; j < gR; j++) {
			ulong gl = gl0 + j;
			ulong V, g;
			uint mfrom;
			int full;

			if (gl >= gcount)
				break;
			V = gbase + gl;

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

				/* Rebuild the raw-byte key[] template for this length: literals
				 * where defined, zero padding elsewhere. No 0x80 goes into key[]
				 * (the NTLM terminator lives in the UTF-16 domain and is applied
				 * to nt_buffer below). nt_buffer[14] holds the single-block bit
				 * length, set once per segment. */
				len = glen;
#pragma unroll
				for (i = 0; i < GEN_MAX_POS; i++)
					key[i] = (i < len) ? g_littmpl[i] : 0;
				md4_size = len << 4;
				nt_buffer[14] = md4_size;
				full = 1;
			} else {
				full = 0;
			}

			g = seg_lstart + (V - seg_vbase);

#ifdef GEN_REGS
			if (full) {
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
				int pivot = -1;
				#pragma unroll
				for (i = 0; i < GEN_REG_MAX; i++) {
					int is_valid = ((int)i + 1 < (int)glimit) && (iter[i] > 0) && (iter[i + 1] < g_count[i + 1] - 1);
					pivot = is_valid ? (int)i : pivot;
				}

				int has_pivot    = (pivot >= 0);
				int is_exhausted = !has_pivot;

				cur_k += is_exhausted;

				int w = 0;
#pragma unroll
				for (i = 0; i < GEN_REG_MAX; i++) {
					int mask_w = (i < glimit) && ((int)i >= pivot + 2);
					w += iter[i] * mask_w;
				}

				int r_weight = has_pivot ? w : (int)cur_k;

#pragma unroll
				for (i = 0; i < GEN_REG_MAX; i++) {
					int cond_glimit = (i < glimit);
					int lookup_idx  = cond_glimit ? i : 0;

					int is_before_pivot = has_pivot && ((int)i < pivot);
					int is_pivot        = has_pivot && ((int)i == pivot);
					int is_after_pivot  = is_exhausted || ((int)i > pivot);

					int val_before = iter[i];
					int val_pivot  = iter[i] - 1;

					int base = (has_pivot && ((int)i == pivot + 1)) ? (iter[i] + 1) : 0;
					int cap  = g_count[lookup_idx] - 1 - base;

					int add       = min(r_weight, cap);
					int val_after = base + add;

					r_weight -= add * (cond_glimit && is_after_pivot);

					int next_val = (is_before_pivot * val_before) +
					               (is_pivot        * val_pivot)  +
					               (is_after_pivot  * val_after);

					iter[i] = cond_glimit ? (uchar)next_val : iter[i];
				}

				mfrom = has_pivot ? (uint)pivot : 0;
			}

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
#ifdef GEN_ODOMETER
			/* Odometer-shell decode: this segment is a magnitude sub-box with
			 * per-position [lo, radix) rank bounds at (g_boxbounds + seg_suf_off):
			 * lo[0..glimit) then radix[0..glimit). Mixed-radix decode the loop-
			 * local index g into iter[], rightmost position least significant
			 * (matches the host mirror), then materialize. Divergence-free; every
			 * candidate is full-decoded so mfrom = 0. */
			{
				__constant uchar *blo  = g_boxbounds + seg_suf_off;
				__constant uchar *brad = blo + glimit;
				int p;

				if (full) {
					/* New sub-box: full mixed-radix decode of the local index g,
					 * rightmost position least significant. */
					ulong gloc = g;

					for (p = (int)glimit - 1; p >= 0; p--) {
						uint rad = brad[p];
						iter[p] = (uchar)(blo[p] + (uint)(gloc % rad));
						gloc /= rad;
					}
					mfrom = 0;
				} else {
					/* Same sub-box, next candidate: odometer +1 with carry. Only
					 * positions from the carried one rightward changed, so mfrom is
					 * that position - the materialize re-runs just key[mfrom..glimit)
					 * (usually one position). The carry is short on average
					 * (geometric in the radix) and divergence-free in the common
					 * no-carry case, which is where the speedup over the simplex
					 * repack comes from. */
					p = (int)glimit - 1;
					for (;;) {
						uint lo = blo[p];
						if ((uint)iter[p] + 1u < lo + brad[p]) {
							iter[p]++;
							break;
						}
						iter[p] = (uchar)lo;   /* wrap this wheel, carry left */
						if (--p < 0) { p = 0; break; }
					}
					mfrom = (uint)p;
				}
			}
#else
			if (full) {
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
#endif /* GEN_ODOMETER */

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
						int block_idx = (i * 256 + prev) * 64 + (ti >> 2);
						uint packed_chars = g_table_packed[block_idx];
						uint shift_amount = (ti & 3) << 3;
						key[kp] = (uchar)(packed_chars >> shift_amount);
					} else {
						key[kp] = g_chars0[i];
					}
				}
			}
#endif /* GEN_REGS */

			/* Pack the UTF-16 message words with COMPILE-TIME indices (mirror of
			 * prepare_key's codepage path): two CP_LUT-encoded chars per uint.
			 * key[] padding past len is 0, so words beyond the last char come out
			 * 0; the single 0x80 terminator at char index len is OR'd in after.
			 * nt_buffer[14] (bit length) and any words >= GEN_NDW stay as set/zero
			 * across candidates. */
#pragma unroll
			for (i = 0; i < GEN_NDW; i++)
				nt_buffer[i] = CP_LUT(key[2 * i]) |
				               (CP_LUT(key[2 * i + 1]) << 16);
			nt_buffer[len >> 1] |= (uint)0x80 << ((len & 1) << 4);

			if (nt_crypt(hash, nt_buffer, md4_size,
#if USE_LOCAL_BITMAPS
			    s_bitmaps
#else
			    bitmaps
#endif
			    ))
				cmp(gid, j, hash,
#if USE_LOCAL_BITMAPS
				    s_bitmaps
#else
				    bitmaps
#endif
				    , offset_table, hash_table, out_hash_ids, bitmap_dupe);
		}
	}
}
#endif /* GPU_GEN */
