/*
 * This is an OpenSSL-compatible implementation of the RSA Data Security, Inc.
 * MD4 Message-Digest Algorithm (RFC 1320).
 *
 * Homepage:
 * https://openwall.info/wiki/people/solar/software/public-domain-source-code/md4
 *
 * Author:
 * Alexander Peslyak, better known as Solar Designer <solar at openwall.com>
 *
 * This software was written by Alexander Peslyak in 2001.  No copyright is
 * claimed, and the software is hereby placed in the public domain.
 * In case this attempt to disclaim copyright and place the software in the
 * public domain is deemed null and void, then the software is
 * Copyright (c) 2001 Alexander Peslyak and it is hereby released to the
 * general public under the following terms:
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted.
 *
 * There's ABSOLUTELY NO WARRANTY, express or implied.
 *
 * See md4.c for more information.
 */

/*
 * The code below has been modified in many ways for use specifically in John
 * the Ripper jumbo.  If you want to reuse it outside of that project, please
 * obtain the original from the homepage URL above.
 */

#if !defined(_MD4_H)
#define _MD4_H

#include <stdint.h>

#include "arch.h" /* also includes autoconfig.h for HAVE_LIBCRYPTO */

#if HAVE_LIBCRYPTO
#include <openssl/md4.h>

#else

#define MD4_Init john_MD4_Init
#define MD4_Update john_MD4_Update
#define MD4_Final john_MD4_Final

/* Any 32-bit or wider unsigned integer data type will do */
typedef unsigned int MD4_u32plus;

typedef struct {
	MD4_u32plus A, B, C, D;
	MD4_u32plus lo, hi;
	unsigned char buffer[64];
#if !(ARCH_LITTLE_ENDIAN && ARCH_ALLOWS_UNALIGNED)
	MD4_u32plus block[16];
#endif
} MD4_CTX;

extern void MD4_Init(MD4_CTX *ctx);
extern void MD4_Update(MD4_CTX *ctx, const void *data, unsigned long size);
extern void MD4_Final(unsigned char *result, MD4_CTX *ctx);

#endif /* HAVE_LIBCRYPTO */

extern void md4_reverse(uint32_t *hash);
extern void md4_unreverse(uint32_t *hash);

#endif /* _MD4_H */
