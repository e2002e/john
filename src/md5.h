/*
 * This is an OpenSSL-compatible implementation of the RSA Data Security, Inc.
 * MD5 Message-Digest Algorithm (RFC 1321).
 *
 * Homepage:
 * https://openwall.info/wiki/people/solar/software/public-domain-source-code/md5
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
 * See md5.c for more information.
 */

/*
 * The code below has been modified in many ways for use specifically in John
 * the Ripper jumbo.  If you want to reuse it outside of that project, please
 * obtain the original from the homepage URL above.
 */

#if !defined(_MD5_H)
#define _MD5_H

#include <stdint.h>

/* Any 32-bit or wider unsigned integer data type will do */
/* this needs to be defined no matter if building with HAVE_LIBCRYPTO or not */
typedef unsigned int MD5_u32plus;

#include "arch.h" /* also includes autoconfig.h for HAVE_LIBCRYPTO */

#if HAVE_LIBCRYPTO
#include <openssl/md5.h>

#else

#define MD5_Init john_MD5_Init
#define MD5_Update john_MD5_Update
#define MD5_Final john_MD5_Final

typedef struct {
	MD5_u32plus A, B, C, D;
	MD5_u32plus lo, hi;
	unsigned char buffer[64];
#if !(ARCH_LITTLE_ENDIAN && ARCH_ALLOWS_UNALIGNED)
	MD5_u32plus block[16];
#endif
} MD5_CTX;

extern void MD5_Init(MD5_CTX *ctx);
extern void MD5_Update(MD5_CTX *ctx, const void *data, unsigned long size);
extern void MD5_PreFinal(MD5_CTX *ctx);
extern void MD5_Final(unsigned char *result, MD5_CTX *ctx);

#endif /* HAVE_LIBCRYPTO */

extern void md5_reverse(uint32_t *hash);
extern void md5_unreverse(uint32_t *hash);

#endif /* _MD5_H */
