/*
 * This software is Copyright (c) 2024 magnum, and it is hereby released to
 * the general public under the following terms:
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted.
 */

#ifndef JTR_AES_H
#define JTR_AES_H

#include "mbedtls/aes.h"

typedef mbedtls_aes_context AES_KEY;

#define AES_ENCRYPT     MBEDTLS_AES_ENCRYPT
#define AES_DECRYPT     MBEDTLS_AES_DECRYPT
#define AES_BLOCK_SIZE  16

#define AES_set_encrypt_key(key, size, ctx)	mbedtls_aes_setkey_enc(ctx, key, size)
#define AES_encrypt(in, out, ctx)	mbedtls_aes_crypt_ecb(ctx, MBEDTLS_AES_ENCRYPT, in, out)

#define AES_set_decrypt_key(key, size, ctx)	mbedtls_aes_setkey_dec(ctx, key, size)
#define AES_decrypt(in, out, ctx)	mbedtls_aes_crypt_ecb(ctx, MBEDTLS_AES_DECRYPT, in, out)

#define AES_ecb_encrypt(in, out, ctx, mode)	mbedtls_aes_crypt_ecb(ctx, mode, in, out)
#define AES_cbc_encrypt(in, out, len, ctx, iv, mode)	mbedtls_aes_crypt_cbc(ctx, mode, len, iv, in, out)
#define AES_cfb128_encrypt(in, out, len, ctx, iv, iv_off, mode)	mbedtls_aes_crypt_cfb128(ctx, mode, len, iv_off, iv, in, out)

#endif /* JTR_AES_H */
