/*
 * MD5 OpenCL code is based on Alain Espinosa's OpenCL patches.
 *
 * This software is
 * Copyright (c) 2010, Dhiru Kholia <dhiru.kholia at gmail.com>
 * Copyright (c) 2012-2025, magnum
 * Copyright (c) 2015, Sayantan Datta <std2048@gmail.com>
 * and it is hereby released to the general public under the following terms:
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted.
 */

#ifdef HAVE_OPENCL
#define FMT_STRUCT fmt_opencl_rawMD5

#if FMT_EXTERNS_H
extern struct fmt_main FMT_STRUCT;
#elif FMT_REGISTERS_H
john_register_one(&FMT_STRUCT);
#else

#include <string.h>
#include <sys/time.h>

#include "arch.h"
#include "params.h"
#include "path.h"
#include "common.h"
#include "formats.h"
#include "base64_convert.h"
#include "config.h"
#include "options.h"
#include "mask_ext.h"
#include "opencl_hash_check.h"

#define PLAINTEXT_LENGTH    55 /* Max. is 55 with current kernel */
#define BUFSIZE             ((PLAINTEXT_LENGTH+3)/4*4)
#define FORMAT_LABEL        "raw-MD5-opencl"
#define FORMAT_NAME         ""
#define ALGORITHM_NAME      "MD5 OpenCL"
#define BENCHMARK_COMMENT   ""
#define BENCHMARK_LENGTH    0x107
#define CIPHERTEXT_LENGTH   32
#define DIGEST_SIZE         16
#define BINARY_SIZE         16
#define BINARY_ALIGN        sizeof(uint32_t)
#define SALT_SIZE           0
#define SALT_ALIGN          1

#define FORMAT_TAG			"$dynamic_0$"
#define TAG_LENGTH			(sizeof(FORMAT_TAG) - 1)
#define FORMAT_TAG2			"{MD5}"
#define FORMAT_TAG2_LEN		(sizeof(FORMAT_TAG2) - 1)

static cl_mem pinned_saved_keys, pinned_saved_idx, pinned_int_key_loc;
static cl_mem buffer_keys, buffer_idx, buffer_int_keys, buffer_int_key_loc;
static cl_uint *saved_plain, *saved_idx, *saved_int_key_loc;
static int static_gpu_locations[MASK_FMT_INT_PLHDR];
static char (*ui_keys)[PLAINTEXT_LENGTH + 1] = NULL;

/* Static variables from opencl_hash_check_128.c needed by ocl_hc_128_extract_info */
extern cl_uint *bitmaps;
extern cl_ulong bitmap_size_bits;
extern OFFSET_TABLE_WORD *offset_table;
static unsigned int shift64_ht_sz, shift64_ot_sz;

static unsigned int key_idx = 0;
static struct fmt_main *self;

#define MAX_LIMIT 16
/* These exist inside opencl_hash_check_128.c – just declare them. */
extern cl_uint *loaded_hashes;          /* host array for return hashes */
extern cl_uint *zero_buffer;            /* zero buffer for dupe bitmap reset */
extern cl_uint *bitmaps;
extern cl_ulong bitmap_size_bits;
extern OFFSET_TABLE_WORD *offset_table;
extern unsigned int *bt_hash_table_128;
static cl_uint *my_loaded_hashes = NULL;

/* Our local handles – filled by ocl_hc_128_get_buffers() */
static cl_mem hc_bitmaps       = NULL;
static cl_mem hc_offset_table  = NULL;
static cl_mem hc_hash_table    = NULL;
static cl_mem hc_return_hashes = NULL;
static cl_mem hc_hash_ids      = NULL;
static cl_mem hc_bitmap_dupe   = NULL;

// 1:1 Mapping of your OpenCL structures
typedef struct {
    int count;
    int pos;
    unsigned char chars[256];
} opencl_mask_range;

typedef struct {
	int active_idx[MAX_LIMIT];
	int current_k;
	int limit;
} opencl_placeholder;

typedef struct {
	int num_loops_interleaved;
	int iterations_per_thread;
} opencl_mask_config;

// Host-side arrays to build the data before transferring to the device
static opencl_mask_range  *host_mask_ranges = NULL;
static opencl_placeholder *host_mask_plhdrs = NULL;
static int                *host_global_iters = NULL;
static opencl_mask_config host_mask_config;

// OpenCL device memory objects
static cl_mem mem_mask_ranges = NULL;
static cl_mem mem_mask_plhdrs = NULL;
static cl_mem mem_global_iters = NULL;
static cl_mem mem_mask_config = NULL;

// ---- FIX BEGIN: track allocated buffer size to avoid overruns ----
static size_t allocated_kpc = 0;
// ---- FIX END ----

static int custom_mask_initialized = 0;
static int num_loops_interleaved = 1; // Default to 1 if not otherwise defined
static int new_mask_data_ready = 0;

#define MIN_KEYS_PER_CRYPT      1
#define MAX_KEYS_PER_CRYPT      1


static struct fmt_tests tests[] = {
	//{"5a105e8b9d40e1329780d62ea2265d8a", "test1"},
	{FORMAT_TAG "5a105e8b9d40e1329780d62ea2265d8a", "test1"},
	/*{"098f6bcd4621d373cade4e832627b4f6", "test"},
	{FORMAT_TAG "378e2c4a07968da2eca692320136433d", "thatsworking"},
	{FORMAT_TAG "8ad8757baa8564dc136c1e07507f4a98", "test3"},*/
	{"d41d8cd98f00b204e9800998ecf8427e", ""},/*
#ifdef DEBUG
	{FORMAT_TAG "c9ccf168914a1bcfc3229f1948e67da0","1234567890123456789012345678901234567890123456789012345"},
#if PLAINTEXT_LENGTH >= 80
	{FORMAT_TAG "57edf4a22be3c955ac49da2e2107b67a","12345678901234567890123456789012345678901234567890123456789012345678901234567890"},
#endif
#endif
	{"{MD5}CY9rzUYh03PK3k6DJie09g==", "test"},
	*/{NULL}
};

struct fmt_main FMT_STRUCT;

static void set_kernel_args(void)
{
    if (!mem_mask_ranges || !mem_mask_plhdrs || !mem_global_iters || !mem_mask_config) {
        fprintf(stderr, "FATAL: mask buffers not allocated!\n");
        exit(1);
    }

    // Arguments 0‑3 (mask)
    HANDLE_CLERROR(clSetKernelArg(crypt_kernel, 0, sizeof(cl_mem), &mem_mask_ranges),  "Error arg 0");
    HANDLE_CLERROR(clSetKernelArg(crypt_kernel, 1, sizeof(cl_mem), &mem_mask_plhdrs),  "Error arg 1");
    HANDLE_CLERROR(clSetKernelArg(crypt_kernel, 2, sizeof(cl_mem), &mem_global_iters), "Error arg 2");
    HANDLE_CLERROR(clSetKernelArg(crypt_kernel, 3, sizeof(cl_mem), &mem_mask_config),  "Error arg 3");

    // Arguments 4 & 5 (key data)
    if (!buffer_keys || !buffer_idx) {
        fprintf(stderr, "FATAL: key buffers not allocated!\n");
        exit(1);
    }
    HANDLE_CLERROR(clSetKernelArg(crypt_kernel, 4, sizeof(cl_mem), &buffer_keys), "Error arg 4");
    HANDLE_CLERROR(clSetKernelArg(crypt_kernel, 5, sizeof(cl_mem), &buffer_idx),  "Error arg 5");

    // Arguments 6‑11 (hash‑check) – only if already initialized (after create_clobj())
    if (hc_bitmaps) {
        HANDLE_CLERROR(clSetKernelArg(crypt_kernel, 6,  sizeof(cl_mem), &hc_bitmaps),       "Error arg 6");
        HANDLE_CLERROR(clSetKernelArg(crypt_kernel, 7,  sizeof(cl_mem), &hc_offset_table),  "Error arg 7");
        HANDLE_CLERROR(clSetKernelArg(crypt_kernel, 8,  sizeof(cl_mem), &hc_hash_table),    "Error arg 8");
        HANDLE_CLERROR(clSetKernelArg(crypt_kernel, 9,  sizeof(cl_mem), &hc_return_hashes), "Error arg 9");
        HANDLE_CLERROR(clSetKernelArg(crypt_kernel, 10, sizeof(cl_mem), &hc_hash_ids),      "Error arg 10");
        HANDLE_CLERROR(clSetKernelArg(crypt_kernel, 11, sizeof(cl_mem), &hc_bitmap_dupe),   "Error arg 11");
    }
}

static void set_kernel_args_kpc(void)
{
	// Call the same function to guarantee slots 0-3 are always fully populated
	set_kernel_args();
}

static void release_clobj_kpc(void);
static void release_clobj(void);

static void create_clobj_kpc(size_t kpc)
{
    // If any mask buffers exist, release them completely
    if (mem_mask_ranges || mem_mask_plhdrs || host_mask_ranges) {
        release_clobj_kpc();
    }

    allocated_kpc = kpc;

    // Allocate host arrays fresh (never reuse old pointers)
    host_mask_ranges  = mem_calloc(MAX_LIMIT, sizeof(opencl_mask_range));
    host_mask_plhdrs  = mem_calloc(kpc, sizeof(opencl_placeholder));
    host_global_iters = mem_calloc(kpc * MAX_LIMIT, sizeof(int));

    // mask_config is a static struct, just clear it
    memset(&host_mask_config, 0, sizeof(host_mask_config));
    host_mask_config.iterations_per_thread = 1;
    host_mask_config.num_loops_interleaved = 1;

    // Key buffers (even if not used by this kernel, some host code expects them)
    saved_plain = mem_calloc(kpc, 64);
    saved_idx   = mem_calloc(kpc, sizeof(cl_uint));
    ui_keys     = mem_calloc(kpc, sizeof(*ui_keys));

    // Create mask device buffers
    mem_mask_ranges = clCreateBuffer(context[gpu_id], CL_MEM_READ_ONLY,
        MAX_LIMIT * sizeof(opencl_mask_range), NULL, &ret_code);
    HANDLE_CLERROR(ret_code, "mem_mask_ranges");

    mem_mask_plhdrs = clCreateBuffer(context[gpu_id], CL_MEM_READ_WRITE,
        kpc * sizeof(opencl_placeholder), NULL, &ret_code);
    HANDLE_CLERROR(ret_code, "mem_mask_plhdrs");

    mem_global_iters = clCreateBuffer(context[gpu_id], CL_MEM_READ_WRITE,
        kpc * MAX_LIMIT * sizeof(int), NULL, &ret_code);
    HANDLE_CLERROR(ret_code, "mem_global_iters");

    mem_mask_config = clCreateBuffer(context[gpu_id], CL_MEM_READ_ONLY,
        sizeof(opencl_mask_config), NULL, &ret_code);
    HANDLE_CLERROR(ret_code, "mem_mask_config");

    // Key device buffers (again, for completeness)
    buffer_keys = clCreateBuffer(context[gpu_id], CL_MEM_READ_WRITE,
        kpc * 64, NULL, &ret_code);
    HANDLE_CLERROR(ret_code, "buffer_keys");

    buffer_idx = clCreateBuffer(context[gpu_id], CL_MEM_READ_WRITE,
        kpc * sizeof(cl_uint), NULL, &ret_code);
    HANDLE_CLERROR(ret_code, "buffer_idx");

    // Upload initial mask config
    clEnqueueWriteBuffer(queue[gpu_id], mem_mask_config, CL_TRUE, 0,
        sizeof(opencl_mask_config), &host_mask_config, 0, NULL, NULL);

    // Set kernel arguments (mask‑only at this point – hash args will be added later)
    set_kernel_args();
}

static cl_uint *local_zero_bitmap = NULL;

static void create_clobj(void)
{
    if (buffer_int_keys) {
        release_clobj();
    }
    if (!local_zero_bitmap)
		local_zero_bitmap = mem_calloc(ocl_hc_hash_table_size / 32 + 1, sizeof(cl_uint));

    cl_uint dummy = 0;
    buffer_int_keys = clCreateBuffer(context[gpu_id],
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        4 * mask_int_cand.num_int_cand,
        mask_int_cand.int_cand ? mask_int_cand.int_cand : &dummy, &ret_code);
    HANDLE_CLERROR(ret_code, "buffer_int_keys");

    // Let the hash‑check subsystem create & fill all its buffers
    ocl_hc_128_crobj(crypt_kernel);   // This also sets kernel args incorrectly

    // Retrieve the actual handles
    ocl_hc_128_get_buffers(&hc_bitmaps, &hc_offset_table,
                           &hc_hash_table, &hc_return_hashes,
                           &hc_hash_ids, &hc_bitmap_dupe);

	my_loaded_hashes = ocl_hc_128_get_loaded_hashes();

    // Now bind all 12 kernel arguments correctly
    set_kernel_args();
}

static void release_clobj_kpc(void)
{
    // Unmap and free key buffers
    if (buffer_keys) {
        if (pinned_saved_keys) {
            clEnqueueUnmapMemObject(queue[gpu_id], pinned_saved_keys, saved_plain, 0, NULL, NULL);
            clReleaseMemObject(pinned_saved_keys);
            pinned_saved_keys = NULL;
        } else {
            MEM_FREE(saved_plain);
            saved_plain = NULL;
        }
        if (pinned_saved_idx) {
            clEnqueueUnmapMemObject(queue[gpu_id], pinned_saved_idx, saved_idx, 0, NULL, NULL);
            clReleaseMemObject(pinned_saved_idx);
            pinned_saved_idx = NULL;
        } else {
            MEM_FREE(saved_idx);
            saved_idx = NULL;
        }
        if (pinned_int_key_loc) {
            clEnqueueUnmapMemObject(queue[gpu_id], pinned_int_key_loc, saved_int_key_loc, 0, NULL, NULL);
            clReleaseMemObject(pinned_int_key_loc);
            pinned_int_key_loc = NULL;
        } else {
            MEM_FREE(saved_int_key_loc);
            saved_int_key_loc = NULL;
        }

        clFinish(queue[gpu_id]);

        clReleaseMemObject(buffer_keys);    buffer_keys = NULL;
		clReleaseMemObject(buffer_idx);     buffer_idx  = NULL;
		if (buffer_int_key_loc) {
			clReleaseMemObject(buffer_int_key_loc);
			buffer_int_key_loc = NULL;
		}
    }

    // Release mask device buffers
    if (mem_mask_ranges)  { clReleaseMemObject(mem_mask_ranges);  mem_mask_ranges  = NULL; }
    if (mem_mask_plhdrs)  { clReleaseMemObject(mem_mask_plhdrs);  mem_mask_plhdrs  = NULL; }
    if (mem_global_iters) { clReleaseMemObject(mem_global_iters); mem_global_iters = NULL; }
    if (mem_mask_config)  { clReleaseMemObject(mem_mask_config);  mem_mask_config  = NULL; }

    // Free host mask arrays – CRITICAL: set pointers to NULL!
    MEM_FREE(host_mask_ranges);   host_mask_ranges  = NULL;
    MEM_FREE(host_mask_plhdrs);   host_mask_plhdrs  = NULL;
    MEM_FREE(host_global_iters);  host_global_iters = NULL;

    MEM_FREE(ui_keys);  ui_keys = NULL;

    allocated_kpc = 0;
}

static void release_clobj(void)
{
	if (buffer_int_keys) {
		HANDLE_CLERROR(clReleaseMemObject(buffer_int_keys), "Error Releasing buffer_int_keys.");
		buffer_int_keys = 0;

		ocl_hc_128_rlobj();
	}
	MEM_FREE(local_zero_bitmap); local_zero_bitmap = NULL;
}

static void done(void)
{
	release_clobj_kpc();
	release_clobj();

	if (crypt_kernel) {
		HANDLE_CLERROR(clReleaseKernel(crypt_kernel), "Release kernel.");
		HANDLE_CLERROR(clReleaseProgram(program[gpu_id]), "Release Program.");

		crypt_kernel = NULL;
	}
}

static void init_kernel(unsigned int num_ld_hashes, char *bitmap_para)
{
	char build_opts[5000];
	int i;
	cl_ulong const_cache_size;

	if (crypt_kernel) {
		HANDLE_CLERROR(clReleaseKernel(crypt_kernel), "Release kernel.");
		HANDLE_CLERROR(clReleaseProgram(program[gpu_id]), "Release Program.");

		crypt_kernel = NULL;
	}

	shift64_ht_sz = (((1ULL << 63) % ocl_hc_hash_table_size) * 2) % ocl_hc_hash_table_size;
	shift64_ot_sz = (((1ULL << 63) % ocl_hc_offset_table_size) * 2) % ocl_hc_offset_table_size;

	for (i = 0; i < MASK_FMT_INT_PLHDR; i++)
		if (mask_skip_ranges && mask_skip_ranges[i] != -1)
			static_gpu_locations[i] = mask_int_cand.int_cpu_mask_ctx->
				ranges[mask_skip_ranges[i]].pos;
		else
			static_gpu_locations[i] = -1;

	HANDLE_CLERROR(clGetDeviceInfo(devices[gpu_id], CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(cl_ulong), &const_cache_size, 0), "failed to get CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE.");

	sprintf(build_opts, "-D OFFSET_TABLE_SIZE=%u -D HASH_TABLE_SIZE=%u"
		" -D SHIFT64_OT_SZ=%u -D SHIFT64_HT_SZ=%u -D NUM_LOADED_HASHES=%u"
		" -D NUM_INT_KEYS=%u %s -D IS_STATIC_GPU_MASK=%d"
		" -D CONST_CACHE_SIZE=%llu -D LOC_0=%d"
#if MASK_FMT_INT_PLHDR > 1
	" -D LOC_1=%d "
#endif
#if MASK_FMT_INT_PLHDR > 2
	"-D LOC_2=%d "
#endif
#if MASK_FMT_INT_PLHDR > 3
	"-D LOC_3=%d"
#endif
	, ocl_hc_offset_table_size, ocl_hc_hash_table_size, shift64_ot_sz, shift64_ht_sz,
	num_ld_hashes, mask_int_cand.num_int_cand, bitmap_para, mask_gpu_is_static,
	(unsigned long long)const_cache_size, static_gpu_locations[0]
#if MASK_FMT_INT_PLHDR > 1
	, static_gpu_locations[1]
#endif
#if MASK_FMT_INT_PLHDR > 2
	, static_gpu_locations[2]
#endif
#if MASK_FMT_INT_PLHDR > 3
	, static_gpu_locations[3]
#endif
	);

	opencl_build_kernel("$JOHN/opencl/md5_kernel.cl", gpu_id, build_opts, 0);
	crypt_kernel = clCreateKernel(program[gpu_id], "md5", &ret_code);
	HANDLE_CLERROR(ret_code, "Error creating kernel. Double-check kernel name?");
}

static void init(struct fmt_main *_self)
{
	self = _self;
    ocl_hc_num_loaded_hashes = 0;
    ocl_hc_128_init(_self);
    opencl_prepare_dev(gpu_id);
	mask_int_cand_target = 1;
}

/* Convert {MD5}CY9rzUYh03PK3k6DJie09g== to 098f6bcd4621d373cade4e832627b4f6 */
static char *prepare(char *fields[10], struct fmt_main *self)
{
	static char out[CIPHERTEXT_LENGTH + 1];

	if (!strncmp(fields[1], FORMAT_TAG2, FORMAT_TAG2_LEN) &&
	    strlen(fields[1]) == FORMAT_TAG2_LEN + 24) {
		int res;

		res = base64_convert(&fields[1][FORMAT_TAG2_LEN], e_b64_mime, 24,
		                     out, e_b64_hex, sizeof(out),
		                     flg_Base64_HEX_LOCASE, 0);
		if (res >= 0)
			return out;
	}

	return fields[1];
}

static int valid(char *ciphertext, struct fmt_main *self)
{
	char *p, *q;

	p = ciphertext;
	if (!strncmp(p, FORMAT_TAG, TAG_LENGTH))
		p += TAG_LENGTH;

	q = p;
	while (atoi16[ARCH_INDEX(*q)] != 0x7F) {
		if (*q >= 'A' && *q <= 'F') /* support lowercase only */
			return 0;
		q++;
	}
	return !*q && q - p == CIPHERTEXT_LENGTH;
}

static char *split(char *ciphertext, int index, struct fmt_main *self)
{
	static char out[TAG_LENGTH + CIPHERTEXT_LENGTH + 1];
	int len;

	if (!strncmp(ciphertext, FORMAT_TAG, TAG_LENGTH))
		return ciphertext;

	memset(out, 0, sizeof(out));
	memcpy(out, FORMAT_TAG, TAG_LENGTH);
	len = strlen(ciphertext)+1;
	if (len > CIPHERTEXT_LENGTH + 1)
		len = CIPHERTEXT_LENGTH + 1;
	memcpy(out + TAG_LENGTH, ciphertext, len);
	return out;
}

static void *get_binary(char *ciphertext)
{
	static uint32_t out[DIGEST_SIZE / 4];
	char *p;
	int i;
	p = ciphertext + TAG_LENGTH;
	for (i = 0; i < sizeof(out); i++) {
		((unsigned char*)out)[i] = (atoi16[ARCH_INDEX(*p)] << 4) | atoi16[ARCH_INDEX(p[1])];
		p += 2;
	}
	return out;
}

static int get_hash_0(int index) { return bt_hash_table_128[ocl_hc_hash_ids[3 + 3 * index]] & PH_MASK_0; }
static int get_hash_1(int index) { return bt_hash_table_128[ocl_hc_hash_ids[3 + 3 * index]] & PH_MASK_1; }
static int get_hash_2(int index) { return bt_hash_table_128[ocl_hc_hash_ids[3 + 3 * index]] & PH_MASK_2; }
static int get_hash_3(int index) { return bt_hash_table_128[ocl_hc_hash_ids[3 + 3 * index]] & PH_MASK_3; }
static int get_hash_4(int index) { return bt_hash_table_128[ocl_hc_hash_ids[3 + 3 * index]] & PH_MASK_4; }
static int get_hash_5(int index) { return bt_hash_table_128[ocl_hc_hash_ids[3 + 3 * index]] & PH_MASK_5; }
static int get_hash_6(int index) { return bt_hash_table_128[ocl_hc_hash_ids[3 + 3 * index]] & PH_MASK_6; }

static void clear_keys(void)
{
    // 1. Clear existing JTR buffers defensively
    if (saved_idx != NULL) {
        memset(saved_idx, 0, sizeof(cl_uint) * global_work_size);
    }

    // 2. Clear your NEW custom buffers defensively
    if (host_mask_ranges != NULL) {
        memset(host_mask_ranges, 0, MAX_LIMIT * sizeof(opencl_mask_range));
    }

    if (host_mask_plhdrs != NULL) {
        // ---- FIX BEGIN: use allocated capacity for safety ----
        memset(host_mask_plhdrs, 0, allocated_kpc * sizeof(opencl_placeholder));
        // ---- FIX END ----
    }

    if (host_global_iters != NULL) {
        // ---- FIX BEGIN ----
        memset(host_global_iters, 0, allocated_kpc * MAX_LIMIT * sizeof(int));
        // ---- FIX END ----
    }

	memset(&host_mask_config, 0, sizeof(host_mask_config));
	host_mask_config.iterations_per_thread = 1;
	host_mask_config.num_loops_interleaved = 1;

    key_idx = 0;
}

static void set_key(char *_key, int index)
{
    // Allocate UI array if needed
    if (ui_keys == NULL) {
        ui_keys = mem_calloc(self->params.max_keys_per_crypt, sizeof(*ui_keys));
    }

    // Allocate key buffers for GPU transfer if not yet done
    if (saved_plain == NULL) {
        saved_plain = mem_calloc(self->params.max_keys_per_crypt, 64);
        saved_idx   = mem_calloc(self->params.max_keys_per_crypt, sizeof(cl_uint));
    }

    // Save the plaintext for host retrieval (get_key)
    int len = strlen(_key);
    if (len > PLAINTEXT_LENGTH) len = PLAINTEXT_LENGTH;
    memcpy(ui_keys[index], _key, len);
    ui_keys[index][len] = '\0';

    // Save the base key for the GPU (64 bytes per key, padded with zeros)
    char *key_buf = (char*)&saved_plain[index * 16];  // 16 uints = 64 bytes
    memset(key_buf, 0, 64);
    memcpy(key_buf, _key, len);
    saved_idx[index] = len;   // for direct mode: full password length
                              // for mask mode: template length = mask length

    // Mask‑mode setup (only once per batch, at index 0)
    mask_cpu_context *ctx = mask_int_cand.int_cpu_mask_ctx;
	if (ctx && index == 0) {
		int current_limit = ctx->active_count;
		if (current_limit > MAX_LIMIT) current_limit = MAX_LIMIT;

		for (int i = 0; i < current_limit; i++) {
			int ri = ctx->active_idx[i];          // source: actual JtR range index (11,12,13,14)
			host_mask_ranges[i].count = ctx->ranges[ri].count;   // dest: sequential i
			host_mask_ranges[i].pos   = ctx->ranges[ri].pos;
			memcpy(&host_mask_ranges[i].chars, ctx->ranges[ri].chars,
				ctx->ranges[ri].count);
		}

		for (size_t gid = 0; gid < allocated_kpc; gid++) {
			host_mask_plhdrs[gid].limit = current_limit;
			host_mask_plhdrs[gid].current_k = 0;
			for (int i = 0; i < current_limit; i++)
				host_mask_plhdrs[gid].active_idx[i] = i;  // sequential: 0,1,2,3
		}
		memset(host_global_iters, 0, allocated_kpc * MAX_LIMIT * sizeof(int));

        // Set mask config (iterations per thread)
        host_mask_config.iterations_per_thread = (mask_int_cand.num_int_cand > 0) ? mask_int_cand.num_int_cand : 1;
        host_mask_config.num_loops_interleaved = 1;

        new_mask_data_ready = 1;
    }

    key_idx = index;
}

/*
 * Host-side C port of the OpenCL opencl_flat_next_state() kernel function.
 * Advances the anti-diagonal mask enumeration by exactly one step.
 * Returns 0 on success, 1 when the entire keyspace is exhausted.
 *
 * All parameters are identical in semantics to the OpenCL version so that
 * replaying N steps on the host produces the same character sequence as the
 * GPU loop counter `iter`.
 */
static int host_flat_next_state(
        const opencl_mask_range *ranges,
        const int               *active_idx,
        int                     *local_iter,
        int                     *current_k,
        int                      limit)
{
    int i, j, weight;

    /* ── Anti-diagonal weight-shifting within the current K-plane ── */
    for (i = limit - 2; i >= 0; i--) {
        int ri      = active_idx[i];
        int ri_next = active_idx[i + 1];

        if (local_iter[i] > 0 &&
            local_iter[i + 1] < ranges[ri_next].count - 1)
        {
            local_iter[i]--;
            local_iter[i + 1]++;

            /* Redistribute weight from positions i+2 onwards */
            weight = 0;
            for (j = i + 2; j < limit; j++) {
                weight         += local_iter[j];
                local_iter[j]   = 0;
            }
            j = i + 1;
            while (weight > 0 && j < limit) {
                int rj          = active_idx[j];
                int can_add     = ranges[rj].count - 1 - local_iter[j];
                int add         = (weight > can_add) ? can_add : weight;
                local_iter[j]  += add;
                weight         -= add;
                j++;
            }
            return 0; /* still inside current K-plane */
        }
    }

    /* ── Current K-plane exhausted: move to the next anti-diagonal ── */
    (*current_k)++;

    {
        int remaining = *current_k;
        for (i = 0; i < limit; i++) {
            int ri      = active_idx[i];
            int max_idx = ranges[ri].count - 1;
            if (remaining <= max_idx) {
                local_iter[i] = remaining;
                remaining     = 0;
            } else {
                local_iter[i] = max_idx;
                remaining    -= max_idx;
            }
        }
        if (remaining > 0)
            return 1; /* keyspace fully exhausted */
    }
    return 0;
}

/*
 * Reconstruct the exact plaintext that the GPU produced for match record
 * `index` (0-based within the cracks found in the current batch).
 *
 * After the crypt_all() fix, JtR passes a MATCH index, not a candidate
 * index.  We recover the originating work-item (gid) and the kernel loop
 * counter (iter) from ocl_hc_hash_ids, then replay the anti-diagonal state
 * machine `iter` steps from the all-zero initial state to find the exact
 * mask characters the GPU used.
 */
static char *get_key(int index)
{
    static char reconstructed[PLAINTEXT_LENGTH + 1];
    cl_uint gid, iter;
    int     i, len, limit;
    int     local_iter_state[MAX_LIMIT];
    int     current_k;

    if (index < 0)
        index = 0;

    /* ── Safety: before any GPU run ui_keys[0] is the best we can do ── */
    if (ui_keys == NULL)
        return "";
    if (ocl_hc_hash_ids == NULL || (cl_uint)ocl_hc_hash_ids[0] == 0)
        return ui_keys[index % self->params.max_keys_per_crypt];

    /* ── Extract (gid, iter) from the GPU match record ── */
    gid  = ocl_hc_hash_ids[1 + 3 * index]; /* work-item that found the match  */
    iter = ocl_hc_hash_ids[2 + 3 * index]; /* kernel loop counter at the match */

    if (gid >= (cl_uint)self->params.max_keys_per_crypt)
        gid = 0;

    /* ── Start from the base/template key that set_key() stored ── */
    len = (saved_idx != NULL) ? (int)saved_idx[gid] : 0;
    if (len < 0 || len > PLAINTEXT_LENGTH)
        len = PLAINTEXT_LENGTH;
    memcpy(reconstructed, ui_keys[gid], len);
    reconstructed[len] = '\0';

    /* ── Replay the mask state machine `iter` steps from [0,0,...,0] ── *
     *                                                                    *
     * Every GID begins with the same initial state (all local_iter = 0, *
     * current_k = 0) as set up in set_key().  Advancing `iter` times    *
     * yields exactly the characters the GPU thread used.                 */
    if (host_mask_ranges == NULL || host_mask_plhdrs == NULL)
        return reconstructed; /* no mask – template is the full key */

    limit = host_mask_plhdrs[gid].limit;
    if (limit <= 0)
        return reconstructed;

    memset(local_iter_state, 0, sizeof(local_iter_state));
    current_k = 0;

    for (cl_uint step = 0; step < iter; step++) {
        if (host_flat_next_state(
                host_mask_ranges,
                host_mask_plhdrs[gid].active_idx,
                local_iter_state,
                &current_k,
                limit))
            break; /* exhausted – shouldn't happen for a real crack */
    }

    /* ── Inject the reconstructed characters into the template ── */
    for (i = 0; i < limit; i++) {
        int active_pos = host_mask_plhdrs[gid].active_idx[i];
        int pos        = host_mask_ranges[active_pos].pos;
        int c_idx      = local_iter_state[i];
        if (pos >= 0 && pos < PLAINTEXT_LENGTH)
            reconstructed[pos] = (char)host_mask_ranges[active_pos].chars[c_idx];
    }

    return reconstructed;
}

static int crypt_all(int *pcount, struct db_salt *salt)
{
    int count = *pcount;
    if (count == 0)
        return 0;

    if (!crypt_kernel) {
        fprintf(stderr, "FATAL: crypt_kernel is NULL in crypt_all!\n");
        exit(1);
    }

    // Ensure mask + key buffers exist
    if (!mem_mask_ranges || !mem_mask_plhdrs || !buffer_keys) {
        create_clobj_kpc(self->params.max_keys_per_crypt);
        set_kernel_args();
    }

    // Determine local work size
    size_t max_lws = 0;
    cl_int err = clGetKernelWorkGroupInfo(crypt_kernel, devices[gpu_id],
                             CL_KERNEL_WORK_GROUP_SIZE,
                             sizeof(max_lws), &max_lws, NULL);
    if (err != CL_SUCCESS || max_lws == 0)
        max_lws = 1;

    if (local_work_size == 0)
        local_work_size = (max_lws > 64) ? 64 : max_lws;
    if (local_work_size > max_lws)
        local_work_size = max_lws;

    // Determine global work size
    if (mask_int_cand.num_int_cand > 1) {
        size_t step = local_work_size;
        while (step % mask_int_cand.num_int_cand != 0)
            step += local_work_size;
        global_work_size = GET_NEXT_MULTIPLE(count, step);
    } else {
        global_work_size = GET_NEXT_MULTIPLE(count, local_work_size);
    }

    if (global_work_size > allocated_kpc) {
        create_clobj_kpc(global_work_size);
    }

    // Write key buffers (full size – unused items get length 0)
    BENCH_CLERROR(clEnqueueWriteBuffer(queue[gpu_id], buffer_keys, CL_TRUE, 0,
        global_work_size * 64, saved_plain, 0, NULL, NULL),
        "Write buffer_keys");
    BENCH_CLERROR(clEnqueueWriteBuffer(queue[gpu_id], buffer_idx, CL_TRUE, 0,
        global_work_size * sizeof(cl_uint), saved_idx, 0, NULL, NULL),
        "Write buffer_idx");

    // Upload mask data if changed
    if (new_mask_data_ready) {
        BENCH_CLERROR(clEnqueueWriteBuffer(queue[gpu_id], mem_mask_ranges, CL_FALSE, 0,
            MAX_LIMIT * sizeof(opencl_mask_range), host_mask_ranges, 0, NULL, NULL),
            "Ranges");
        BENCH_CLERROR(clEnqueueWriteBuffer(queue[gpu_id], mem_mask_plhdrs, CL_FALSE, 0,
            global_work_size * sizeof(opencl_placeholder), host_mask_plhdrs, 0, NULL, NULL),
            "Placeholders");
        BENCH_CLERROR(clEnqueueWriteBuffer(queue[gpu_id], mem_global_iters, CL_FALSE, 0,
            global_work_size * MAX_LIMIT * sizeof(int), host_global_iters, 0, NULL, NULL),
            "Iters");
        host_mask_config.iterations_per_thread = (mask_int_cand.num_int_cand > 0) ? mask_int_cand.num_int_cand : 1;
        host_mask_config.num_loops_interleaved = 1;
        BENCH_CLERROR(clEnqueueWriteBuffer(queue[gpu_id], mem_mask_config, CL_TRUE, 0,
            sizeof(opencl_mask_config), &host_mask_config, 0, NULL, NULL),
            "Config");
        new_mask_data_ready = 0;
    }

    // ── RESET THE **ENTIRE** OUTPUT BUFFER AND DUPLICATE BITMAP BEFORE LAUNCH ──
    // This is the only safe way to prevent stale data from causing phantom cracks.
    if (hc_hash_ids) {
        size_t output_sz = (3 * ocl_hc_num_loaded_hashes + 1) * sizeof(cl_uint);
        cl_uint *zero_output = mem_calloc(3 * ocl_hc_num_loaded_hashes + 1, sizeof(cl_uint));
        BENCH_CLERROR(clEnqueueWriteBuffer(queue[gpu_id], hc_hash_ids, CL_TRUE, 0,
            output_sz, zero_output, 0, NULL, NULL), "Reset output buffer");
        MEM_FREE(zero_output);
    }
    if (hc_bitmap_dupe && local_zero_bitmap) {
        BENCH_CLERROR(clEnqueueWriteBuffer(queue[gpu_id], hc_bitmap_dupe, CL_TRUE, 0,
            (ocl_hc_hash_table_size / 32 + 1) * sizeof(cl_uint),
            local_zero_bitmap, 0, NULL, NULL), "Reset dupe bitmap");
    }

    // Bind all kernel arguments (0‑11)
    set_kernel_args();

    // ── Launch kernel ──
    BENCH_CLERROR(clEnqueueNDRangeKernel(queue[gpu_id], crypt_kernel, 1, NULL,
        &global_work_size, &local_work_size, 0, NULL, NULL),
        "Enqueue md5 kernel");
    BENCH_CLERROR(clFinish(queue[gpu_id]), "clFinish");

    // ── Read back cracks ──
    cl_uint num_cracks = 0;
    BENCH_CLERROR(clEnqueueReadBuffer(queue[gpu_id], hc_hash_ids, CL_TRUE, 0,
        sizeof(cl_uint), &num_cracks, 0, NULL, NULL), "Read crack count");

    if (num_cracks > 0 && num_cracks <= ocl_hc_num_loaded_hashes) {
        // Read the extra hash words into the real loaded_hashes array
        BENCH_CLERROR(clEnqueueReadBuffer(queue[gpu_id], hc_return_hashes, CL_TRUE, 0,
            2 * sizeof(cl_uint) * num_cracks, my_loaded_hashes, 0, NULL, NULL),
            "Read return hashes");
        // Read the full output buffer into ocl_hc_hash_ids
        BENCH_CLERROR(clEnqueueReadBuffer(queue[gpu_id], hc_hash_ids, CL_TRUE, 0,
            (3 * num_cracks + 1) * sizeof(cl_uint), ocl_hc_hash_ids, 0, NULL, NULL),
            "Read crack data");
    }

    // Ensure the host crack count is correct (even if 0)
    ocl_hc_hash_ids[0] = num_cracks;

    // ── Reset output buffer again for the next call ──
    if (hc_hash_ids) {
        size_t output_sz = (3 * ocl_hc_num_loaded_hashes + 1) * sizeof(cl_uint);
        cl_uint *zero_output = mem_calloc(3 * ocl_hc_num_loaded_hashes + 1, sizeof(cl_uint));
        BENCH_CLERROR(clEnqueueWriteBuffer(queue[gpu_id], hc_hash_ids, CL_TRUE, 0,
            output_sz, zero_output, 0, NULL, NULL), "Reset output buffer after read");
        MEM_FREE(zero_output);
    }

    *pcount = count * mask_int_cand.num_int_cand;

    // Return the MATCH count so JtR iterates 0..num_cracks-1 in the compare loop.
    // get_hash_N(i) = bt_hash_table_128[ocl_hc_hash_ids[3+3*i]] is valid for i < num_cracks
    // because num_cracks ≤ num_loaded_hashes = 100000.
    return (int)num_cracks;
}

static void auto_tune(struct db_main *db, long double kernel_run_ms)
{
	size_t gws_limit, gws_init;
	size_t lws_limit, lws_init;

	struct timeval startc, endc;
	long double time_ms = 0, old_time_ms = 0;

	size_t pcount, count;
	size_t i;

	int tune_gws = 1, tune_lws = 1;

	char key[PLAINTEXT_LENGTH + 1];

	memset(key, 0xF5, PLAINTEXT_LENGTH);
	key[PLAINTEXT_LENGTH] = 0;

	gws_limit = MIN((0xf << 22) * 4 / BUFSIZE,
			get_max_mem_alloc_size(gpu_id) / BUFSIZE);
	get_power_of_two(gws_limit);
	if (gws_limit > MIN((0xf << 22) * 4 / BUFSIZE,
		get_max_mem_alloc_size(gpu_id) / BUFSIZE))
		gws_limit >>= 1;

#if SIZEOF_SIZE_T > 4
	while (gws_limit * mask_int_cand.num_int_cand > 0xffffffffUL)
		gws_limit >>= 1;
#endif

	lws_limit = get_kernel_max_lws(gpu_id, crypt_kernel);
	lws_init = get_kernel_preferred_multiple(gpu_id, crypt_kernel);

	if (gpu_amd(device_info[gpu_id]))
		gws_init = gws_limit >> 6;
	else if (gpu_nvidia(device_info[gpu_id]))
		gws_init = gws_limit >> 8;
	else
		gws_init = 1024;

	if (gws_init > gws_limit)
		gws_init = gws_limit;
	if (gws_init < lws_init)
		lws_init = gws_init;

	if (self_test_running) {
		opencl_get_sane_lws_gws_values();
	} else {
		local_work_size = 0;
		global_work_size = 0;
		opencl_get_user_preferences(FORMAT_LABEL);
	}
	if (local_work_size) {
		tune_lws = 0;
		if (local_work_size & (local_work_size - 1))
			get_power_of_two(local_work_size);
		if (local_work_size > lws_limit)
			local_work_size = lws_limit;
	}
	if (global_work_size)
		tune_gws = 0;

	/* Auto tune start.*/
	pcount = gws_init;
	count = 0;
#define calc_ms(start, end)	\
		((long double)(end.tv_sec - start.tv_sec) * 1000.000 + \
			(long double)(end.tv_usec - start.tv_usec) / 1000.000)
	if (tune_gws) {
		create_clobj_kpc(pcount);
		set_kernel_args_kpc();
		clear_keys();
		for (i = 0; i < pcount; i++)
			set_key(key, i);
		gettimeofday(&startc, NULL);
		crypt_all((int *)&pcount, NULL);
		gettimeofday(&endc, NULL);
		time_ms = calc_ms(startc, endc);
		count = (size_t)((kernel_run_ms / time_ms) * (long double)gws_init);
		get_power_of_two(count);
	}

	if (tune_gws && tune_lws)
		release_clobj_kpc();

	if (tune_lws) {
		count = tune_gws ? count : global_work_size;
		if (count > gws_limit)
			count = gws_limit;
		create_clobj_kpc(count);
		set_kernel_args_kpc();
		pcount = count;
		clear_keys();
		for (i = 0; i < pcount; i++)
			set_key(key, i);
		local_work_size = lws_init;
		gettimeofday(&startc, NULL);
		crypt_all((int *)&pcount, NULL);
		gettimeofday(&endc, NULL);
		old_time_ms = calc_ms(startc, endc);
		local_work_size = 2 * lws_init;

		while (local_work_size <= lws_limit) {
			gettimeofday(&startc, NULL);
			pcount = count;
			crypt_all((int *)&pcount, NULL);
			gettimeofday(&endc, NULL);
			time_ms = calc_ms(startc, endc);
			if (old_time_ms < time_ms) {
				local_work_size /= 2;
				break;
			}
			old_time_ms = time_ms;
			local_work_size *= 2;
		}

		if (local_work_size > lws_limit)
			local_work_size = lws_limit;
	}

	if (tune_gws && tune_lws) {
		if (old_time_ms > kernel_run_ms) {
			count /= 2;
		}
		else {
			count = (size_t)((kernel_run_ms / old_time_ms) * (long double)count);
			get_power_of_two(count);
		}
	}

	if (tune_gws) {
		if (count > gws_limit)
			count = gws_limit;
		release_clobj_kpc();
		create_clobj_kpc(count);
		set_kernel_args_kpc();
		global_work_size = count;
	}

	if (!tune_gws && !tune_lws) {
		create_clobj_kpc(global_work_size);
		set_kernel_args_kpc();
	}
	/* Auto tune finish.*/

	if (global_work_size % local_work_size) {
		global_work_size = GET_NEXT_MULTIPLE(global_work_size, local_work_size);
		get_power_of_two(global_work_size);
		release_clobj_kpc();
		if (global_work_size > gws_limit)
			global_work_size = gws_limit;
		create_clobj_kpc(global_work_size);
		set_kernel_args_kpc();
	}
	if (global_work_size > gws_limit) {
		release_clobj_kpc();
		global_work_size = gws_limit;
		create_clobj_kpc(global_work_size);
		set_kernel_args_kpc();
	}

	// ---- FIX BEGIN: final sanity check ----
	if (global_work_size > allocated_kpc) {
		fprintf(stderr, "FATAL: auto_tune resulted in GWS (%zu) > allocated (%zu)\n",
		        global_work_size, allocated_kpc);
		exit(1);
	}
	// ---- FIX END ----

	clear_keys();

	self->params.max_keys_per_crypt = global_work_size;

	if ((!self_test_running && options.verbosity >= VERB_DEFAULT) || ocl_always_show_ws) {
		if (mask_int_cand.num_int_cand > 1)
			fprintf(stderr, "LWS="Zu" GWS="Zu" x%d%s", local_work_size,
			        global_work_size, mask_int_cand.num_int_cand, (options.flags & FLG_TEST_CHK) ? " " : "\n");
		else
			fprintf(stderr, "LWS="Zu" GWS="Zu"%s", local_work_size,
			        global_work_size, (options.flags & FLG_TEST_CHK) ? " " : "\n");
	}

#undef calc_ms
}

static void reset(struct db_main *db)
{
    release_clobj();
    release_clobj_kpc();

    ocl_hc_num_loaded_hashes = db->salts->count;
    ocl_hc_128_prepare_table(db->salts);
    init_kernel(ocl_hc_num_loaded_hashes, ocl_hc_128_select_bitmap(ocl_hc_num_loaded_hashes));

    create_clobj_kpc(self->params.max_keys_per_crypt);
    create_clobj();

    if (mem_mask_ranges == NULL) {
        fprintf(stderr, "FATAL: Buffers failed to allocate. Check logs.\n");
        exit(1);
    }

    set_kernel_args();
    auto_tune(db, 100);
}

struct fmt_main FMT_STRUCT = {
	{
		FORMAT_LABEL,
		FORMAT_NAME,
		ALGORITHM_NAME,
		BENCHMARK_COMMENT,
		BENCHMARK_LENGTH,
		0,
		PLAINTEXT_LENGTH,
		BINARY_SIZE,
		BINARY_ALIGN,
		SALT_SIZE,
		SALT_ALIGN,
		MIN_KEYS_PER_CRYPT,
		MAX_KEYS_PER_CRYPT,
		FMT_CASE | FMT_8_BIT | FMT_REMOVE | FMT_MASK,
		{ NULL },
		{ FORMAT_TAG, FORMAT_TAG2 },
		tests
	}, {
		init,
		done,
		reset,
		prepare,
		valid,
		split,
		get_binary,
		fmt_default_salt,
		{ NULL },
		fmt_default_source,
		{
			fmt_default_binary_hash_0,
			fmt_default_binary_hash_1,
			fmt_default_binary_hash_2,
			fmt_default_binary_hash_3,
			fmt_default_binary_hash_4,
			fmt_default_binary_hash_5,
			fmt_default_binary_hash_6
		},
		fmt_default_salt_hash,
		NULL,
		fmt_default_set_salt,
		set_key,
		get_key,
		clear_keys,
		crypt_all,
		{
			get_hash_0,
			get_hash_1,
			get_hash_2,
			get_hash_3,
			get_hash_4,
			get_hash_5,
			get_hash_6
		},
		ocl_hc_128_cmp_all,
		ocl_hc_128_cmp_one,
		ocl_hc_128_cmp_exact
	}
};

#endif /* plugin stanza */

#endif /* HAVE_OPENCL */
