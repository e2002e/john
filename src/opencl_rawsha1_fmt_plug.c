/*
 * Copyright (c) 2012-2023, magnum
 * and Copyright (c) 2015, Sayantan Datta <sdatta@openwall.com>
 * This software is hereby released to the general public under
 * the following terms: Redistribution and use in source and binary
 * forms, with or without modification, are permitted.
 */

#ifdef HAVE_OPENCL
#define FMT_STRUCT fmt_opencl_rawSHA1

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
#include "opencl_common.h"
#include "config.h"
#include "options.h"
#include "base64_convert.h"
#include "rawSHA1_common.h"
#include "mask_ext.h"
#include "bt_interface.h"

#define FORMAT_LABEL			"raw-SHA1-opencl"
#define FORMAT_NAME			""
#define ALGORITHM_NAME			"SHA1 OpenCL"

#define BENCHMARK_COMMENT		""
#define BENCHMARK_LENGTH		0x107

#define PLAINTEXT_LENGTH		55 /* Max. is 55 with current kernel */
#define BUFSIZE				((PLAINTEXT_LENGTH+3)/4*4)

#define DIGEST_SIZE			20
#define BINARY_SIZE			20
#define BINARY_ALIGN			4
#define SALT_SIZE			0
#define SALT_ALIGN			1

#define MIN_KEYS_PER_CRYPT		1
#define MAX_KEYS_PER_CRYPT		1

static cl_mem pinned_saved_keys, pinned_saved_idx, pinned_int_key_loc;
static cl_mem buffer_keys, buffer_idx, buffer_int_keys, buffer_int_key_loc;
static cl_uint *saved_plain, *saved_idx, *saved_int_key_loc;
static int static_gpu_locations[MASK_FMT_INT_PLHDR];

/* GPU K-ordered generation state (port of the raw-md5-opencl generator). */
static cl_mem g_buf_suf, g_buf_table, g_buf_startv, g_buf_rowcnt, g_buf_littmpl,
              g_buf_keypos, g_buf_count, g_buf_cstart, g_buf_chars0, g_buf_segs,
              g_buf_boxbounds;   /* odometer-shell per-seg lo/radix bounds (arg 24) */
static unsigned g_uploaded_serial;       /* mask_gpu_serial of last table upload  */
static unsigned g_uploaded_plan_serial;  /* mask_gpu_serial of last plan upload    */
static cl_uint g_nseg;                    /* segments in the uploaded plan          */

/* Device mirror of one mask_gpu_plan segment (must match gen_seg in the kernel).
 * Four ulongs then four uints = 48 bytes, naturally 8-aligned. */
typedef struct {
	cl_ulong vbase;
	cl_ulong vcnt;
	cl_ulong lstart;
	cl_ulong suf_off;
	cl_uint  limit;
	cl_uint  len;
	cl_uint  max_k;
	cl_uint  ksize;
} gen_seg;
/* Candidates generated per work-item in the gen kernel. Tunable via JOHN_GEN_R. */
static cl_uint gen_R = 256;
/* >0 builds the register-resident gen kernel (-D GEN_REGS) with this unroll width
 * (compile-time GEN_REG_MAX). Set from JOHN_GEN_REGS in reset(). */
static cl_uint gen_reg_max = 0;
/* >=0 (JOHN_GEN_HEAD=H / JOHN_GEN_MARGINAL) builds the hybrid head/tail kernel
 * (-D GEN_HEAD=H). -1 = disabled (full conditional). */
static int gen_head = -1;
/* >0 (JOHN_GEN_DEEPEN) builds the odometer-shell deepening gen kernel
 * (-D GEN_ODOMETER): each segment is a magnitude-threshold sub-box enumerated by a
 * plain mixed-radix odometer over per-position rank ranges instead of the rank-sum
 * simplex, so the GPU runs divergence-free while the host emits the sub-boxes
 * most-probable-band first. Mutually exclusive with GEN_REGS. */
static int gen_deepen = 0;
static void gen_release_all(void);

/* Parameters of the gen launch in flight, so set_kernel_args() can re-bind the
 * gen-kernel args (10-24) if the kernel is rebuilt mid-launch (a rebuild wipes all
 * kernel args; the generic re-bind covers only 0-9). */
static int      g_cur_gen = 0;
static cl_ulong g_cur_gbase;
static cl_uint  g_cur_gcount;
static void set_kernel_args_gen(void);

/* True when the GPU actually generates candidates. mask_gpu_gen EXCEPT in
 * MASK_GPU_CPU validation, where mask mode streams host-materialized candidates
 * through the normal crypt path. Set in reset() before the kernel is built. */
static int gen_active = 0;

static cl_mem buffer_offset_table, buffer_hash_table, buffer_return_hashes, buffer_hash_ids, buffer_bitmap_dupe, buffer_bitmaps;
static OFFSET_TABLE_WORD *offset_table = NULL;
static cl_uint *loaded_hashes = NULL, num_loaded_hashes, *hash_ids = NULL, *bitmaps = NULL;
static unsigned int hash_table_size, offset_table_size, shift64_ht_sz, shift64_ot_sz, shift128_ht_sz, shift128_ot_sz;
static cl_ulong bitmap_size_bits = 0;

static unsigned int key_idx = 0;
static struct fmt_main *self;
static cl_uint *zero_buffer;

#define MIN_KEYS_PER_CRYPT      1
#define MAX_KEYS_PER_CRYPT      1


struct fmt_main FMT_STRUCT;

static void set_kernel_args_kpc()
{
	HANDLE_CLERROR(clSetKernelArg(crypt_kernel, 0, sizeof(buffer_keys), (void *) &buffer_keys), "Error setting argument 1.");
	HANDLE_CLERROR(clSetKernelArg(crypt_kernel, 1, sizeof(buffer_idx), (void *) &buffer_idx), "Error setting argument 2.");
	HANDLE_CLERROR(clSetKernelArg(crypt_kernel, 2, sizeof(buffer_int_key_loc), (void *) &buffer_int_key_loc), "Error setting argument 3.");
}

static void set_kernel_args()
{
	HANDLE_CLERROR(clSetKernelArg(crypt_kernel, 3, sizeof(buffer_int_keys), (void *) &buffer_int_keys), "Error setting argument 4.");
	HANDLE_CLERROR(clSetKernelArg(crypt_kernel, 4, sizeof(buffer_bitmaps), (void *) &buffer_bitmaps), "Error setting argument 5.");
	HANDLE_CLERROR(clSetKernelArg(crypt_kernel, 5, sizeof(buffer_offset_table), (void *) &buffer_offset_table), "Error setting argument 6.");
	HANDLE_CLERROR(clSetKernelArg(crypt_kernel, 6, sizeof(buffer_hash_table), (void *) &buffer_hash_table), "Error setting argument 7.");
	HANDLE_CLERROR(clSetKernelArg(crypt_kernel, 7, sizeof(buffer_return_hashes), (void *) &buffer_return_hashes), "Error setting argument 8.");
	HANDLE_CLERROR(clSetKernelArg(crypt_kernel, 8, sizeof(buffer_hash_ids), (void *) &buffer_hash_ids), "Error setting argument 9.");
	HANDLE_CLERROR(clSetKernelArg(crypt_kernel, 9, sizeof(buffer_bitmap_dupe), (void *) &buffer_bitmap_dupe), "Error setting argument 10.");

	/* If a gen launch is in flight and the kernel was just rebuilt, re-bind the
	 * gen args (10-24) too; the bindings above only cover the hash-check args. */
	if (gen_active && g_cur_gen) {
		set_kernel_args_gen();
		HANDLE_CLERROR(clSetKernelArg(crypt_kernel, 21, sizeof(cl_ulong), &g_cur_gbase), "arg21");
		HANDLE_CLERROR(clSetKernelArg(crypt_kernel, 22, sizeof(cl_uint), &g_cur_gcount), "arg22");
		HANDLE_CLERROR(clSetKernelArg(crypt_kernel, 23, sizeof(cl_uint), &gen_R), "arg23");
	}
}

static void release_clobj_kpc(void);
static void release_clobj(void);

static void create_clobj_kpc(size_t kpc)
{
	release_clobj_kpc();

	pinned_saved_keys = clCreateBuffer(context[gpu_id], CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, BUFSIZE * kpc, NULL, &ret_code);
	if (ret_code != CL_SUCCESS) {
		saved_plain = (cl_uint *) mem_alloc(BUFSIZE * kpc);
		if (saved_plain == NULL)
			HANDLE_CLERROR(ret_code, "Error creating page-locked memory pinned_saved_keys.");
	}
	else {
		saved_plain = (cl_uint *) clEnqueueMapBuffer(queue[gpu_id], pinned_saved_keys, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, BUFSIZE * kpc, 0, NULL, NULL, &ret_code);
		HANDLE_CLERROR(ret_code, "Error mapping page-locked memory saved_plain.");
	}

	pinned_saved_idx = clCreateBuffer(context[gpu_id], CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(cl_uint) * kpc, NULL, &ret_code);
	HANDLE_CLERROR(ret_code, "Error creating page-locked memory pinned_saved_idx.");
	saved_idx = (cl_uint *) clEnqueueMapBuffer(queue[gpu_id], pinned_saved_idx, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(cl_uint) * kpc, 0, NULL, NULL, &ret_code);
	HANDLE_CLERROR(ret_code, "Error mapping page-locked memory saved_idx.");

	pinned_int_key_loc = clCreateBuffer(context[gpu_id], CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(cl_uint) * kpc, NULL, &ret_code);
	HANDLE_CLERROR(ret_code, "Error creating page-locked memory pinned_int_key_loc.");
	saved_int_key_loc = (cl_uint *) clEnqueueMapBuffer(queue[gpu_id], pinned_int_key_loc, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(cl_uint) * kpc, 0, NULL, NULL, &ret_code);
	HANDLE_CLERROR(ret_code, "Error mapping page-locked memory saved_int_key_loc.");

	// create and set arguments
	buffer_keys = clCreateBuffer(context[gpu_id], CL_MEM_READ_ONLY, BUFSIZE * kpc, NULL, &ret_code);
	HANDLE_CLERROR(ret_code, "Error creating buffer argument buffer_keys.");

	buffer_idx = clCreateBuffer(context[gpu_id], CL_MEM_READ_ONLY, 4 * kpc, NULL, &ret_code);
	HANDLE_CLERROR(ret_code, "Error creating buffer argument buffer_idx.");

	buffer_int_key_loc = clCreateBuffer(context[gpu_id], CL_MEM_READ_ONLY, sizeof(cl_uint) * kpc, NULL, &ret_code);
	HANDLE_CLERROR(ret_code, "Error creating buffer argument buffer_int_key_loc.");
}

static void create_clobj(void)
{
	cl_ulong max_alloc_size_bytes = 0;
	cl_ulong cache_size_bytes = 0;
	cl_uint dummy = 0;

	release_clobj();

	HANDLE_CLERROR(clGetDeviceInfo(devices[gpu_id], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &max_alloc_size_bytes, 0), "failed to get CL_DEVICE_MAX_MEM_ALLOC_SIZE.");
	HANDLE_CLERROR(clGetDeviceInfo(devices[gpu_id], CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(cl_ulong), &cache_size_bytes, 0), "failed to get CL_DEVICE_GLOBAL_MEM_CACHE_SIZE.");

	if (max_alloc_size_bytes & (max_alloc_size_bytes - 1)) {
		get_power_of_two(max_alloc_size_bytes);
		max_alloc_size_bytes >>= 1;
	}
	if (max_alloc_size_bytes >= 536870912) max_alloc_size_bytes = 536870912;

	if (!cache_size_bytes) cache_size_bytes = 1024;

	zero_buffer = (cl_uint *) mem_calloc(hash_table_size/32 + 1, sizeof(cl_uint));

	buffer_return_hashes = clCreateBuffer(context[gpu_id], CL_MEM_WRITE_ONLY, 2 * sizeof(cl_uint) * num_loaded_hashes, NULL, &ret_code);
	HANDLE_CLERROR(ret_code, "Error creating buffer argument buffer_return_hashes.");

	buffer_hash_ids = clCreateBuffer(context[gpu_id], CL_MEM_READ_WRITE, (3 * num_loaded_hashes + 1) * sizeof(cl_uint), NULL, &ret_code);
	HANDLE_CLERROR(ret_code, "Error creating buffer argument buffer_buffer_hash_ids.");

	buffer_bitmap_dupe = clCreateBuffer(context[gpu_id], CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, (hash_table_size/32 + 1) * sizeof(cl_uint), zero_buffer, &ret_code);
	HANDLE_CLERROR(ret_code, "Error creating buffer argument buffer_bitmap_dupe.");

	buffer_bitmaps = clCreateBuffer(context[gpu_id], CL_MEM_READ_WRITE, max_alloc_size_bytes, NULL, &ret_code);
	HANDLE_CLERROR(ret_code, "Error creating buffer argument buffer_bitmaps.");

	//dummy is used as dummy parameter
	buffer_int_keys = clCreateBuffer(context[gpu_id], CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 4 * mask_int_cand.num_int_cand, mask_int_cand.int_cand ? mask_int_cand.int_cand : (void *)&dummy, &ret_code);
	HANDLE_CLERROR(ret_code, "Error creating buffer argument buffer_int_keys.");

	buffer_offset_table = clCreateBuffer(context[gpu_id], CL_MEM_READ_ONLY, offset_table_size * sizeof(OFFSET_TABLE_WORD), NULL, &ret_code);
	HANDLE_CLERROR(ret_code, "Error creating buffer argument buffer_offset_table.");

	buffer_hash_table = clCreateBuffer(context[gpu_id], CL_MEM_READ_ONLY, hash_table_size * sizeof(unsigned int) * 2, NULL, &ret_code);
	HANDLE_CLERROR(ret_code, "Error creating buffer argument buffer_hash_table.");

	HANDLE_CLERROR(clEnqueueWriteBuffer(queue[gpu_id], buffer_hash_ids, CL_TRUE, 0, sizeof(cl_uint), zero_buffer, 0, NULL, NULL), "failed in clEnqueueWriteBuffer buffer_hash_ids.");
}

static void release_clobj_kpc(void)
{
	if (buffer_idx) {
		if (pinned_saved_keys) {
			HANDLE_CLERROR(clEnqueueUnmapMemObject(queue[gpu_id], pinned_saved_keys, saved_plain, 0, NULL, NULL), "Error Unmapping saved_plain.");
			HANDLE_CLERROR(clReleaseMemObject(pinned_saved_keys), "Error Releasing pinned_saved_keys.");
		}
		else
			MEM_FREE(saved_plain);
		HANDLE_CLERROR(clEnqueueUnmapMemObject(queue[gpu_id], pinned_saved_idx, saved_idx, 0, NULL, NULL), "Error Unmapping saved_idx.");
		HANDLE_CLERROR(clEnqueueUnmapMemObject(queue[gpu_id], pinned_int_key_loc, saved_int_key_loc, 0, NULL, NULL), "Error Unmapping saved_int_key_loc.");
		HANDLE_CLERROR(clFinish(queue[gpu_id]), "Error releasing mappings.");
		HANDLE_CLERROR(clReleaseMemObject(buffer_keys), "Error Releasing buffer_keys.");
		HANDLE_CLERROR(clReleaseMemObject(buffer_idx), "Error Releasing buffer_idx.");
		HANDLE_CLERROR(clReleaseMemObject(buffer_int_key_loc), "Error Releasing buffer_int_key_loc.");
		HANDLE_CLERROR(clReleaseMemObject(pinned_saved_idx), "Error Releasing pinned_saved_idx.");
		HANDLE_CLERROR(clReleaseMemObject(pinned_int_key_loc), "Error Releasing pinned_int_key_loc.");
		buffer_idx = 0;
		pinned_saved_keys = 0;
	}
}

static void release_clobj(void)
{
	if (buffer_offset_table) {
		HANDLE_CLERROR(clReleaseMemObject(buffer_int_keys), "Error Releasing buffer_int_keys.");
		HANDLE_CLERROR(clReleaseMemObject(buffer_return_hashes), "Error Releasing buffer_return_hashes.");
		HANDLE_CLERROR(clReleaseMemObject(buffer_offset_table), "Error Releasing buffer_offset_table.");
		HANDLE_CLERROR(clReleaseMemObject(buffer_hash_table), "Error Releasing buffer_hash_table.");
		HANDLE_CLERROR(clReleaseMemObject(buffer_bitmap_dupe), "Error Releasing buffer_bitmap_dupe.");
		HANDLE_CLERROR(clReleaseMemObject(buffer_hash_ids), "Error Releasing buffer_hash_ids.");
		HANDLE_CLERROR(clReleaseMemObject(buffer_bitmaps), "Error Releasing buffer_bitmap.");
		MEM_FREE(zero_buffer);
		buffer_offset_table = 0;
	}
}

static void done(void)
{
	release_clobj_kpc();
	release_clobj();
	gen_release_all();

	if (crypt_kernel) {
		HANDLE_CLERROR(clReleaseKernel(crypt_kernel), "Release kernel.");
		HANDLE_CLERROR(clReleaseProgram(program[gpu_id]), "Release Program.");

		crypt_kernel = NULL;
	}

	if (loaded_hashes)
		MEM_FREE(loaded_hashes);
	if (hash_ids)
		MEM_FREE(hash_ids);
	if (bitmaps)
		MEM_FREE(bitmaps);
	if (offset_table)
		MEM_FREE(offset_table);
	if (bt_hash_table_128)
		MEM_FREE(bt_hash_table_192);
}

static void init_kernel(unsigned int num_ld_hashes, char *bitmap_para)
{
	char build_opts[5000];
	int i;
	uint64_t shift128;
	cl_ulong const_cache_size;

	if (crypt_kernel) {
		HANDLE_CLERROR(clReleaseKernel(crypt_kernel), "Release kernel.");
		HANDLE_CLERROR(clReleaseProgram(program[gpu_id]), "Release Program.");

		crypt_kernel = NULL;
	}

	shift64_ht_sz = (((1ULL << 63) % hash_table_size) * 2) % hash_table_size;
	shift64_ot_sz = (((1ULL << 63) % offset_table_size) * 2) % offset_table_size;

	shift128 = (uint64_t)shift64_ht_sz * shift64_ht_sz;
	shift128_ht_sz = shift128 % hash_table_size;

	shift128 = (uint64_t)shift64_ot_sz * shift64_ot_sz;
	shift128_ot_sz = shift128 % offset_table_size;

	for (i = 0; i < MASK_FMT_INT_PLHDR; i++)
		if (mask_skip_ranges && mask_skip_ranges[i] != -1)
			static_gpu_locations[i] = mask_int_cand.int_cpu_mask_ctx->
				ranges[mask_skip_ranges[i]].pos;
		else
			static_gpu_locations[i] = -1;

	HANDLE_CLERROR(clGetDeviceInfo(devices[gpu_id], CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(cl_ulong), &const_cache_size, 0), "failed to get CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE.");

	sprintf(build_opts, "-D OFFSET_TABLE_SIZE=%u -D HASH_TABLE_SIZE=%u"
		" -D SHIFT64_OT_SZ=%u -D SHIFT64_HT_SZ=%u -D SHIFT128_OT_SZ=%u"
		" -D SHIFT128_HT_SZ=%u -D NUM_LOADED_HASHES=%u"
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
	, offset_table_size, hash_table_size, shift64_ot_sz, shift64_ht_sz,
	shift128_ot_sz, shift128_ht_sz, num_ld_hashes,
	mask_int_cand.num_int_cand, bitmap_para, mask_gpu_is_static,
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

	if (gen_active) {
		/*
		 * Size the gen kernel's per-work-item arrays and message-word pack to
		 * this run's max candidate length. GEN_NDW = data words spanning
		 * [0, maxlen] (incl. the 0x80); the SHA1 bit length lives in W[15] and
		 * W[14], both beyond GEN_NDW (<=14). GEN_MAX_POS = GEN_NDW*4 covers
		 * key[len] and the iter[] index range.
		 */
		int maxlen = options.eff_maxlength;
		int gen_ndw, gen_max_pos;
		char go[64];

		if (maxlen < 1 || maxlen > PLAINTEXT_LENGTH)
			maxlen = PLAINTEXT_LENGTH;
		gen_ndw = (maxlen + 4) / 4;
		if (gen_ndw > 14)
			gen_ndw = 14;
		gen_max_pos = gen_ndw * 4;
		snprintf(go, sizeof(go), " -D GPU_GEN -D GEN_NDW=%d -D GEN_MAX_POS=%d",
		         gen_ndw, gen_max_pos);
		strcat(build_opts, go);
	}

	if (gen_active && gen_head >= 0) {
		char ho[32];

		snprintf(ho, sizeof(ho), " -D GEN_HEAD=%d", gen_head);
		strcat(build_opts, ho);
	}

	if (gen_active && gen_reg_max && gen_head < 0 && !gen_deepen) {
		char ro[64];

		snprintf(ro, sizeof(ro), " -D GEN_REGS -D GEN_REG_MAX=%u", gen_reg_max);
		strcat(build_opts, ro);
	}

	/* Odometer-shell deepening: divergence-free mixed-radix enumeration of
	 * magnitude sub-boxes (see gen_deepen). Owns the index->iter step, so it sits
	 * in the non-REGS materialize path; force GEN_REGS off (done in reset). */
	if (gen_active && gen_deepen)
		strcat(build_opts, " -D GEN_ODOMETER");

	/*
	 * Stage the compact Markov table in local memory when the mask is eligible
	 * (mask.c sets ltab_ok). Finalized once the mask template is built; on a build
	 * that precedes that (npos==0) it stays off and the kernel uses the global
	 * table.
	 */
	if (gen_active && !gen_reg_max && gen_head < 0 && !gen_deepen) {
		const mask_gpu_tables *t = mask_gpu_get_tables();
		size_t bytes = t->ltab_ok
		    ? (size_t)t->npos * t->ltab_nc * t->ltab_nc : 0;

		if (t->ltab_ok && bytes > 0 && bytes <= 16384 && !getenv("JOHN_NO_LTAB")) {
			char lo[112];

			snprintf(lo, sizeof(lo), " -D GEN_LOCALTAB -D GEN_LTAB_POS=%d"
			    " -D GEN_LTAB_NC=%d -D GEN_LTAB_BASE=%d",
			    t->npos, t->ltab_nc, t->ltab_base);
			strcat(build_opts, lo);
		}
	}

	opencl_build_kernel("$JOHN/opencl/sha1_kernel.cl", gpu_id, build_opts, 0);
	crypt_kernel = clCreateKernel(program[gpu_id],
	                              gen_active ? "sha1_gen" : "sha1", &ret_code);
	HANDLE_CLERROR(ret_code, "Error creating kernel. Double-check kernel name?");
}

static void init(struct fmt_main *_self)
{
	self = _self;
	num_loaded_hashes = 0;

	opencl_prepare_dev(gpu_id);
	mask_int_cand_target = opencl_speed_index(gpu_id) / 300;
}

static void *get_binary(char *ciphertext)
{
	static uint32_t full[DIGEST_SIZE / 4];
	unsigned char *realcipher = (unsigned char*)full;

	ciphertext += TAG_LENGTH;
	base64_convert(ciphertext, e_b64_hex, HASH_LENGTH,
	               realcipher, e_b64_raw, sizeof(full),
	               flg_Base64_NO_FLAGS, 0);
	alter_endianity(realcipher, DIGEST_SIZE);

	return (void*)realcipher;
}

static int get_hash_0(int index) { return bt_hash_table_192[hash_ids[3 + 3 * index]] & PH_MASK_0; }
static int get_hash_1(int index) { return bt_hash_table_192[hash_ids[3 + 3 * index]] & PH_MASK_1; }
static int get_hash_2(int index) { return bt_hash_table_192[hash_ids[3 + 3 * index]] & PH_MASK_2; }
static int get_hash_3(int index) { return bt_hash_table_192[hash_ids[3 + 3 * index]] & PH_MASK_3; }
static int get_hash_4(int index) { return bt_hash_table_192[hash_ids[3 + 3 * index]] & PH_MASK_4; }
static int get_hash_5(int index) { return bt_hash_table_192[hash_ids[3 + 3 * index]] & PH_MASK_5; }
static int get_hash_6(int index) { return bt_hash_table_192[hash_ids[3 + 3 * index]] & PH_MASK_6; }

static void clear_keys(void)
{
	memset(saved_idx, 0, sizeof(cl_uint) * global_work_size);
	key_idx = 0;
}

static void set_key(char *_key, int index)
{
	const uint32_t *key = (uint32_t*)_key;
	int len = strlen(_key);

	if (mask_int_cand.num_int_cand > 1 && !mask_gpu_is_static) {
		int i;
		saved_int_key_loc[index] = 0;
		for (i = 0; i < MASK_FMT_INT_PLHDR; i++) {
			if (mask_skip_ranges[i] != -1)  {
				saved_int_key_loc[index] |= ((mask_int_cand.
				int_cpu_mask_ctx->ranges[mask_skip_ranges[i]].offset +
				mask_int_cand.int_cpu_mask_ctx->
				ranges[mask_skip_ranges[i]].pos) & 0xff) << (i << 3);
			}
			else
				saved_int_key_loc[index] |= 0x80 << (i << 3);
		}
	}

	saved_idx[index] = (key_idx << 6) | len;

	while (len > 4) {
		saved_plain[key_idx++] = *key++;
		len -= 4;
	}
	if (len)
		saved_plain[key_idx++] = *key & (0xffffffffU >> (32 - (len << 3)));
}

static char *get_key(int index)
{
	static char out[PLAINTEXT_LENGTH + 1];
	int i, len, int_index, t;
	char *key;

	if (gen_active) {
		int kl, loop;
		uint64_t off, g;

		/* Each work-item gid produced gen_R contiguous candidates; the kernel
		 * stored gid in the id slot and the sub-index j in the int_index slot,
		 * so the candidate's offset within the block is gid*gen_R + j. The block
		 * covers virtual indices [mask_gpu_cur_base, +count) of the multi-length
		 * plan, so map the virtual index to its length-loop before unranking. */
		if (hash_ids == NULL || hash_ids[0] == 0 ||
		    index >= hash_ids[0] || hash_ids[0] > num_loaded_hashes)
			off = (uint64_t)index;
		else
			off = (uint64_t)hash_ids[1 + 3 * index] * gen_R +
			      hash_ids[2 + 3 * index];

		mask_gpu_virt_to_loop(mask_gpu_cur_base + off, &loop, &g);
		mask_gpu_unrank_key(loop, g, out, &kl);
		return out;
	}

	if (hash_ids == NULL || hash_ids[0] == 0 ||
	    index >= hash_ids[0] || hash_ids[0] > num_loaded_hashes) {
		t = index;
		int_index = 0;
	}
	else  {
		t = hash_ids[1 + 3 * index];
		int_index = hash_ids[2 + 3 * index];

	}

	if (t >= global_work_size) {
		//fprintf(stderr, "Get key error! %d %d\n", t, index);
		t = 0;
	}

	len = saved_idx[t] & 63;
	key = (char*)&saved_plain[saved_idx[t] >> 6];

	for (i = 0; i < len; i++)
		out[i] = *key++;
	out[i] = 0;

	if (len && mask_skip_ranges && mask_int_cand.num_int_cand > 1) {
		for (i = 0; i < MASK_FMT_INT_PLHDR && mask_skip_ranges[i] != -1; i++)
			if (mask_gpu_is_static)
				out[static_gpu_locations[i]] =
				mask_int_cand.int_cand[int_index].x[i];
			else
				out[(saved_int_key_loc[t]& (0xff << (i * 8))) >> (i * 8)] =
				mask_int_cand.int_cand[int_index].x[i];
	}

	return out;
}

static void prepare_table(struct db_salt *salt) {
	unsigned int *bin, i;
	struct db_password *pw, *last;

	num_loaded_hashes = (salt->count);

	MEM_FREE(loaded_hashes);
	MEM_FREE(hash_ids);
	MEM_FREE(offset_table);
	MEM_FREE(bt_hash_table_192);

	loaded_hashes = (cl_uint*) mem_alloc(6 * num_loaded_hashes * sizeof(cl_uint));
	hash_ids = (cl_uint*) mem_calloc((3 * num_loaded_hashes + 1), sizeof(cl_uint));

	last = pw = salt->list;
	i = 0;
	do {
		bin = (unsigned int *)pw->binary;
		if (bin == NULL) {
			if (last == pw)
				salt->list = pw->next;
			else
				last->next = pw->next;
		} else {
			last = pw;
			loaded_hashes[6 * i] = bin[0];
			loaded_hashes[6 * i + 1] = bin[1];
			loaded_hashes[6 * i + 2] = bin[2];
			loaded_hashes[6 * i + 3] = bin[3];
			loaded_hashes[6 * i + 4] = bin[4];
			loaded_hashes[6 * i + 5] = 0;
			i++;
		}
	} while ((pw = pw->next)) ;

	if (i != (salt->count)) {
		fprintf(stderr,
			"Something went wrong while preparing hashes..Exiting..\n");
		error();
	}

	num_loaded_hashes = bt_create_perfect_hash_table(192, (void *)loaded_hashes,
				num_loaded_hashes,
			        &offset_table,
			        &offset_table_size,
			        &hash_table_size, 0);

	if (!num_loaded_hashes) {
		MEM_FREE(bt_hash_table_192);
		MEM_FREE(offset_table);
		fprintf(stderr, "Failed to create Hash Table for cracking.\n");
		error();
	}
}

/* Use only for bitmaps up to 64K (0xffff) */
static void prepare_bitmap_8(cl_ulong bmp_sz, cl_uint **bitmap_ptr)
{
	unsigned int i;
	MEM_FREE(*bitmap_ptr);
	*bitmap_ptr = (cl_uint*) mem_calloc((bmp_sz >> 2), sizeof(cl_uint));

	for (i = 0; i < num_loaded_hashes; i++) {
		unsigned int bmp_idx = (loaded_hashes[6 * i]) & (bmp_sz - 1);
		(*bitmap_ptr)[bmp_idx >> 5] |= (1U << (bmp_idx & 31));

		bmp_idx = (loaded_hashes[6 * i] >> 16) & (bmp_sz - 1);
		(*bitmap_ptr)[(bmp_sz >> 5) + (bmp_idx >> 5)] |=
			(1U << (bmp_idx & 31));

		bmp_idx = (loaded_hashes[6 * i + 1]) & (bmp_sz - 1);
		(*bitmap_ptr)[(bmp_sz >> 4) + (bmp_idx >> 5)] |=
			(1U << (bmp_idx & 31));

		bmp_idx = (loaded_hashes[6 * i + 1] >> 16) & (bmp_sz - 1);
		(*bitmap_ptr)[(bmp_sz >> 5) * 3 + (bmp_idx >> 5)] |=
			(1U << (bmp_idx & 31));

		bmp_idx = (loaded_hashes[6 * i + 2]) & (bmp_sz - 1);
		(*bitmap_ptr)[(bmp_sz >> 3) + (bmp_idx >> 5)] |=
			(1U << (bmp_idx & 31));

		bmp_idx = (loaded_hashes[6 * i + 2] >> 16) & (bmp_sz - 1);
		(*bitmap_ptr)[(bmp_sz >> 5) * 5 + (bmp_idx >> 5)] |=
			(1U << (bmp_idx & 31));

		bmp_idx = (loaded_hashes[6 * i + 3]) & (bmp_sz - 1);
		(*bitmap_ptr)[(bmp_sz >> 5) * 6 + (bmp_idx >> 5)] |=
			(1U << (bmp_idx & 31));

		bmp_idx = (loaded_hashes[6 * i + 3] >> 16) & (bmp_sz - 1);
		(*bitmap_ptr)[(bmp_sz >> 5) * 7 + (bmp_idx >> 5)] |=
			(1U << (bmp_idx & 31));
	}
}

static void prepare_bitmap_4(cl_ulong bmp_sz, cl_uint **bitmap_ptr)
{
	unsigned int i;
	MEM_FREE(*bitmap_ptr);
	*bitmap_ptr = (cl_uint*) mem_calloc((bmp_sz >> 3), sizeof(cl_uint));

	for (i = 0; i < num_loaded_hashes; i++) {
		unsigned int bmp_idx = loaded_hashes[6 * i + 3] & (bmp_sz - 1);
		(*bitmap_ptr)[bmp_idx >> 5] |= (1U << (bmp_idx & 31));

		bmp_idx = loaded_hashes[6 * i + 2] & (bmp_sz - 1);
		(*bitmap_ptr)[(bmp_sz >> 5) + (bmp_idx >> 5)] |=
			(1U << (bmp_idx & 31));

		bmp_idx = loaded_hashes[6 * i + 1] & (bmp_sz - 1);
		(*bitmap_ptr)[(bmp_sz >> 4) + (bmp_idx >> 5)] |=
			(1U << (bmp_idx & 31));

		bmp_idx = loaded_hashes[6 * i] & (bmp_sz - 1);
		(*bitmap_ptr)[(bmp_sz >> 5) * 3 + (bmp_idx >> 5)] |=
			(1U << (bmp_idx & 31));
	}
}

static void prepare_bitmap_1(cl_ulong bmp_sz, cl_uint **bitmap_ptr)
{
	unsigned int i;
	MEM_FREE(*bitmap_ptr);
	*bitmap_ptr = (cl_uint*) mem_calloc((bmp_sz >> 5), sizeof(cl_uint));

	for (i = 0; i < num_loaded_hashes; i++) {
		unsigned int bmp_idx = loaded_hashes[6 * i + 3] & (bmp_sz - 1);
		(*bitmap_ptr)[bmp_idx >> 5] |= (1U << (bmp_idx & 31));
	}
}

static char* select_bitmap(unsigned int num_ld_hashes)
{	static char kernel_params[200];
	cl_ulong max_local_mem_sz_bytes = 0;
	unsigned int cmp_steps = 2, use_local = 0;

	HANDLE_CLERROR(clGetDeviceInfo(devices[gpu_id], CL_DEVICE_LOCAL_MEM_SIZE,
		sizeof(cl_ulong), &max_local_mem_sz_bytes, 0),
		"failed to get CL_DEVICE_LOCAL_MEM_SIZE.");

	if (num_loaded_hashes <= 5100) {
		if (amd_gcn_10(device_info[gpu_id]) ||
			amd_vliw4(device_info[gpu_id]))
			bitmap_size_bits = 512 * 1024;

		else if (amd_gcn_11(device_info[gpu_id]) ||
			max_local_mem_sz_bytes < 16384 ||
			cpu(device_info[gpu_id]))
			bitmap_size_bits = 256 * 1024;

		else {
			bitmap_size_bits = 32 * 1024;
			cmp_steps = 4;
			use_local = 1;
		}
	}

	else if (num_loaded_hashes <= 10100) {
		if (amd_gcn_10(device_info[gpu_id]) ||
			amd_vliw4(device_info[gpu_id]))
			bitmap_size_bits = 512 * 1024;

		else if (amd_gcn_11(device_info[gpu_id]) ||
			max_local_mem_sz_bytes < 32768 ||
			cpu(device_info[gpu_id]))
			bitmap_size_bits = 256 * 1024;

		else {
			bitmap_size_bits = 64 * 1024;
			cmp_steps = 4;
			use_local = 1;
		}
	}

	else if (num_loaded_hashes <= 20100) {
		if (amd_gcn_10(device_info[gpu_id]))
			bitmap_size_bits = 1024 * 1024;

		else if (amd_gcn_11(device_info[gpu_id]) ||
			max_local_mem_sz_bytes < 32768)
			bitmap_size_bits = 512 * 1024;

		else if (amd_vliw4(device_info[gpu_id]) ||
			cpu(device_info[gpu_id])) {
			bitmap_size_bits = 256 * 1024;
			cmp_steps = 4;
		}

		else {
			bitmap_size_bits = 32 * 1024;
			cmp_steps = 8;
			use_local = 1;
		}
	}

	else if (num_loaded_hashes <= 250100)
		bitmap_size_bits = 2048 * 1024;

	else if (num_loaded_hashes <= 1100100) {
		if (!amd_gcn_11(device_info[gpu_id]))
			bitmap_size_bits = 4096 * 1024;

		else
			bitmap_size_bits = 2048 * 1024;
	}

	else if (num_loaded_hashes <= 1500100) {
		bitmap_size_bits = 4096 * 1024 * 2;
		cmp_steps = 1;
	}

	else if (num_loaded_hashes <= 2700100) {
		bitmap_size_bits = 4096 * 1024 * 2 * 2;
		cmp_steps = 1;
	}

	else {
		cl_ulong mult = num_loaded_hashes / 2700100;
		cl_ulong buf_sz;
		bitmap_size_bits = 4096 * 4096;
		get_power_of_two(mult);
		bitmap_size_bits *= mult;
		buf_sz = get_max_mem_alloc_size(gpu_id);
		if (buf_sz & (buf_sz - 1)) {
			get_power_of_two(buf_sz);
			buf_sz >>= 1;
		}
		if (buf_sz >= 536870912)
			buf_sz = 536870912;
		if ((bitmap_size_bits >> 3) > buf_sz)
			bitmap_size_bits = buf_sz << 3;
		cmp_steps = 1;
	}

	if (cmp_steps == 1)
		prepare_bitmap_1(bitmap_size_bits, &bitmaps);

	else if (cmp_steps <= 4)
		prepare_bitmap_4(bitmap_size_bits, &bitmaps);

	else
		prepare_bitmap_8(bitmap_size_bits, &bitmaps);

	sprintf(kernel_params,
	        "-D SELECT_CMP_STEPS=%u -D BITMAP_MASK=0x%xU -D USE_LOCAL_BITMAPS=%u",
	        cmp_steps, (uint32_t)(bitmap_size_bits - 1), use_local);

	bitmap_size_bits *= cmp_steps;

	return kernel_params;
}

static cl_mem gen_copy_buf(size_t sz, const void *host)
{
	cl_mem m = clCreateBuffer(context[gpu_id],
	    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sz ? sz : 1,
	    (void *)host, &ret_code);
	HANDLE_CLERROR(ret_code, "Error creating GPU-gen buffer.");
	return m;
}

static void gen_release_buf(cl_mem *m)
{
	if (*m) {
		HANDLE_CLERROR(clReleaseMemObject(*m), "Error releasing GPU-gen buffer.");
		*m = 0;
	}
}

static void gen_release_all(void)
{
	gen_release_buf(&g_buf_suf);
	gen_release_buf(&g_buf_table);
	gen_release_buf(&g_buf_startv);
	gen_release_buf(&g_buf_rowcnt);
	gen_release_buf(&g_buf_littmpl);
	gen_release_buf(&g_buf_keypos);
	gen_release_buf(&g_buf_count);
	gen_release_buf(&g_buf_cstart);
	gen_release_buf(&g_buf_chars0);
	gen_release_buf(&g_buf_segs);
	gen_release_buf(&g_buf_boxbounds);
	g_uploaded_serial = 0;
	g_uploaded_plan_serial = 0;
}

/* Upload the per-mask Markov tables once per mask config (tracked by serial). */
static void gen_upload_tables(void)
{
	const mask_gpu_tables *t = mask_gpu_get_tables();
	size_t npos = t->npos;

	if (g_buf_table && g_uploaded_serial == mask_gpu_serial)
		return;

	gen_release_buf(&g_buf_table);
	gen_release_buf(&g_buf_startv);
	gen_release_buf(&g_buf_rowcnt);
	gen_release_buf(&g_buf_littmpl);
	gen_release_buf(&g_buf_keypos);
	gen_release_buf(&g_buf_count);
	gen_release_buf(&g_buf_cstart);
	gen_release_buf(&g_buf_chars0);

	size_t table_bytes = npos * 256 * 64 * sizeof(cl_uint);
	g_buf_table   = gen_copy_buf(table_bytes, t->uint_table);
	g_buf_startv  = gen_copy_buf(npos * 256, t->startv);
	g_buf_rowcnt  = gen_copy_buf(npos * 256, t->rowcnt);
	g_buf_littmpl = gen_copy_buf(t->littmpl_len, t->littmpl);
	g_buf_keypos  = gen_copy_buf(npos * sizeof(cl_int), t->keypos);
	g_buf_count   = gen_copy_buf(npos * sizeof(cl_int), t->count);
	g_buf_cstart  = gen_copy_buf(npos, t->cstart);
	g_buf_chars0  = gen_copy_buf(npos, t->chars0);

	g_uploaded_serial = mask_gpu_serial;
}

/* Upload the multi-length plan: per-segment metadata + concatenated suffix-DP
 * tables (or odometer lo/radix bounds). Tracked by serial. */
static void gen_upload_plan(void)
{
	const mask_gpu_plan *plan = mask_gpu_get_plan();
	gen_seg *segs;
	cl_ulong *suf;
	char seen[MASK_MAX_INC_LEN + 2];
	unsigned char dummy_zero = 0;
	int s;

	if (g_buf_segs && g_uploaded_plan_serial == mask_gpu_serial)
		return;

	gen_release_buf(&g_buf_segs);
	gen_release_buf(&g_buf_suf);
	gen_release_buf(&g_buf_boxbounds);

	/* Odometer-shell mode: no suffix-DP; upload the per-segment lo/radix bounds
	 * (seg->suf_off indexes into them) instead. */
	if (plan->odometer) {
		g_buf_boxbounds = gen_copy_buf(
		    plan->boxbounds_n ? plan->boxbounds_n : 1,
		    plan->boxbounds_n ? (void *)plan->boxbounds : (void *)&dummy_zero);
	}

	suf = mem_alloc((plan->suf_total ? plan->suf_total : 1) * sizeof(cl_ulong));
	segs = mem_alloc((plan->nseg ? plan->nseg : 1) * sizeof(gen_seg));
	memset(seen, 0, sizeof(seen));
	for (s = 0; s < plan->nseg; s++) {
		const mask_gpu_seg *sg = &plan->seg[s];

		if (gen_reg_max && (cl_uint)sg->limit > gen_reg_max) {
			fprintf(stderr, "Error: JOHN_GEN_REGS width %u too small for a "
			    "mask length-loop with %d generated positions; rebuild with "
			    "JOHN_GEN_REGS=%d or larger.\n",
			    gen_reg_max, sg->limit, sg->limit);
			error();
		}

		if (!plan->odometer && !seen[sg->loop]) {
			const mask_gpu_loop *gl = mask_gpu_get_loop(sg->loop);

			memcpy(suf + sg->suf_off, gl->suf,
			       (size_t)(gl->limit + 1) * gl->ksize * sizeof(cl_ulong));
			seen[sg->loop] = 1;
		}
		segs[s].vbase   = sg->vbase;
		segs[s].vcnt    = sg->vcnt;
		segs[s].lstart  = sg->lstart;
		segs[s].suf_off = sg->suf_off;
		segs[s].limit   = sg->limit;
		segs[s].len     = sg->len;
		segs[s].max_k   = sg->max_k;
		segs[s].ksize   = sg->ksize;
	}

	g_buf_suf  = gen_copy_buf((plan->suf_total ? plan->suf_total : 1) *
	                          sizeof(cl_ulong), suf);
	g_buf_segs = gen_copy_buf((plan->nseg ? plan->nseg : 1) * sizeof(gen_seg),
	                          segs);
	g_nseg = plan->nseg;

	MEM_FREE(suf);
	MEM_FREE(segs);
	g_uploaded_plan_serial = mask_gpu_serial;
}

static void set_kernel_args_gen(void)
{
	HANDLE_CLERROR(clSetKernelArg(crypt_kernel, 10, sizeof(g_buf_suf), &g_buf_suf), "arg10");
	HANDLE_CLERROR(clSetKernelArg(crypt_kernel, 11, sizeof(g_buf_table), &g_buf_table), "arg11");
	HANDLE_CLERROR(clSetKernelArg(crypt_kernel, 12, sizeof(g_buf_startv), &g_buf_startv), "arg12");
	HANDLE_CLERROR(clSetKernelArg(crypt_kernel, 13, sizeof(g_buf_rowcnt), &g_buf_rowcnt), "arg13");
	HANDLE_CLERROR(clSetKernelArg(crypt_kernel, 14, sizeof(g_buf_littmpl), &g_buf_littmpl), "arg14");
	HANDLE_CLERROR(clSetKernelArg(crypt_kernel, 15, sizeof(g_buf_keypos), &g_buf_keypos), "arg15");
	HANDLE_CLERROR(clSetKernelArg(crypt_kernel, 16, sizeof(g_buf_count), &g_buf_count), "arg16");
	HANDLE_CLERROR(clSetKernelArg(crypt_kernel, 17, sizeof(g_buf_cstart), &g_buf_cstart), "arg17");
	HANDLE_CLERROR(clSetKernelArg(crypt_kernel, 18, sizeof(g_buf_chars0), &g_buf_chars0), "arg18");
	HANDLE_CLERROR(clSetKernelArg(crypt_kernel, 19, sizeof(g_buf_segs), &g_buf_segs), "arg19");
	HANDLE_CLERROR(clSetKernelArg(crypt_kernel, 20, sizeof(cl_uint), &g_nseg), "arg20");
	/* Odometer-shell bounds buffer (kernel arg 24, only present under
	 * -D GEN_ODOMETER). Args 21-23 (gbase/gcount/gR) are set per launch. */
	if (gen_deepen)
		HANDLE_CLERROR(clSetKernelArg(crypt_kernel, 24, sizeof(g_buf_boxbounds), &g_buf_boxbounds), "arg24");
}

static int gen_crypt(int *pcount, struct db_salt *salt)
{
	const int count = *pcount;
	size_t *lws = local_work_size ? &local_work_size : NULL;
	cl_ulong gbase = mask_gpu_cur_base;
	cl_uint gcount = count;
	size_t threads;

	(void)salt;
	if (count <= 0) {
		*pcount = 0;
		return 0;
	}

	gen_upload_tables();
	gen_upload_plan();

	g_cur_gen = 1;
	g_cur_gbase = gbase;
	g_cur_gcount = gcount;
	set_kernel_args_gen();
	HANDLE_CLERROR(clSetKernelArg(crypt_kernel, 21, sizeof(cl_ulong), &gbase), "arg21");
	HANDLE_CLERROR(clSetKernelArg(crypt_kernel, 22, sizeof(cl_uint), &gcount), "arg22");
	HANDLE_CLERROR(clSetKernelArg(crypt_kernel, 23, sizeof(cl_uint), &gen_R), "arg23");

	/* gcount candidates spread gen_R per work-item -> ceil(count/gen_R) threads. */
	threads = ((size_t)count + gen_R - 1) / gen_R;
	global_work_size = GET_NEXT_MULTIPLE(threads, local_work_size);

	BENCH_CLERROR(clEnqueueNDRangeKernel(queue[gpu_id], crypt_kernel, 1, NULL, &global_work_size, lws, 0, NULL, NULL), "failed in clEnqueueNDRangeKernel");
	BENCH_CLERROR(clEnqueueReadBuffer(queue[gpu_id], buffer_hash_ids, CL_TRUE, 0, sizeof(cl_uint), hash_ids, 0, NULL, NULL), "failed in reading back num cracked hashes.");

	if (hash_ids[0] > num_loaded_hashes) {
		fprintf(stderr, "Error, gen_crypt kernel.\n");
		error();
	}

	if (hash_ids[0]) {
		BENCH_CLERROR(clEnqueueReadBuffer(queue[gpu_id], buffer_return_hashes, CL_TRUE, 0, 2 * sizeof(cl_uint) * hash_ids[0], loaded_hashes, 0, NULL, NULL), "failed in reading back return_hashes.");
		BENCH_CLERROR(clEnqueueReadBuffer(queue[gpu_id], buffer_hash_ids, CL_TRUE, 0, (3 * hash_ids[0] + 1) * sizeof(cl_uint), hash_ids, 0, NULL, NULL), "failed in reading data back hash_ids.");
		BENCH_CLERROR(clEnqueueWriteBuffer(queue[gpu_id], buffer_bitmap_dupe, CL_TRUE, 0, (hash_table_size/32 + 1) * sizeof(cl_uint), zero_buffer, 0, NULL, NULL), "failed in clEnqueueWriteBuffer buffer_bitmap_dupe.");
		BENCH_CLERROR(clEnqueueWriteBuffer(queue[gpu_id], buffer_hash_ids, CL_TRUE, 0, sizeof(cl_uint), zero_buffer, 0, NULL, NULL), "failed in clEnqueueWriteBuffer buffer_hash_ids.");

		/* GPU mask generation tries candidates in K (Markov) order, but the device
		 * reports cracks in arbitrary atomic-fire order. Sort this batch by
		 * work-item id (== ascending global index == ascending K) so cracks are
		 * reported in exact K order, permuting the id triples and the parallel
		 * return-hash pairs (loaded_hashes) together. */
		if (mask_gpu_gen && hash_ids[0] > 1) {
			cl_uint a, n = hash_ids[0];

			for (a = 1; a < n; a++) {
				cl_uint g0 = hash_ids[1 + 3 * a];
				cl_uint g1 = hash_ids[2 + 3 * a];
				cl_uint g2 = hash_ids[3 + 3 * a];
				cl_uint h0 = loaded_hashes[2 * a];
				cl_uint h1 = loaded_hashes[2 * a + 1];
				int b = (int)a - 1;

				while (b >= 0 && hash_ids[1 + 3 * b] > g0) {
					hash_ids[1 + 3 * (b + 1)] = hash_ids[1 + 3 * b];
					hash_ids[2 + 3 * (b + 1)] = hash_ids[2 + 3 * b];
					hash_ids[3 + 3 * (b + 1)] = hash_ids[3 + 3 * b];
					loaded_hashes[2 * (b + 1)]     = loaded_hashes[2 * b];
					loaded_hashes[2 * (b + 1) + 1] = loaded_hashes[2 * b + 1];
					b--;
				}
				hash_ids[1 + 3 * (b + 1)] = g0;
				hash_ids[2 + 3 * (b + 1)] = g1;
				hash_ids[3 + 3 * (b + 1)] = g2;
				loaded_hashes[2 * (b + 1)]     = h0;
				loaded_hashes[2 * (b + 1) + 1] = h1;
			}
		}
	}

	*pcount *= mask_int_cand.num_int_cand;
	return hash_ids[0];
}

static int crypt_all(int *pcount, struct db_salt *salt)
{
	const int count = *pcount;

	size_t *lws = local_work_size ? &local_work_size : NULL;

	if (gen_active)
		return gen_crypt(pcount, salt);

	global_work_size = GET_NEXT_MULTIPLE(count, local_work_size);

	//fprintf(stderr, "%s(%d) lws "Zu" gws "Zu" idx %u int_cand%d\n", __FUNCTION__, count, local_work_size, global_work_size, key_idx, mask_int_cand.num_int_cand);

	// copy keys to the device
	if (key_idx)
		BENCH_CLERROR(clEnqueueWriteBuffer(queue[gpu_id], buffer_keys, CL_TRUE, 0, 4 * key_idx, saved_plain, 0, NULL, NULL), "failed in clEnqueueWriteBuffer buffer_keys.");

	BENCH_CLERROR(clEnqueueWriteBuffer(queue[gpu_id], buffer_idx, CL_TRUE, 0, 4 * global_work_size, saved_idx, 0, NULL, NULL), "failed in clEnqueueWriteBuffer buffer_idx.");

	if (!mask_gpu_is_static)
		BENCH_CLERROR(clEnqueueWriteBuffer(queue[gpu_id], buffer_int_key_loc, CL_TRUE, 0, 4 * global_work_size, saved_int_key_loc, 0, NULL, NULL), "failed in clEnqueueWriteBuffer buffer_int_key_loc.");

	if (salt != NULL && salt->count > 4500 &&
		(num_loaded_hashes - num_loaded_hashes / 10) > salt->count) {
		size_t old_ot_sz_bytes, old_ht_sz_bytes;
		prepare_table(salt);
		init_kernel(salt->count, select_bitmap(salt->count));

		BENCH_CLERROR(clGetMemObjectInfo(buffer_offset_table, CL_MEM_SIZE, sizeof(size_t), &old_ot_sz_bytes, NULL), "failed to query buffer_offset_table.");

		if (old_ot_sz_bytes < offset_table_size *
			sizeof(OFFSET_TABLE_WORD)) {
			BENCH_CLERROR(clReleaseMemObject(buffer_offset_table), "Error Releasing buffer_offset_table.");

			buffer_offset_table = clCreateBuffer(context[gpu_id], CL_MEM_READ_ONLY, offset_table_size * sizeof(OFFSET_TABLE_WORD), NULL, &ret_code);
			BENCH_CLERROR(ret_code, "Error creating buffer argument buffer_offset_table.");
		}

		BENCH_CLERROR(clGetMemObjectInfo(buffer_hash_table, CL_MEM_SIZE, sizeof(size_t), &old_ht_sz_bytes, NULL), "failed to query buffer_hash_table.");

		if (old_ht_sz_bytes < hash_table_size * sizeof(cl_uint) * 2) {
			BENCH_CLERROR(clReleaseMemObject(buffer_hash_table), "Error Releasing buffer_hash_table.");
			BENCH_CLERROR(clReleaseMemObject(buffer_bitmap_dupe), "Error Releasing buffer_bitmap_dupe.");
			MEM_FREE(zero_buffer);

			zero_buffer = (cl_uint *) mem_calloc(hash_table_size/32 + 1, sizeof(cl_uint));
			buffer_bitmap_dupe = clCreateBuffer(context[gpu_id], CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, (hash_table_size/32 + 1) * sizeof(cl_uint), zero_buffer, &ret_code);
			BENCH_CLERROR(ret_code, "Error creating buffer argument buffer_bitmap_dupe.");
			buffer_hash_table = clCreateBuffer(context[gpu_id], CL_MEM_READ_ONLY, hash_table_size * sizeof(cl_uint) * 2, NULL, &ret_code);
			BENCH_CLERROR(ret_code, "Error creating buffer argument buffer_hash_table.");
		}

		BENCH_CLERROR(clEnqueueWriteBuffer(queue[gpu_id], buffer_bitmaps, CL_TRUE, 0, (bitmap_size_bits >> 3), bitmaps, 0, NULL, NULL), "failed in clEnqueueWriteBuffer buffer_bitmaps.");
		BENCH_CLERROR(clEnqueueWriteBuffer(queue[gpu_id], buffer_offset_table, CL_TRUE, 0, sizeof(OFFSET_TABLE_WORD) * offset_table_size, offset_table, 0, NULL, NULL), "failed in clEnqueueWriteBuffer buffer_offset_table.");
		BENCH_CLERROR(clEnqueueWriteBuffer(queue[gpu_id], buffer_hash_table, CL_TRUE, 0, sizeof(cl_uint) * hash_table_size * 2, bt_hash_table_192, 0, NULL, NULL), "failed in clEnqueueWriteBuffer buffer_hash_table.");
		set_kernel_args();
		set_kernel_args_kpc();
	}

	BENCH_CLERROR(clEnqueueNDRangeKernel(queue[gpu_id], crypt_kernel, 1, NULL, &global_work_size, lws, 0, NULL, NULL), "failed in clEnqueueNDRangeKernel");

	BENCH_CLERROR(clEnqueueReadBuffer(queue[gpu_id], buffer_hash_ids, CL_TRUE, 0, sizeof(cl_uint), hash_ids, 0, NULL, NULL), "failed in reading back num cracked hashes.");

	if (hash_ids[0] > num_loaded_hashes) {
		fprintf(stderr, "Error, crypt_all kernel.\n");
		error();
	}

	if (hash_ids[0]) {
		BENCH_CLERROR(clEnqueueReadBuffer(queue[gpu_id], buffer_return_hashes, CL_TRUE, 0, 2 * sizeof(cl_uint) * hash_ids[0], loaded_hashes, 0, NULL, NULL), "failed in reading back return_hashes.");
		BENCH_CLERROR(clEnqueueReadBuffer(queue[gpu_id], buffer_hash_ids, CL_TRUE, 0, (3 * hash_ids[0] + 1) * sizeof(cl_uint), hash_ids, 0, NULL, NULL), "failed in reading data back hash_ids.");
		BENCH_CLERROR(clEnqueueWriteBuffer(queue[gpu_id], buffer_bitmap_dupe, CL_TRUE, 0, (hash_table_size/32 + 1) * sizeof(cl_uint), zero_buffer, 0, NULL, NULL), "failed in clEnqueueWriteBuffer buffer_bitmap_dupe.");
		BENCH_CLERROR(clEnqueueWriteBuffer(queue[gpu_id], buffer_hash_ids, CL_TRUE, 0, sizeof(cl_uint), zero_buffer, 0, NULL, NULL), "failed in clEnqueueWriteBuffer buffer_hash_ids.");
	}

	*pcount *=  mask_int_cand.num_int_cand;
	return hash_ids[0];
}

static int cmp_all(void *binary, int count)
{
	if (count) return 1;
	return 0;
}

static int cmp_one(void *binary, int index)
{
	return (((unsigned int*)binary)[0] ==
		bt_hash_table_192[hash_ids[3 + 3 * index]]);
}

static int cmp_exact(char *source, int index)
{
	unsigned int *t = (unsigned int *) get_binary(source);

	if (t[2] != loaded_hashes[2 * index])
		return 0;
	if (t[3] != loaded_hashes[2 * index + 1])
		return 0;
	return 1;
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

	/* GPU generation pushes no host keys, so the key-streaming tuner doesn't
	 * apply. Pick a fixed work size and prepare the (otherwise unused) kpc
	 * buffers so the arg-0..2 contract is still satisfied. */
	if (gen_active) {
		size_t maxlws;
		const char *renv = getenv("JOHN_GEN_R");
		const char *genv = getenv("JOHN_GEN_GWS");

		if (renv && atoi(renv) > 0)
			gen_R = atoi(renv);
		if (!local_work_size)
			local_work_size = 8;
		maxlws = get_kernel_max_lws(gpu_id, crypt_kernel);
		if (local_work_size > maxlws)
			local_work_size = maxlws;
		/* 1<<18 work-items keeps a modern GPU's SMs saturated; override with
		 * JOHN_GEN_GWS. */
		global_work_size = GET_NEXT_MULTIPLE(
		    (genv && atoi(genv) > 0) ? (size_t)atoi(genv) : (1 << 18),
		    local_work_size);

		release_clobj_kpc();
		create_clobj_kpc(global_work_size);
		set_kernel_args_kpc();
		/* Each of global_work_size work-items emits gen_R candidates, so a block
		 * (mask mode's per-launch candidate count) is that wide. The kpc key
		 * buffers stay sized to global_work_size - the gen kernel never reads
		 * them, and the NDRange only launches ceil(count/gen_R). */
		self->params.max_keys_per_crypt = global_work_size * gen_R;
		clear_keys();
		return;
	}

	memset(key, 0xF5, PLAINTEXT_LENGTH);
	key[PLAINTEXT_LENGTH] = 0;

	gws_limit = MIN((0xf << 22) * 4 / BUFSIZE,
			get_max_mem_alloc_size(gpu_id) / BUFSIZE);
	get_power_of_two(gws_limit);
	if (gws_limit > MIN((0xf << 22) * 4 / BUFSIZE,
		get_max_mem_alloc_size(gpu_id) / BUFSIZE))
		gws_limit >>= 1;

#if SIZEOF_SIZE_T > 4
	/* We can't process more than 4G keys per crypt() */
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

	/* Use the K-ordered GPU generator for native (non-stacked) mask mode. */
	mask_gpu_gen = !self_test_running &&
	               (options.flags & FLG_MASK_CHK) &&
	               !(options.flags & FLG_MASK_STACKED);

	/* In MASK_GPU_CPU validation the host streams candidates through the normal
	 * crypt path, so keep mask_gpu_gen set (mask.c drives the host unrank) but run
	 * the format as non-gen: normal sha1 kernel, crypt and get_key. */
	gen_active = mask_gpu_gen && !mask_gpu_cpu_validate;

	/* JOHN_GEN_REGS[=width] selects the register-resident gen kernel. */
	{
		const char *e = getenv("JOHN_GEN_REGS");
		int v = e ? atoi(e) : 0;

		gen_reg_max = e ? (cl_uint)(v >= 2 ? v : 16) : 0;
	}

	/* JOHN_GEN_HEAD=H (or JOHN_GEN_MARGINAL for H==0) selects the hybrid head/tail
	 * generator. It owns the materialize, so force off the regs/localtab variants. */
	{
		const char *he = getenv("JOHN_GEN_HEAD");

		if (he)
			gen_head = atoi(he);
		else if (getenv("JOHN_GEN_MARGINAL"))
			gen_head = 0;
		else
			gen_head = -1;
	}
	if (gen_head >= 0)
		gen_reg_max = 0;

	/* JOHN_GEN_DEEPEN selects the odometer-shell deepening generator. It owns the
	 * index->iter step (mixed-radix odometer) so it lives in the non-REGS path;
	 * force GEN_REGS off. mask.c reads the same env to build the matching plan. */
	gen_deepen = (getenv("JOHN_GEN_DEEPEN") != NULL);
	if (gen_deepen)
		gen_reg_max = 0;

	num_loaded_hashes = db->salts->count;
	prepare_table(db->salts);
	init_kernel(num_loaded_hashes, select_bitmap(num_loaded_hashes));

	create_clobj();
	set_kernel_args();

	HANDLE_CLERROR(clEnqueueWriteBuffer(queue[gpu_id], buffer_bitmaps, CL_TRUE, 0, (size_t)(bitmap_size_bits >> 3), bitmaps, 0, NULL, NULL), "failed in clEnqueueWriteBuffer buffer_bitmaps.");
	HANDLE_CLERROR(clEnqueueWriteBuffer(queue[gpu_id], buffer_offset_table, CL_TRUE, 0, sizeof(OFFSET_TABLE_WORD) * offset_table_size, offset_table, 0, NULL, NULL), "failed in clEnqueueWriteBuffer buffer_offset_table.");
	HANDLE_CLERROR(clEnqueueWriteBuffer(queue[gpu_id], buffer_hash_table, CL_TRUE, 0, sizeof(cl_uint) * hash_table_size * 2, bt_hash_table_192, 0, NULL, NULL), "failed in clEnqueueWriteBuffer buffer_hash_table.");

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
		FMT_CASE | FMT_8_BIT | FMT_SPLIT_UNIFIES_CASE | FMT_REMOVE | FMT_MASK,
		{ NULL },
		{ FORMAT_TAG, FORMAT_TAG_OLD },
		rawsha1_common_tests
	}, {
		init,
		done,
		reset,
		rawsha1_common_prepare,
		rawsha1_common_valid,
		rawsha1_common_split,
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
		cmp_all,
		cmp_one,
		cmp_exact
	}

};

#endif /* plugin stanza */

#endif /* HAVE_OPENCL */
