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

/* GPU K-ordered generation state */
static cl_mem g_buf_suf, g_buf_table, g_buf_startv, g_buf_rowcnt, g_buf_littmpl,
              g_buf_keypos, g_buf_count, g_buf_cstart, g_buf_chars0, g_buf_segs;
static unsigned g_uploaded_serial;       /* mask_gpu_serial of last table upload  */
static unsigned g_uploaded_plan_serial;  /* mask_gpu_serial of last plan upload    */
static cl_uint g_nseg;                    /* segments in the uploaded plan          */

/* Device mirror of one mask_gpu_plan segment (must match gen_seg in the kernel).
 * Layout: four ulongs then four uints = 48 bytes, naturally 8-aligned. */
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
/* Candidates generated per work-item in the gen kernel. Each launch covers
 * global_work_size * gen_R candidates, restoring the work-per-launch the old
 * internal-mask kernel got from its NUM_INT_KEYS inner loop (without this every
 * work-item did a single hash, so a fast GPU was starved by launch overhead).
 * Tunable via JOHN_GEN_R for a given device. */
static cl_uint gen_R = 256;
static void gen_release_all(void);

/* Parameters of the gen launch currently in flight. Stored so set_kernel_args()
 * can re-bind the gen-kernel args (10-23) if ocl_hc_128_extract_info rebuilds
 * (releases+recreates) crypt_kernel mid-launch when the hash table shrinks -
 * a rebuild wipes all kernel args and the generic re-bind only covers 0-9. */
static int      g_cur_gen = 0;       /* a gen launch is in flight */
static cl_ulong g_cur_gbase;
static cl_uint  g_cur_gcount;
static void set_kernel_args_gen(void);

/* True when the GPU actually generates candidates. This is mask_gpu_gen EXCEPT in
 * MASK_GPU_CPU validation, where mask mode streams host-materialized candidates
 * through the normal crypt path - so the gen kernel/crypt/get_key must stay off.
 * Set in reset() before the kernel is built. */
static int gen_active = 0;

static unsigned int shift64_ht_sz, shift64_ot_sz;

static unsigned int key_idx = 0;
static struct fmt_main *self;

#define MIN_KEYS_PER_CRYPT      1
#define MAX_KEYS_PER_CRYPT      1


static struct fmt_tests tests[] = {
	{"5a105e8b9d40e1329780d62ea2265d8a", "test1"},
	{FORMAT_TAG "5a105e8b9d40e1329780d62ea2265d8a", "test1"},
	{"098f6bcd4621d373cade4e832627b4f6", "test"},
	{FORMAT_TAG "378e2c4a07968da2eca692320136433d", "thatsworking"},
	{FORMAT_TAG "8ad8757baa8564dc136c1e07507f4a98", "test3"},
	{"d41d8cd98f00b204e9800998ecf8427e", ""},
#ifdef DEBUG
	{FORMAT_TAG "c9ccf168914a1bcfc3229f1948e67da0","1234567890123456789012345678901234567890123456789012345"},
#if PLAINTEXT_LENGTH >= 80
	{FORMAT_TAG "57edf4a22be3c955ac49da2e2107b67a","12345678901234567890123456789012345678901234567890123456789012345678901234567890"},
#endif
#endif
	{"{MD5}CY9rzUYh03PK3k6DJie09g==", "test"},
	{NULL}
};

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

	/* If a gen launch is in flight and the kernel was just rebuilt by
	 * ocl_hc_128_extract_info, re-bind the gen args (10-23) too. */
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
	cl_uint dummy = 0;

	release_clobj();

	//dummy is used as dummy parameter
	buffer_int_keys = clCreateBuffer(context[gpu_id], CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 4 * mask_int_cand.num_int_cand, mask_int_cand.int_cand ? mask_int_cand.int_cand : (void *)&dummy, &ret_code);
	HANDLE_CLERROR(ret_code, "Error creating buffer argument buffer_int_keys.");

	ocl_hc_128_crobj(crypt_kernel);
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
	if (buffer_int_keys) {
		HANDLE_CLERROR(clReleaseMemObject(buffer_int_keys), "Error Releasing buffer_int_keys.");
		buffer_int_keys = 0;

		ocl_hc_128_rlobj();
	}
}

static void done(void)
{
	gen_release_all();
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

	if (gen_active)
		strcat(build_opts, " -D GPU_GEN");

	opencl_build_kernel("$JOHN/opencl/md5_kernel.cl", gpu_id, build_opts, 0);
	crypt_kernel = clCreateKernel(program[gpu_id],
	                              gen_active ? "md5_gen" : "md5", &ret_code);
	HANDLE_CLERROR(ret_code, "Error creating kernel. Double-check kernel name?");
}

static void init(struct fmt_main *_self)
{
	self = _self;
	ocl_hc_num_loaded_hashes = 0;

	ocl_hc_128_init(_self);

	opencl_prepare_dev(gpu_id);
	mask_int_cand_target = opencl_speed_index(gpu_id) / 300;
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
		if (ocl_hc_hash_ids == NULL || ocl_hc_hash_ids[0] == 0 ||
		    index >= ocl_hc_hash_ids[0] ||
		    ocl_hc_hash_ids[0] > ocl_hc_num_loaded_hashes)
			off = (uint64_t)index;
		else
			off = (uint64_t)ocl_hc_hash_ids[1 + 3 * index] * gen_R +
			      ocl_hc_hash_ids[2 + 3 * index];

		mask_gpu_virt_to_loop(mask_gpu_cur_base + off, &loop, &g);
		mask_gpu_unrank_key(loop, g, out, &kl);
		return out;
	}

	if (ocl_hc_hash_ids == NULL || ocl_hc_hash_ids[0] == 0 ||
	    index >= ocl_hc_hash_ids[0] || ocl_hc_hash_ids[0] > ocl_hc_num_loaded_hashes) {
		t = index;
		int_index = 0;
	}
	else  {
		t = ocl_hc_hash_ids[1 + 3 * index];
		int_index = ocl_hc_hash_ids[2 + 3 * index];

	}

	if (t >= global_work_size) {
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

	g_buf_table   = gen_copy_buf(npos * 256 * 256, t->table);
	g_buf_startv  = gen_copy_buf(npos * 256, t->startv);
	g_buf_rowcnt  = gen_copy_buf(npos * 256, t->rowcnt);
	g_buf_littmpl = gen_copy_buf(t->littmpl_len, t->littmpl);
	g_buf_keypos  = gen_copy_buf(npos * sizeof(cl_int), t->keypos);
	g_buf_count   = gen_copy_buf(npos * sizeof(cl_int), t->count);
	g_buf_cstart  = gen_copy_buf(npos, t->cstart);
	g_buf_chars0  = gen_copy_buf(npos, t->chars0);

	g_uploaded_serial = mask_gpu_serial;
}

/* Upload the multi-length plan: the per-segment metadata array and the
 * concatenated suffix-DP tables (all active length-loops). Changes per mask
 * config, tracked by serial. */
static void gen_upload_plan(void)
{
	const mask_gpu_plan *plan = mask_gpu_get_plan();
	gen_seg *segs;
	cl_ulong *suf;
	int s;

	if (g_buf_segs && g_uploaded_plan_serial == mask_gpu_serial)
		return;

	gen_release_buf(&g_buf_segs);
	gen_release_buf(&g_buf_suf);

	/* Concatenated suf: each loop's (limit+1)*ksize ulongs at suf_off[s]. */
	suf = mem_alloc((plan->suf_total ? plan->suf_total : 1) * sizeof(cl_ulong));
	segs = mem_alloc((plan->nseg ? plan->nseg : 1) * sizeof(gen_seg));
	for (s = 0; s < plan->nseg; s++) {
		const mask_gpu_loop *gl = mask_gpu_get_loop(plan->loop[s]);

		memcpy(suf + plan->suf_off[s], gl->suf,
		       (size_t)(gl->limit + 1) * gl->ksize * sizeof(cl_ulong));
		segs[s].vbase   = plan->vbase[s];
		segs[s].vcnt    = plan->vcnt[s];
		segs[s].lstart  = plan->lstart[s];
		segs[s].suf_off = plan->suf_off[s];
		segs[s].limit   = plan->limit[s];
		segs[s].len     = plan->len[s];
		segs[s].max_k   = plan->max_k[s];
		segs[s].ksize   = plan->ksize[s];
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
}

static int gen_crypt(int *pcount, struct db_salt *salt)
{
	const int count = *pcount;
	size_t *lws = local_work_size ? &local_work_size : NULL;
	cl_ulong gbase = mask_gpu_cur_base;
	cl_uint gcount = count;

	if (count <= 0) {
		*pcount = 0;
		return 0;
	}

	gen_upload_tables();
	gen_upload_plan();
	/* Remember this launch so set_kernel_args() can re-bind the gen args if
	 * extract_info rebuilds the kernel mid-launch. */
	g_cur_gen = 1;
	g_cur_gbase = gbase;
	g_cur_gcount = gcount;
	set_kernel_args_gen();
	HANDLE_CLERROR(clSetKernelArg(crypt_kernel, 21, sizeof(cl_ulong), &gbase), "arg21");
	HANDLE_CLERROR(clSetKernelArg(crypt_kernel, 22, sizeof(cl_uint), &gcount), "arg22");
	HANDLE_CLERROR(clSetKernelArg(crypt_kernel, 23, sizeof(cl_uint), &gen_R), "arg23");

	/* gcount candidates spread gen_R per work-item -> ceil(count/gen_R) threads. */
	{
		size_t threads = ((size_t)count + gen_R - 1) / gen_R;
		global_work_size = GET_NEXT_MULTIPLE(threads, local_work_size);
	}

	return ocl_hc_128_extract_info(salt, set_kernel_args, set_kernel_args_kpc,
	                               init_kernel, global_work_size, lws, pcount);
}

static int crypt_all(int *pcount, struct db_salt *salt)
{
	const int count = *pcount;

	size_t *lws = local_work_size ? &local_work_size : NULL;

	if (gen_active)
		return gen_crypt(pcount, salt);

	global_work_size = GET_NEXT_MULTIPLE(count, local_work_size);

	// copy keys to the device
	if (key_idx)
		BENCH_CLERROR(clEnqueueWriteBuffer(queue[gpu_id], buffer_keys, CL_TRUE, 0, 4 * key_idx, saved_plain, 0, NULL, NULL), "failed in clEnqueueWriteBuffer buffer_keys.");

	BENCH_CLERROR(clEnqueueWriteBuffer(queue[gpu_id], buffer_idx, CL_TRUE, 0, 4 * global_work_size, saved_idx, 0, NULL, NULL), "failed in clEnqueueWriteBuffer buffer_idx.");

	if (!mask_gpu_is_static)
		BENCH_CLERROR(clEnqueueWriteBuffer(queue[gpu_id], buffer_int_key_loc, CL_TRUE, 0, 4 * global_work_size, saved_int_key_loc, 0, NULL, NULL), "failed in clEnqueueWriteBuffer buffer_int_key_loc.");

	return ocl_hc_128_extract_info(salt, set_kernel_args, set_kernel_args_kpc, init_kernel, global_work_size, lws, pcount);
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
		/* 1<<18 work-items keeps a modern GPU's SMs saturated for the
		 * register-resident gen kernel; override with JOHN_GEN_GWS. */
		global_work_size = GET_NEXT_MULTIPLE(
		    (genv && atoi(genv) > 0) ? (size_t)atoi(genv) : (1 << 18),
		    local_work_size);

		release_clobj_kpc();
		create_clobj_kpc(global_work_size);
		set_kernel_args_kpc();
		/* Each of the global_work_size work-items emits gen_R candidates, so a
		 * block (and thus mask mode's per-launch candidate count) is that wide.
		 * The kpc key buffers stay sized to global_work_size - the gen kernel
		 * never reads them, and the NDRange only launches ceil(count/gen_R). */
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
		/* The kpc buffers were just (re)allocated for 'count' entries, but
		 * global_work_size may still hold a larger value from the gws-tuning
		 * crypt above; clear_keys() memsets sizeof(cl_uint) * global_work_size,
		 * so sync it to the allocation to avoid overflowing saved_idx. */
		global_work_size = count;
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
	 * the format as non-gen: normal md5 kernel, crypt and get_key. */
	gen_active = mask_gpu_gen && !mask_gpu_cpu_validate;

	ocl_hc_num_loaded_hashes = db->salts->count;
	ocl_hc_128_prepare_table(db->salts);
	init_kernel(ocl_hc_num_loaded_hashes, ocl_hc_128_select_bitmap(ocl_hc_num_loaded_hashes));

	create_clobj();
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
