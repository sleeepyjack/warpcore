#include <iostream>
#include <algorithm>
#include <warpcore/warpcore.cuh>
#include <helpers/timers.cuh>

// This example implements a filtered histogram over a multi-set of keys
// using warpcore. The task is to output the counts of all distinct keys
// which occure more than once in the input data.
// We make use of the device-sided operations provided by warpcore to
// implement a custom CUDA kernel that does the job.

template<class Key, class BloomFilter, class HashTable>
__global__ void filter_and_count(
    Key* keys,
    uint64_t size_keys,
    BloomFilter bloom_filter,
    HashTable hash_table)
{
    using namespace warpcore;

    // global thread id
    const auto tid = helpers::global_thread_id();
    // cooperative group to use for hash table probing
    auto ht_group =
        cg::tiled_partition<hash_table.cg_size()>(cg::this_thread_block());
    // cooperative group to use for bloom filter
    auto bf_group =
        cg::tiled_partition<bloom_filter.cg_size()>(cg::this_thread_block());

    // register to store an input key
    Key key;
    // flag which specifies whether an input key is unique or not
    bool key_not_unique = false;
    // CG mask which shows which threads hold non-unique keys
    uint32_t not_unique_mask;

    // check for array bounds
    if(tid < size_keys)
    {
        // each thread loads its associated key into a register
        key = keys[tid];
        // next, we check if a key has already been seen (i.e. is NOT unique)
        // by checking against the bloom filter
        // NOTE this operation only works with a group size of 1
        key_not_unique = bloom_filter.insert_and_query(key, bf_group);
    }

    // compute the mask of all active threads (i.e. with non-unique keys)
    // in the cooperative group
    not_unique_mask = ht_group.ballot(key_not_unique);

    // for each non-unique key in the CG
    while(not_unique_mask)
    {
        // elect one thread as the leader
        const auto leader = __ffs(not_unique_mask)-1;
        // broadcast the key of the leader to all threads in the CG
        const auto filtered_key = ht_group.shfl(key, leader);
        // insert the key into the counting hash table using the CG
        hash_table.insert(filtered_key, ht_group);
        // remove the leader from the mask
        not_unique_mask ^= 1UL << leader;
    }
}

int main()
{
    using namespace warpcore;

    // key type (std::uint32_t or std::uint64_t)
    using key_t   = std::uint32_t;

    // count type of the hash table
    using count_t = std::uint64_t;

    // in this example we use 2^20 dummy keys as inputs where half of them
    // will be unique and the other half occurs more than once
    // (in this dummy example each non-unique key occurs 4 times)
    const uint64_t input_size = 1UL << 20;
    const uint64_t multiplicity = 4;

    // allocate host-sided (pinned) array for inputs
    key_t* keys_in_h; cudaMallocHost(&keys_in_h, sizeof(key_t)*input_size);

    // allocate device-sided array for inputs
    key_t* keys_in_d; cudaMalloc(&keys_in_d, sizeof(key_t)*input_size);

    // first, lets generate input_size/2 unique keys
    for(key_t i = 0; i < input_size/2; i++)
    {
        keys_in_h[i] = i+1;
    }

    // second, generate input_size/2 non-unique keys where each key
    // has a multiplicity of 4
    for(key_t i = input_size/2; i < input_size; i += multiplicity)
    {
        for(key_t j = 0; j < multiplicity; j++)
        {
            keys_in_h[i+j] = i+1;
        }
    }

    // randomly permute the input data
    std::random_shuffle(keys_in_h, keys_in_h+input_size);

    // subsequently copy the input keys to the CUDA device
    cudaMemcpy(keys_in_d, keys_in_h, sizeof(key_t)*input_size, cudaMemcpyHostToDevice);

    // The bloom filter gets an additional random seed 420 and should use
    // warpcore::hashers:MuellerHash as its dedicated hash function.
    using bloom_filter_t = BloomFilter<
        key_t, // key type
        hashers::MuellerHash,
        key_t>; // hash function

    // The counting hash table should use
    // warpcore::probingschemes::QuadraticProbing as its dedicated probing
    // scheme together with the warpcore::hashers:MurmurHash hash function.
    // The empty key and tombstone (deleted) key specifiers may not be contained
    // in our input data and are set to key_t(0) and key_t(0)-1 respectively.
    // Since the first occurrence of a non-unique key will not be inserted in
    // our hash table (since it is seen as unique at this first occurrence)
    // we can set the initial count value of each entry to 2, so we get
    // correct counting results
    using hash_table_t = CountingHashTable<
        key_t, // key type
        count_t, //value type
        key_t(0), // empty key
        key_t(0)-1,  // tombstone key
        probing_schemes::QuadraticProbing<hashers::MurmurHash<key_t>, 8>>;

    // we initialize the bloom filter with a sufficient amount of bits
    // in order to achieve a low false positive rate and set the number of
    // hash functions to be used
    const uint64_t num_bits = 1UL << 29;
    const uint8_t k = 6;

    // INITIALIZE the bloom filter
    bloom_filter_t bloom_filter = bloom_filter_t(num_bits, k);

    // since half of the inputs should be filtered out by the bloom filter
    // and the other half has a multiplicity of 4, we can safely set the
    // capacity of the counting hash table to be
    const uint64_t capacity = input_size/4;

    // INITIALIZE the counting hash table
    hash_table_t hash_table(capacity);

    // INSERT the input keys using our custom filter/count kernel

    helpers::GpuTimer timer("count");

    filter_and_count
    <key_t, bloom_filter_t, hash_table_t>
    <<<SDIV(input_size, 1024), 1024>>>
    (keys_in_d, input_size, bloom_filter, hash_table);

    timer.print();

    cudaDeviceSynchronize(); CUERR

    // allocate host-sided (pinned) arrays for output
    key_t*   keys_out_h;   cudaMallocHost(&keys_out_h,   sizeof(key_t)*input_size);
    count_t* counts_out_h; cudaMallocHost(&counts_out_h, sizeof(count_t)*input_size);
    uint64_t output_size_h;

    // allocate device-sided (pinned) arrays for output
    key_t*    keys_out_d;    cudaMalloc(&keys_out_d,    sizeof(key_t)*input_size);
    count_t*  counts_out_d;  cudaMalloc(&counts_out_d,  sizeof(count_t)*input_size);

    // we can retrieve all counts at once as follows
    hash_table.retrieve_all(keys_out_d, counts_out_d, output_size_h);

    // copy the results to the host
    cudaMemcpy(keys_out_h, keys_out_d, sizeof(key_t)*output_size_h, cudaMemcpyDeviceToHost);
    cudaMemcpy(counts_out_h, counts_out_d, sizeof(count_t)*output_size_h, cudaMemcpyDeviceToHost);

    // let's count the number of unwanted keys (multiplicity != 4)
    std::uint64_t num_unwanted_keys = 0;
    for(std::uint64_t i = 0; i < output_size_h; ++i)
    {
        // multiplicity is 4 but one element per key is filtered out
        if(counts_out_h[i] != 3)
        {
            num_unwanted_keys++;
        }
    }
    std::cout << "number of unwanted keys: " << num_unwanted_keys << std::endl;

    // check for any errors
    std::cout << "hash table errors: " << hash_table.pop_status().get_errors() << std::endl;

    // free all allocated recources
    cudaFreeHost(keys_in_h);
    cudaFree(keys_in_d);
    cudaFreeHost(keys_out_h);
    cudaFree(keys_out_d);
    cudaFreeHost(counts_out_h);
    cudaFree(counts_out_d);

    // check for any CUDA errors
    cudaDeviceSynchronize(); CUERR
}
