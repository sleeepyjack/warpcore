#include <warpcore/bloom_filter.cuh>

#include <nvbench/nvbench.cuh>

#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <cuda/std/bit>

#include <cstdint>
#include <vector>

template <typename Key, typename Word, nvbench::int32_t BlockBits>
void bloom_filter_insert(nvbench::state& state,
                         nvbench::type_list<Key,
                                            Word,
                                            nvbench::enum_type<BlockBits>>)
{
    static constexpr std::uint64_t seed = 42;

    const auto words_per_block = BlockBits / (sizeof(Word) * 8);
    if constexpr ((not cuda::std::has_single_bit(static_cast<uint32_t>(BlockBits))) or (words_per_block == 0)) {
        state.skip("Invalid filter block size");
    } else {
        const auto num_keys = state.get_int64("NumInputs");
        const auto pattern_bits_per_word = state.get_int64("PatternBitsPerWord");
        const auto pattern_bits = pattern_bits_per_word * words_per_block;
        const auto filter_size_mb = state.get_int64("FilterSizeMB");
        const auto filter_bits = filter_size_mb * 1024 * 1024 * 8;

        state.add_element_count(num_keys);
        state.collect_dram_throughput();
        state.collect_l2_hit_rates();

        using filter_t = warpcore::BloomFilter<
            Key,
            warpcore::hashers::MurmurHash<Key>,
            Word,
            words_per_block>;

        filter_t filter(filter_bits, pattern_bits, seed);

        thrust::device_vector<Key> keys(num_keys);
        thrust::sequence(thrust::device, keys.begin(), keys.end(), 0);

        state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
            filter.insert(keys.data().get(), num_keys, launch.get_stream());
        });
    }
}

template <typename Key, typename Word, nvbench::int32_t BlockBits>
void bloom_filter_retrieve(nvbench::state& state,
                           nvbench::type_list<Key,
                                              Word,
                                              nvbench::enum_type<BlockBits>>)
{
    static constexpr std::uint64_t seed = 42;

    const auto words_per_block = BlockBits / (sizeof(Word) * 8);
    if constexpr ((not cuda::std::has_single_bit(static_cast<uint32_t>(BlockBits))) or (words_per_block == 0)) {
        state.skip("Invalid filter block size");
    } else {
        const auto num_keys = state.get_int64("NumInputs");
        const auto pattern_bits_per_word = state.get_int64("PatternBitsPerWord");
        const auto pattern_bits = pattern_bits_per_word * words_per_block;
        const auto filter_size_mb = state.get_int64("FilterSizeMB");
        const auto filter_bits = filter_size_mb * 1024 * 1024 * 8;

        state.add_element_count(num_keys);
        state.collect_dram_throughput();
        state.collect_l2_hit_rates();

        using filter_t = warpcore::BloomFilter<
            Key,
            warpcore::hashers::MurmurHash<Key>,
            Word,
            words_per_block>;

        filter_t filter(filter_bits, pattern_bits, seed);

        
        // insert FPR-optimal number of keys
        auto const num_build_keys = (filter_size_mb * 1024 * 1024 * 8) / (2 * pattern_bits);
        thrust::device_vector<Key> keys(num_keys + num_build_keys);
        thrust::sequence(thrust::device, keys.begin(), keys.end(), 0);

        filter.insert(keys.data().get(), num_build_keys);

        // FPR summary
        thrust::device_vector<bool> result(num_keys, false);
        filter.retrieve(keys.data().get() + num_build_keys, num_keys, result.data().get());

        double const fp = thrust::count(thrust::device, result.begin(), result.end(), true);

        auto& summ_fpr = state.add_summary("FalsePositiveRate");
        summ_fpr.set_string("hint", "FPR");
        summ_fpr.set_string("short_name", "FPR");
        summ_fpr.set_string("description", "False-positive rate of the bloom filter.");
        summ_fpr.set_float64("value", fp / static_cast<double>(num_keys));

        auto& summ_k = state.add_summary("PatternBits");
        summ_k.set_string("hint", "K");
        summ_k.set_string("short_name", "K");
        summ_k.set_string("description", "Cardinality of a key's bit pattern.");
        summ_k.set_int64("value", pattern_bits);

        state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
            filter.retrieve(keys.data().get(), num_keys, result.data().get(), launch.get_stream());
        });
    }
}

// Specify parameter dimensions
auto constexpr max_noise = 3;
using key_t_list = nvbench::type_list<nvbench::uint64_t>;
using block_bits_list = nvbench::enum_type_list<32, 64, 128, 256, 512>;
auto const filter_size_mb_list =
  std::vector<nvbench::int64_t>{1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};
auto const pattern_bits_per_word_list =
  std::vector<nvbench::int64_t>{1, 20};
auto const num_inputs_list =
  std::vector<nvbench::int64_t>{1'000'000'000};


// Register benchmarks with parameter sweeps
NVBENCH_BENCH_TYPES(
    bloom_filter_insert,
    NVBENCH_TYPE_AXES(key_t_list, // Key
                      nvbench::type_list<nvbench::uint32_t>, // Word
                      block_bits_list)) // BlockBits
    .set_name("bloom_filter_add_u32")
    .set_type_axes_names({"Key", "Word", "BlockBits"})
    .set_max_noise(max_noise)
    .add_int64_axis("NumInputs", num_inputs_list)
    .add_int64_axis("FilterSizeMB", filter_size_mb_list)
    .add_int64_axis("PatternBitsPerWord", pattern_bits_per_word_list);

NVBENCH_BENCH_TYPES(
    bloom_filter_insert,
    NVBENCH_TYPE_AXES(key_t_list, // Key
                        nvbench::type_list<nvbench::uint64_t>, // Word
                        block_bits_list)) // BlockBits
    .set_name("bloom_filter_add_u64")
    .set_type_axes_names({"Key", "Word", "BlockBits"})
    .set_max_noise(max_noise)
    .add_int64_axis("NumInputs", num_inputs_list)
    .add_int64_axis("FilterSizeMB", filter_size_mb_list)
    .add_int64_axis("PatternBitsPerWord", pattern_bits_per_word_list);

NVBENCH_BENCH_TYPES(
    bloom_filter_retrieve,
    NVBENCH_TYPE_AXES(key_t_list, // Key
                      nvbench::type_list<nvbench::uint32_t>, // Word
                      block_bits_list)) // BlockBits
    .set_name("bloom_filter_retrieve_u32")
    .set_type_axes_names({"Key", "Word", "BlockBits"})
    .set_max_noise(max_noise)
    .add_int64_axis("NumInputs", num_inputs_list)
    .add_int64_axis("FilterSizeMB", filter_size_mb_list)
    .add_int64_axis("PatternBitsPerWord", pattern_bits_per_word_list);

NVBENCH_BENCH_TYPES(
    bloom_filter_retrieve,
    NVBENCH_TYPE_AXES(key_t_list, // Key
                        nvbench::type_list<nvbench::uint64_t>, // Word
                        block_bits_list)) // BlockBits
    .set_name("bloom_filter_retrieve_u64")
    .set_type_axes_names({"Key", "Word", "BlockBits"})
    .set_max_noise(max_noise)
    .add_int64_axis("NumInputs", num_inputs_list)
    .add_int64_axis("FilterSizeMB", filter_size_mb_list)
    .add_int64_axis("PatternBitsPerWord", pattern_bits_per_word_list);

NVBENCH_BENCH_TYPES(
    bloom_filter_retrieve,
    NVBENCH_TYPE_AXES(key_t_list, // Key
                        nvbench::type_list<nvbench::uint32_t, nvbench::uint64_t>, // Word
                        block_bits_list)) // BlockBits
    .set_name("bloom_filter_fpr")
    .set_type_axes_names({"Key", "Word", "BlockBits"})
    .set_max_noise(max_noise)
    .add_int64_axis("NumInputs", num_inputs_list)
    .add_int64_axis("FilterSizeMB", {1024})
    .add_int64_axis("PatternBitsPerWord", {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20});
