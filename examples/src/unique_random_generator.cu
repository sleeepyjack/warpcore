#include <iostream>
#include <numeric>
#include <warpcore.cuh>
#include <kiss_rng.cuh>
#include <random_distributions.cuh>

// This example shows the easy generation of gigabytes of unique random values
// in only a few milliseconds using WarpCore's BloomFilter.

/*! \brief checks if input values are unique
* \tparam T input data type
* \param[in] in_d device pointer to the input values
* \param[in] n number of input values
* \return true iff values are unique
*/
template<class T>
HOSTQUALIFIER INLINEQUALIFIER
bool check_unique(T * in_d, std::uint64_t n) noexcept
{
    // allocate memory for the sorted values
    T * out_d = nullptr;
    cudaMalloc(&out_d, sizeof(T)*n); CUERR

    // sort the generated values using CUB
    void * tmp_d = nullptr;
    std::size_t tmp_bytes = 0;
    cub::DeviceRadixSort::SortKeys(tmp_d, tmp_bytes, in_d, out_d, n); CUERR
    cudaMalloc(&tmp_d, tmp_bytes); CUERR
    cub::DeviceRadixSort::SortKeys(tmp_d, tmp_bytes, in_d, out_d, n); CUERR

    // allocate memory for the result
    bool unique_h;
    bool * unique_d = nullptr;
    cudaMalloc(&unique_d, sizeof(bool)); CUERR
    cudaMemset(unique_d, 1, sizeof(bool)); CUERR

    // check if neighbouring values are equal
    lambda_kernel
    <<<SDIV(n, MAXBLOCKSIZE), MAXBLOCKSIZE>>>
    ([=] DEVICEQUALIFIER
    {
        // determine the global thread ID
        const std::uint64_t tid = blockDim.x * blockIdx.x + threadIdx.x;

        // if neighbouring elements are equal throw error
        if(tid < n-1 && out_d[tid] == out_d[tid+1])
        {
            *unique_d = false;
        }
    }
    ); CUERR

    // copy the result to host
    cudaMemcpy(&unique_h, unique_d, sizeof(bool), D2H); CUERR

    // free any allocated memory
    cudaFree(out_d); CUERR
    cudaFree(tmp_d); CUERR
    cudaFree(unique_d); CUERR

    return unique_h;
}

int main()
{
    // define the data types to be generated
    using data_t = std::uint64_t;
    using index_t = std::uint64_t;
    // defined in util/kiss_rng.cuh
    using rng_t = Kiss<std::uint64_t>;

    // number of unique random values to generate
    static constexpr index_t n = 1UL << 28;
    // random seed
    static constexpr index_t seed = 42;

    // allocate GPU memory for the result
    data_t * data_d = nullptr;
    cudaMalloc(&data_d, sizeof(data_t)*n); CUERR

    // generate the values and measure throughput
    THROUGHPUTSTART(generate)
    // defined in util/random_distributions.cuh
    unique_distribution<data_t, rng_t>(data_d, n, seed); CUERR
    THROUGHPUTSTOP(generate, sizeof(data_t), n)

    // check if the generated values are unique
    TIMERSTART(test)
    std::cout << "TEST PASSED: " << std::boolalpha << check_unique(data_d, n) << std::endl;
    TIMERSTOP(test)

    // free any allocated memory
    cudaFree(data_d); CUERR
}