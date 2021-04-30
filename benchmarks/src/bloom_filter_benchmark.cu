#include <iostream>
#include <warpcore/warpcore.cuh>
#include <helpers/timers.cuh>

double binom(
    std::uint64_t n,
    std::uint64_t k,
    double p)
{
    double res = 1.0;

    for(std::uint64_t i = n - k + 1; i <= n; ++i)
    {
        res = res * i;
    }

    for(std::uint64_t i = 1; i <= k; ++i)
    {
        res = res / i;
    }

    res = res * pow(p, k) * pow(1.0 - p, n - k);

    return res;
}

double fpr_std(
    std::uint64_t m,
    std::uint64_t n,
    std::uint64_t k)
{
    return std::pow(1.0-std::pow(1.0-1.0/m, n*k), k);
}

// according to http://algo2.iti.kit.edu/documents/cacheefficientbloomfilters-jea.pdf
double fpr_block(
    std::uint64_t m,
    std::uint64_t n,
    std::uint64_t k,
    std::uint64_t b)
{
    double res = 0.0;

    #pragma omp parallel for reduction(+:res)
    for(std::uint64_t i = 0; i < 5*n/b; ++i)
    {
        res += binom(n, i, 1.0/double(b)) * fpr_std(m/b, i, k);
    }

    return res;
}

int main(int argc, char *argv[])
{
    using data_t   = std::uint64_t;
    using slot_t = std::uint64_t;
    using index_t = std::uint64_t;
    using hasher_t = warpcore::hashers::MurmurHash<data_t>;

    static constexpr std::uint64_t seed = 42;
    std::uint64_t n = 1ULL << 26;
    std::uint64_t m = 1ULL << 33;
    std::uint64_t k = 6;
    static constexpr std::uint64_t cg_size = 1;
    static constexpr std::uint64_t block_bits = sizeof(slot_t)*8*cg_size;

    if(argc > 1)
    {
        if(argc != 4)
        {
            std::cerr << "invalid argument" << std::endl;
            return -1;
        }

        n = std::stoull(argv[1]);
        m = std::stoull(argv[2]);
        k = std::stoull(argv[3]);
    }

    using filter_t = warpcore::BloomFilter<
        data_t,
        hasher_t,
        slot_t,
        cg_size>;

    filter_t filter(m, k, seed);

    data_t* tp_data_h; cudaMallocHost(&tp_data_h, sizeof(data_t)*n); CUERR
    data_t* fp_data_h; cudaMallocHost(&fp_data_h, sizeof(data_t)*n); CUERR
    data_t* tp_data_d; cudaMalloc(&tp_data_d, sizeof(data_t)*n); CUERR
    data_t* fp_data_d; cudaMalloc(&fp_data_d, sizeof(data_t)*n); CUERR

    bool* flags_h; cudaMallocHost(&flags_h, sizeof(bool)*n); CUERR
    bool* flags_d; cudaMalloc(&flags_d, sizeof(bool)*n); CUERR

    #pragma omp parallel for
    for(index_t i = 0; i < n; i++)
    {
        tp_data_h[i] = i+1;
        fp_data_h[i] = n+i+1;
    }

    cudaMemcpy(tp_data_d, tp_data_h, sizeof(data_t)*n, H2D); CUERR
    cudaMemcpy(fp_data_d, fp_data_h, sizeof(data_t)*n, H2D); CUERR
    cudaMemset(flags_d, 0, sizeof(bool)*n); CUERR

    std::cout
    << "n=" << n
    << "\tm=" << m
    << "\tk=" << k
    << "\tcg=" << cg_size << std::endl;

    {
        helpers::GpuTimer timer("insert");
        filter.insert(tp_data_d, n);
    }

    {
        helpers::GpuTimer timer("retrieve_tp");
        filter.retrieve(tp_data_d, n, flags_d);
    }

    cudaMemcpy(flags_h, flags_d, sizeof(bool)*n, D2H); CUERR

    std::uint64_t tp = 0;
    #pragma omp parallel for reduction(+:tp)
    for(index_t i = 0; i < n; i++)
    {
        if(flags_h[i])
        {
            tp++;
        }
    }

    cudaMemset(flags_d, 0, sizeof(bool)*n); CUERR

    {
        helpers::GpuTimer timer("retrieve_fp");
        filter.retrieve(fp_data_d, n, flags_d);
    }

    cudaMemcpy(flags_h, flags_d, sizeof(bool)*n, D2H); CUERR

    std::uint64_t fp = 0;
    #pragma omp parallel for reduction(+:fp)
    for(index_t i = 0; i < n; i++)
    {
        if(flags_h[i])
        {
            fp++;
        }
    }

    std::string test_result = test_result = (tp == n) ? "pass" : "fail";
    test_result += " (" + std::to_string(tp) + "/" + std::to_string(n) + ")";
    std::cout << "test: " << test_result << std::endl;

    std::cout
    << "fpr: true=" << float(fp)/float(n)
    << " (" << std::to_string(fp) << "/" << std::to_string(n) << ")"
    << "\tstd=" << fpr_std(m, n, k)
    << "\tblock=" << fpr_block(m, n, k, m/block_bits)
    << "\tmember=" << filter.fpr(n) << std::endl << std::endl;

    cudaFreeHost(tp_data_h);
    cudaFree(tp_data_d);
    cudaFreeHost(fp_data_h);
    cudaFree(fp_data_d);
    cudaFreeHost(flags_h);
    cudaFree(flags_d);

    cudaDeviceSynchronize(); CUERR
}
