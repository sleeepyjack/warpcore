#ifndef WARPCORE_PROBING_SCHEMES_CUH
#define WARPCORE_PROBING_SCHEMES_CUH

#include "config.cuh"
#include "hashers.cuh"

namespace warpcore 
{

/*! \brief probing scheme iterators
 */
namespace probing_schemes
{

namespace cg = cooperative_groups;
namespace checks = warpcore::checks;

//TODO add inner (warp-level) probing?

/*! \brief double hashing scheme: \f$hash(k,i) = h_1(k)+i\cdot h_2(k)\f$
 * \tparam Hasher1 first hash function
 * \tparam Hasher1 second hash function
 * \tparam CGSize cooperative group size
 */
template <class Hasher1, class Hasher2, index_t CGSize = 1>
class DoubleHashing
{
    static_assert(
       checks::is_valid_cg_size(CGSize),
        "invalid cooperative group size");

    static_assert(
        std::is_same<
            typename Hasher1::key_type, 
            typename Hasher2::key_type>::value,
        "key types of both hashers must be the same");

public:
    using key_type = typename Hasher1::key_type;
    using index_type = index_t;
    using tag = tags::probing_scheme_tag;

    /*! \brief get cooperative group size
     * \return cooperative group size
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr index_type cg_size() noexcept { return CGSize; }

    /*! \brief constructor
     * \param[in] capacity capacity of the underlying hash table
     * \param[in] probing_length number of probing attempts
     * \param[in] group cooperative group
     */
    DEVICEQUALIFIER INLINEQUALIFIER
    explicit DoubleHashing(
        index_type capacity,
        index_type probing_length,
        const cg::thread_block_tile<CGSize>& group) :
        capacity_(capacity),
        probing_length_(SDIV(probing_length, group.size()) * group.size()),
        group_(group)
    {}

    template<class T>
    DEVICEQUALIFIER INLINEQUALIFIER
    index_type begin(T, T) = delete;

    template<class T>
    DEVICEQUALIFIER INLINEQUALIFIER
    index_type begin(T) = delete;

    /*! \brief begin probing sequence
     * \param[in] key key to be probed
     * \param[in] seed random seed
     * \return initial probing index for \c key
     */
    DEVICEQUALIFIER INLINEQUALIFIER
    index_type begin(key_type key, key_type seed = 0) noexcept 
    {
        i_ = 0;
        base1_ = Hasher1::hash(key + seed) + group_.thread_rank();
        base2_ = Hasher2::hash(key + seed);
        return (base1_ % capacity_);
    }

    /*! \brief next probing index for \c key
     * \return next probing index for \c key
     */
    DEVICEQUALIFIER INLINEQUALIFIER
    index_type next() noexcept 
    {
        i_ += CGSize;
        return (i_ + group_.thread_rank() < probing_length_) ?
            ((base1_ + i_ * base2_) % capacity_) : end();
    }

    /*! \brief end specifier of probing sequence
     * \return end specifier
     */
    DEVICEQUALIFIER INLINEQUALIFIER
    static constexpr index_type end() noexcept 
    {
        return ~index_type(0);
    }

private:
    const index_type capacity_; //< capacity of the underlying hash table
    const index_type probing_length_; //< number of probing attempts
    const cg::thread_block_tile<CGSize>& group_; //< cooperative group

    index_type i_; //< current probing position
    index_type base1_; //< \f$h_1(k)\f$
    index_type base2_; //< \f$h_2(k)\f$

}; // class DoubleHashing

/*! \brief linear probing scheme: \f$hash(k,i) = h(k)+i\f$
 * \tparam Hasher hash function
 * \tparam CGSize cooperative group size
 */
template <class Hasher, index_t CGSize = 1>
class LinearProbing 
{
    static_assert(
        checks::is_valid_cg_size(CGSize),
        "invalid cooperative group size");

public:
    using key_type = typename Hasher::key_type;
    using index_type = index_t;
    using tag = tags::probing_scheme_tag;

    /*! \brief get cooperative group size
     * \return cooperative group size
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr index_t cg_size() noexcept { return CGSize; }

    /*! \brief constructor
     * \param[in] capacity capacity of the underlying hash table
     * \param[in] probing_length number of probing attempts
     * \param[in] group cooperative group
     */
    DEVICEQUALIFIER
    explicit LinearProbing(
        index_type capacity,
        index_type probing_length,
        const cg::thread_block_tile<CGSize>& group) :
        capacity_(capacity),
        probing_length_(SDIV(probing_length, group.size()) * group.size()),
        group_(group)
    {}

    template<class T>
    DEVICEQUALIFIER INLINEQUALIFIER
    index_type begin(T, T) = delete;

    template<class T>
    DEVICEQUALIFIER INLINEQUALIFIER
    index_type begin(T) = delete;

    /*! \brief begin probing sequence
     * \param[in] key key to be probed
     * \param[in] seed random seed
     * \return initial probing index for \c key
     */
    DEVICEQUALIFIER INLINEQUALIFIER
    index_type begin(key_type key, key_type seed = 0) noexcept 
    {
        i_ = 0;
        base_ = Hasher::hash(key + seed) + group_.thread_rank();

        return (base_ % capacity_);
    }

    /*! \brief next probing index for \c key
     * \return next probing index
     */
    DEVICEQUALIFIER INLINEQUALIFIER
    index_type next() noexcept 
    {
        i_ += CGSize;
        return (i_ + group_.thread_rank() < probing_length_) ?
            ((base_ + i_) % capacity_) : end();
    }

    /*! \brief end specifier of probing sequence
     * \return end specifier
     */
    DEVICEQUALIFIER INLINEQUALIFIER
    static constexpr index_type end() noexcept 
    {
        return ~index_type(0);
    }

private:
    const index_type capacity_; //< capacity of the underlying hash table
    const index_type probing_length_; //< number of probing attempts
    const cg::thread_block_tile<CGSize>& group_; //< cooperative group

    index_type i_; //< current probing position
    index_type base_; //< \f$h(k)\f$

}; // class LinearProbing

/*! \brief quadratic probing scheme: \f$hash(k,i) = h(k)+i^2\f$
 * \tparam Hasher hash function
 * \tparam CGSize cooperative group size
 */
template <class Hasher, index_t CGSize = 1>
class QuadraticProbing 
{
    static_assert(
        checks::is_valid_cg_size(CGSize),
        "invalid cooperative group size");

public:
    using key_type = typename Hasher::key_type;
    using index_type = index_t;
    using tag = tags::probing_scheme_tag;

    /*! \brief get cooperative group size
     * \return cooperative group size
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr index_type cg_size() noexcept { return CGSize; }

    /*! \brief constructor
     * \param[in] capacity capacity of the underlying hash table
     * \param[in] probing_length number of probing attempts
     * \param[in] group cooperative group
     */
    DEVICEQUALIFIER
    explicit QuadraticProbing(
        index_type capacity,
        index_type probing_length,
        const cg::thread_block_tile<CGSize>& group) :
        capacity_(capacity),
        probing_length_(SDIV(probing_length, group.size()) * group.size()),
        group_(group)
    {}

    template<class T>
    DEVICEQUALIFIER INLINEQUALIFIER
    index_type begin(T, T) = delete;

    template<class T>
    DEVICEQUALIFIER INLINEQUALIFIER
    index_type begin(T) = delete;

    /*! \brief begin probing sequence
     * \param[in] key key to be probed
     * \param[in] seed random seed
     * \return initial probing index for \c key
     */
    DEVICEQUALIFIER INLINEQUALIFIER
    index_type begin(key_type key, key_type seed = 0) noexcept 
    {
        i_ = 0;
        base_ = Hasher::hash(key + seed) + group_.thread_rank();
        return (base_ % capacity_);
    }

    /*! \brief next probing index for \c key
     * \return next probing index
     */
    DEVICEQUALIFIER INLINEQUALIFIER
    index_type next() noexcept
    {
        i_ += CGSize;
        return (i_ + group_.thread_rank() < probing_length_) ?
            ((base_ + i_ * i_) % capacity_) : end();
    }

    /*! \brief end specifier of probing sequence
     * \return end specifier
     */
    DEVICEQUALIFIER INLINEQUALIFIER
    static constexpr index_type end() noexcept
    {
        return ~index_type(0);
    }

private:
    const index_type capacity_; //< capacity of the underlying hash table
    const index_type probing_length_; //< number of probing attempts
    const cg::thread_block_tile<CGSize>& group_; //< cooperative group

    index_type i_; //< current probing position
    index_type base_; //< \f$h(k)\f&

}; // class QuadraticProbing

} // namespace probing_schemes

} // namespace warpcore

#endif /* WARPCORE_PROBING_SCHEMES_CUH */