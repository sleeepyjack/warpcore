#ifndef WARPCORE_STORAGE_CUH
#define WARPCORE_STORAGE_CUH

namespace warpcore
{

namespace cg = cooperative_groups;

/*! \brief storage classes
 */
namespace storage
{

/*! \brief thread-safe device-sided ring buffer without any overflow checks
 * \tparam T base type
 */
template<class T>
class CyclicStore
{
public:
    using base_type = T;
    using index_type = index_t;
    using status_type = Status;

    /*! \brief constructor
     * \param[in] capacity buffer capacity
     */
    HOSTQUALIFIER INLINEQUALIFIER
    explicit CyclicStore(index_type capacity) noexcept :
        store_(nullptr),
        capacity_(capacity),
        current_(nullptr),
        status_(status_type::not_initialized()),
        is_copy_(false)
    {
        if(capacity != 0)
        {
            const auto total_bytes = sizeof(T) * capacity_;

            if(available_gpu_memory() >= total_bytes && capacity_ > 0)
            {
                cudaMalloc(&store_, sizeof(T) * capacity_);
                current_ = new index_type(0);
                status_ = status_type::none();
            }
            else
            {
                status_ += status_type::out_of_memory();
            }
        }
        else
        {
            status_ += status_type::invalid_configuration();
        }
    }

    /*! \brief copy-constructor (shallow)
     *  \param[in] object to be copied
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    CyclicStore(const CyclicStore& o) noexcept :
        store_(o.store_),
        capacity_(o.capacity_),
        current_(o.current_),
        status_(o.status_),
        is_copy_(true)
    {}

    /*! \brief move-constructor
     *  \param[in] object to be moved
     */
    HOSTQUALIFIER INLINEQUALIFIER
    CyclicStore(CyclicStore&& o) noexcept :
        store_(std::move(o.store_)),
        capacity_(std::move(o.capacity_)),
        current_(std::move(o.current_)),
        status_(std::move(o.status_)),
        is_copy_(std::move(o.is_copy_))
    {
        o.is_copy_ = true;
    }

    #ifndef __CUDA_ARCH__
    /*! \brief destructor
     */
    HOSTQUALIFIER INLINEQUALIFIER
    ~CyclicStore() noexcept
    {
        if(!is_copy_)
        {
            if(store_ != nullptr) cudaFree(store_);
            delete current_;
        }
    }
    #endif

    /*! \brief atomically fetches the next slot in the buffer
     *  \return pointer to the next slot in the buffer
     *  \info \c const on purpose
     */
    HOSTQUALIFIER INLINEQUALIFIER
    T * get() const noexcept
    {
        index_type old;
        index_type val;

        do
        {
            old = *current_;
            val = (old == capacity_ - 1) ? 0 : old + 1;
        }while(!__sync_bool_compare_and_swap(current_, old, val));

        return store_ + old;
    }

    /*! \brief get buffer status
     *  \return status
     */
    HOSTQUALIFIER INLINEQUALIFIER
    status_type status() const noexcept
    {
        return status_;
    }

    /*! \brief get buffer capacity
     *  \return capacity
     */
    HOSTQUALIFIER INLINEQUALIFIER
    index_type capacity() const noexcept
    {
        return capacity_;
    }

private:
    base_type * store_; //< actual buffer
    const index_type capacity_; //< buffer capacity
    index_type * current_; //< current active buffer slot
    status_type status_; //< buffer status
    bool is_copy_;

}; // class CyclicStore

/*! \brief key/value storage classes
 */
namespace key_value
{

// forward-declaration of friends
template<class Key, class Value>
class SoAStore;

template<class Key, class Value>
class AoSStore;

namespace detail
{

template<class Key, class Value>
class pair_t
{
public:
    Key key;
    Value value;

    DEVICEQUALIFIER
    constexpr pair_t(const pair_t& pair) noexcept = delete;

private:
    DEVICEQUALIFIER
    constexpr pair_t(const Key& key_, const Value& value_) noexcept :
        key(key_), value(value_)
    {}

    DEVICEQUALIFIER
    constexpr pair_t() noexcept : key(), value()
    {}

    friend AoSStore<Key, Value>;
    friend SoAStore<Key, Value>;
};

template<class Key, class Value>
class pair_ref_t
{
public:
    Key& key;
    Value& value;

private:
    DEVICEQUALIFIER
    constexpr pair_ref_t(Key& key_, Value& value_) noexcept :
        key(key_), value(value_)
    {}

    using NKey = std::remove_const_t<Key>;
    using NValue = std::remove_const_t<Value>;

    friend AoSStore<NKey, NValue>;
    friend SoAStore<NKey, NValue>;
};

template<class Key, class Value>
using pair_const_ref_t = pair_ref_t<const Key, const Value>;

} // namespace detail

/*! \brief key/value store with struct-of-arrays memory layout
 * \tparam Key key type
 * \tparam Value value type
 */
template<class Key, class Value>
class SoAStore
{
public:
    using key_type = Key;
    using value_type = Value;
    using status_type = Status;
    using index_type = index_t;
    using tag = tags::key_value_storage;

    /*! \brief constructor
     * \param[in] capacity number of key/value slots
     */
    HOSTQUALIFIER INLINEQUALIFIER
    explicit SoAStore(index_type capacity) noexcept :
        status_(Status::not_initialized()),
        capacity_(capacity),
        keys_(nullptr),
        values_(nullptr),
        is_copy_(false)
    {
        if(capacity != 0)
        {
            const auto total_bytes = (((sizeof(key_type) + sizeof(value_type)) *
                capacity) + sizeof(status_type));

            if(available_gpu_memory() >= total_bytes)
            {
                cudaMalloc(&keys_, sizeof(key_type)*capacity);
                cudaMalloc(&values_, sizeof(value_type)*capacity);

                status_ = status_type::none();
            }
            else
            {
                status_ += status_type::out_of_memory();
            }
        }
        else
        {
            status_ += status_type::invalid_configuration();
        }
    }

    /*! \brief copy-constructor (shallow)
     *  \param[in] object to be copied
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    SoAStore(const SoAStore& o) noexcept :
        status_(o.status_),
        capacity_(o.capacity_),
        keys_(o.keys_),
        values_(o.values_),
        is_copy_(true)
    {}

    /*! \brief move-constructor
     *  \param[in] object to be moved
     */
    HOSTQUALIFIER INLINEQUALIFIER
    SoAStore(SoAStore&& o) noexcept :
        status_(std::move(o.status_)),
        capacity_(std::move(o.capacity_)),
        keys_(std::move(o.keys_)),
        values_(std::move(o.values_)),
        is_copy_(std::move(o.is_copy_))
    {
        o.is_copy_ = true;
    }

    #ifndef __CUDA_ARCH__
    /*! \brief destructor
     */
    HOSTQUALIFIER INLINEQUALIFIER
    ~SoAStore() noexcept
    {
        if(!is_copy_)
        {
            if(keys_   != nullptr) cudaFree(keys_);
            if(values_ != nullptr) cudaFree(values_);
        }
    }
    #endif

    /*! \brief initialize keys
     * \param[in] key initializer key
     * \param[in] stream CUDA stream in which this operation is executed in
     */
    HOSTQUALIFIER INLINEQUALIFIER
    void init_keys(key_type key, cudaStream_t stream = 0) noexcept
    {
        if(!status_.has_any())
        {
            lambda_kernel
            <<<SDIV(capacity_, MAXBLOCKSIZE), MAXBLOCKSIZE, 0, stream>>>
            ([=, *this] DEVICEQUALIFIER
            {
                const index_type tid = global_thread_id();

                if(tid < capacity_)
                {
                    keys_[tid] = key;
                }
            });
        }
    }

    /*! \brief initialize values
     * \param[in] value initializer value
     * \param[in] stream CUDA stream in which this operation is executed in
     */
    HOSTQUALIFIER INLINEQUALIFIER
    void init_values(value_type value, cudaStream_t stream = 0) noexcept
    {
        if(!status_.has_any())
        {
            lambda_kernel
            <<<SDIV(capacity_, MAXBLOCKSIZE), MAXBLOCKSIZE, 0, stream>>>
            ([=, *this] DEVICEQUALIFIER
            {
                const index_type tid = global_thread_id();

                if(tid < capacity_)
                {
                    values_[tid] = value;
                }
            });
        }
    }

    /*! \brief initialize key/value pairs
     * \param[in] key initializer key
     * \param[in] value initializer value
     * \param[in] stream CUDA stream in which this operation is executed in
     */
    HOSTQUALIFIER INLINEQUALIFIER
    void init_pairs(
        key_type key,
        value_type value,
        cudaStream_t stream = 0) noexcept
    {
        if(!status_.has_any())
        {
            lambda_kernel
            <<<SDIV(capacity_, MAXBLOCKSIZE), MAXBLOCKSIZE, 0, stream>>>
            ([=, *this] DEVICEQUALIFIER
            {
                const index_type tid = global_thread_id();

                if(tid < capacity_)
                {
                    keys_[tid] = key;
                    values_[tid] = value;
                }
            });
        }
    }

    /*! \brief accessor
     * \param[in] i index to access
     * \return pair at position \c i
     */
    DEVICEQUALIFIER INLINEQUALIFIER
    detail::pair_ref_t<key_type, value_type> operator[](index_type i) noexcept
    {
        assert(i < capacity_);
        return detail::pair_ref_t<key_type, value_type>{keys_[i], values_[i]};
    }

    /*! \brief const accessor
     * \param[in] i index to access
     * \return pair at position \c i
     */
    DEVICEQUALIFIER INLINEQUALIFIER
    detail::pair_const_ref_t<key_type, value_type> operator[](
        index_type i) const noexcept
    {
        assert(i < capacity_);
        return detail::pair_const_ref_t<key_type, value_type>{keys_[i], values_[i]};
    }

    /*! \brief get storage status
     * \return status
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    status_type status() const noexcept
    {
        return status_;
    }

    /*! \brief get storage capacity
     * \return capacity
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    index_type capacity() const noexcept
    {
        return capacity_;
    }

private:
    status_type status_; //< storage status
    const index_type capacity_; //< storage capacity
    key_type * keys_; //< actual key storage in SoA format
    value_type * values_; //< actual value storage in SoA format
    bool is_copy_; //< indicates if this object is a shallow copy

}; // class SoAStore

/*! \brief key/value store with array-of-structs memory layout
 * \tparam Key key type
 * \tparam Value value type
 */
template<class Key, class Value>
class AoSStore
{
    using pair_t = detail::pair_t<Key, Value>;

public:
    using key_type = Key;
    using value_type = Value;
    using status_type = Status;
    using index_type = index_t;
    using tag = tags::key_value_storage;

    /*! \brief constructor
     * \param[in] capacity number of key/value slots
     */
    HOSTQUALIFIER INLINEQUALIFIER
    explicit AoSStore(index_type capacity) noexcept :
        status_(status_type::not_initialized()),
        capacity_(capacity),
        pairs_(nullptr),
        is_copy_(false)
    {
        if(capacity != 0)
        {
            const auto total_bytes = sizeof(pair_t) * capacity;

            if(available_gpu_memory() >= total_bytes)
            {
                cudaMalloc(&pairs_, sizeof(pair_t) * capacity);

                status_ = status_type::none();
            }
            else
            {
                status_ += status_type::out_of_memory();
            }
        }
        else
        {
            status_ += status_type::invalid_configuration();
        }
    }

    /*! \brief copy-constructor (shallow)
     *  \param[in] object to be copied
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    AoSStore(const AoSStore& o) noexcept :
        status_(o.status_),
        capacity_(o.capacity_),
        pairs_(o.pairs_),
        is_copy_(true)
    {}

    /*! \brief move-constructor
     *  \param[in] object to be moved
     */
    HOSTQUALIFIER INLINEQUALIFIER
    AoSStore(AoSStore&& o) noexcept :
        status_(std::move(o.status_)),
        capacity_(std::move(o.capacity_)),
        pairs_(std::move(o.pairs_)),
        is_copy_(std::move(o.is_copy_))
    {
        o.is_copy_ = true;
    }

    #ifndef __CUDA_ARCH__
    /*! \brief destructor
     */
    HOSTQUALIFIER INLINEQUALIFIER
    ~AoSStore() noexcept
    {
        if(!is_copy_)
        {
            if(pairs_ != nullptr) cudaFree(pairs_);
        }
    }
    #endif

    /*! \brief initialize keys
     * \param[in] key initializer key
     * \param[in] stream CUDA stream in which this operation is executed in
     */
    HOSTQUALIFIER INLINEQUALIFIER
    void init_keys(key_type key, cudaStream_t stream = 0) noexcept
    {
        if(!status_.has_any())
        {
            lambda_kernel
            <<<SDIV(capacity_, MAXBLOCKSIZE), MAXBLOCKSIZE, 0, stream>>>
            ([=, *this] DEVICEQUALIFIER
            {
                const index_type tid = global_thread_id();

                if(tid < capacity_)
                {
                    pairs_[tid].key = key;
                }
            });
        }
    }

    /*! \brief initialize values
     * \param[in] value initializer value
     * \param[in] stream CUDA stream in which this operation is executed in
     */
    HOSTQUALIFIER INLINEQUALIFIER
    void init_values(value_type value, cudaStream_t stream = 0) noexcept
    {
        if(!status_.has_any())
        {
            lambda_kernel
            <<<SDIV(capacity_, MAXBLOCKSIZE), MAXBLOCKSIZE, 0, stream>>>
            ([=, *this] DEVICEQUALIFIER
            {
                const index_type tid = global_thread_id();

                if(tid < capacity_)
                {
                    pairs_[tid].value = value;
                }
            });
        }
    }

    /*! \brief initialize key/value pairs
     * \param[in] key initializer key
     * \param[in] value initializer value
     * \param[in] stream CUDA stream in which this operation is executed in
     */
    HOSTQUALIFIER INLINEQUALIFIER
    void init_pairs(
        key_type key,
        value_type value,
        cudaStream_t stream = 0) noexcept
    {
        if(!status_.has_any())
        {
            lambda_kernel
            <<<SDIV(capacity_, MAXBLOCKSIZE), MAXBLOCKSIZE, 0, stream>>>
            ([=, *this] DEVICEQUALIFIER
            {
                const index_type tid = global_thread_id();

                if(tid < capacity_)
                {
                    pairs_[tid].key = key;
                    pairs_[tid].value = value;
                }
            });
        }
    }

    /*! \brief accessor
     * \param[in] i index to access
     * \return pair at position \c i
     */
    DEVICEQUALIFIER INLINEQUALIFIER
    pair_t& operator[](index_type i) noexcept
    {
        assert(i < capacity_);
        return pairs_[i];
    }

    /*! \brief const accessor
     * \param[in] i index to access
     * \return pair at position \c i
     */
    DEVICEQUALIFIER INLINEQUALIFIER
    const pair_t& operator[](index_type i) const noexcept
    {
        assert(i < capacity_);
        return pairs_[i];
    }

    /*! \brief get storage status
     * \return status
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    status_type status() const noexcept
    {
        return status_;
    }

    /*! \brief get storage capacity
     * \return status
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    index_type capacity() const noexcept
    {
        return capacity_;
    }

private:
    status_type status_; //< storage status
    const index_type capacity_; //< storage capacity
    pair_t * pairs_; //< actual pair storage in AoS format
    bool is_copy_; //< indicates if this object is a shallow copy

}; // class AoSStore

} // namespace key_value

/*! \brief multi-value storage classes
 */
namespace multi_value
{

namespace detail
{
    enum class LinkedListState
    {
        uninitialized = 0,
        initialized   = 1,
        blocking      = 2,
        full          = 3
    };

    template<class Store>
    struct StaticSlab
    {
        typename Store::value_type values[Store::slab_size()];
        index_t previous;
    };

    template<class Store>
    union DynamicSlab
    {
    private:
        using value_type = typename Store::value_type;
        using info_type =
            PackedPair<Store::slab_index_bits(), Store::slab_size_bits()>;

        value_type value_;
        info_type info_;

        DEVICEQUALIFIER INLINEQUALIFIER
        constexpr explicit DynamicSlab(
            info_type info) noexcept : info_{info}
        {}

        DEVICEQUALIFIER INLINEQUALIFIER
        constexpr explicit DynamicSlab(
            value_type value) noexcept : value_{value}
        {}

    public:
        // FIXME friend
        DEVICEQUALIFIER INLINEQUALIFIER
        constexpr explicit DynamicSlab(
            index_t previous,
            index_t slab_size) noexcept : info_{previous, slab_size}
        {}

        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        constexpr explicit DynamicSlab() noexcept :
        info_()
        {};

        DEVICEQUALIFIER INLINEQUALIFIER
        DynamicSlab<Store> atomic_exchange_info(DynamicSlab<Store> slab) noexcept
        {
            return DynamicSlab<Store>(atomicExch(&info_, slab.info_));
        }

        DEVICEQUALIFIER INLINEQUALIFIER
        constexpr value_type value() const noexcept
        {
            return value_;
        }

        DEVICEQUALIFIER INLINEQUALIFIER
        constexpr index_t previous() const noexcept
        {
            return info_.first();
        }

        DEVICEQUALIFIER INLINEQUALIFIER
        constexpr index_t slab_size() const noexcept
        {
            return info_.second();
        }

        DEVICEQUALIFIER INLINEQUALIFIER
        constexpr void value(const value_type& val) noexcept
        {
            value_ = val;
        }

        DEVICEQUALIFIER INLINEQUALIFIER
        constexpr void previous(index_t prev) noexcept
        {
            info_.first(prev);
        }

        DEVICEQUALIFIER INLINEQUALIFIER
        constexpr void slab_size(index_t size) noexcept
        {
            info_.second(size);
        }
    };

    template<class Store>
    class StaticSlabListHandle
    {
        using packed_type = PackedTriple<
            2,
            Store::slab_index_bits(),
            Store::value_counter_bits()>;

        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        constexpr explicit StaticSlabListHandle(
            LinkedListState state,
            index_t index,
            index_t counter) noexcept : pack_()
        {
            pack_.first(state);
            pack_.second(index);
            pack_.third(counter);
        };

        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        constexpr explicit StaticSlabListHandle(packed_type pack) noexcept :
        pack_(pack)
        {};

    public:
        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        constexpr explicit StaticSlabListHandle() noexcept :
        pack_()
        {};

    private:
        DEVICEQUALIFIER INLINEQUALIFIER
        constexpr LinkedListState linked_list_state() const noexcept
        {
            return pack_.template first_as<LinkedListState>();
        }

        DEVICEQUALIFIER INLINEQUALIFIER
        constexpr index_t slab_index() const noexcept
        {
            return pack_.second();
        }

        DEVICEQUALIFIER INLINEQUALIFIER
        constexpr index_t value_count() const noexcept
        {
            return pack_.third();
        }

        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        static constexpr index_t max_slab_index() noexcept
        {
            return (index_t{1} << Store::slab_index_bits()) - 1;
        }

        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        static constexpr index_t max_value_count() noexcept
        {
            return (index_t{1} << Store::value_counter_bits()) - 1;
        }

        DEVICEQUALIFIER INLINEQUALIFIER
        constexpr bool is_uninitialized() const noexcept
        {
            return (linked_list_state() == LinkedListState::uninitialized);
        }

        DEVICEQUALIFIER INLINEQUALIFIER
        constexpr bool is_initialized() const noexcept
        {
            return (linked_list_state() == LinkedListState::initialized);
        }

        DEVICEQUALIFIER INLINEQUALIFIER
        constexpr bool is_blocking() const noexcept
        {
            return (linked_list_state() == LinkedListState::blocking);
        }

        DEVICEQUALIFIER INLINEQUALIFIER
        constexpr bool is_full() const noexcept
        {
            return (linked_list_state() == LinkedListState::full);
        }

        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        constexpr bool operator==(
            StaticSlabListHandle<Store> other) const noexcept
        {
            return pack_ == other.pack_;
        }

        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        constexpr bool operator!=(
            StaticSlabListHandle<Store> other) const noexcept
        {
            return !(*this == other);
        }

        packed_type pack_;

        DEVICEQUALIFIER INLINEQUALIFIER
        friend StaticSlabListHandle<Store> atomicCAS(
            StaticSlabListHandle<Store> * address_,
            StaticSlabListHandle<Store>   compare_,
            StaticSlabListHandle<Store>   val_) noexcept
        {
            return StaticSlabListHandle(
                atomicCAS(&(address_->pack_), compare_.pack_, val_.pack_));
        }

        DEVICEQUALIFIER INLINEQUALIFIER
        friend StaticSlabListHandle<Store> atomicExch(
            StaticSlabListHandle<Store> * address_,
            StaticSlabListHandle<Store>   val_) noexcept
        {
            return StaticSlabListHandle(
                atomicExch(&(address_->pack_), val_.pack_));
        }

        friend Store;
    };

    template<class Store>
    class DynamicSlabListHandle
    {
        using packed_type = PackedQuadruple<
            2,
            Store::slab_index_bits(),
            Store::value_counter_bits(),
            Store::slab_size_bits()>;

        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        constexpr explicit DynamicSlabListHandle(
            LinkedListState state,
            index_t index,
            index_t counter,
            index_t offset) noexcept : pack_()
        {
            pack_.first(state);
            pack_.second(index);
            pack_.third(counter);
            pack_.fourth(offset);
        };

        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        constexpr explicit DynamicSlabListHandle(packed_type pack) noexcept :
        pack_(pack)
        {};

    public:
        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        constexpr explicit DynamicSlabListHandle() noexcept :
        pack_()
        {};

    private:
        DEVICEQUALIFIER INLINEQUALIFIER
        constexpr LinkedListState linked_list_state() const noexcept
        {
            return pack_.template first_as<LinkedListState>();
        }

        DEVICEQUALIFIER INLINEQUALIFIER
        constexpr index_t slab_index() const noexcept
        {
            return pack_.second();
        }

        DEVICEQUALIFIER INLINEQUALIFIER
        constexpr index_t value_count() const noexcept
        {
            return pack_.third();
        }

        DEVICEQUALIFIER INLINEQUALIFIER
        constexpr index_t num_values_tail() const noexcept
        {
            return pack_.fourth();
        }

    public:
        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        static constexpr index_t max_slab_index() noexcept
        {
            return (index_t{1} << Store::slab_index_bits()) - 1;
        }

        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        static constexpr index_t max_value_count() noexcept
        {
            return (index_t{1} << Store::value_counter_bits()) - 1;
        }

        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        static constexpr index_t max_slab_size() noexcept
        {
            return (index_t{1} << Store::slab_size_bits()) - 1;
        }

    private:
        DEVICEQUALIFIER INLINEQUALIFIER
        constexpr bool is_uninitialized() const noexcept
        {
            return (linked_list_state() == LinkedListState::uninitialized);
        }

        DEVICEQUALIFIER INLINEQUALIFIER
        constexpr bool is_initialized() const noexcept
        {
            return (linked_list_state() == LinkedListState::initialized);
        }

        DEVICEQUALIFIER INLINEQUALIFIER
        constexpr bool is_blocking() const noexcept
        {
            return (linked_list_state() == LinkedListState::blocking);
        }

        DEVICEQUALIFIER INLINEQUALIFIER
        constexpr bool is_full() const noexcept
        {
            return (linked_list_state() == LinkedListState::full);
        }

        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        constexpr bool operator==(
            DynamicSlabListHandle<Store> other) const noexcept
        {
            return pack_ == other.pack_;
        }

        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        constexpr bool operator!=(
            DynamicSlabListHandle<Store> other) const noexcept
        {
            return !(*this == other);
        }

        packed_type pack_;

        DEVICEQUALIFIER INLINEQUALIFIER
        friend DynamicSlabListHandle<Store> atomicCAS(
            DynamicSlabListHandle<Store> * address_,
            DynamicSlabListHandle<Store>   compare_,
            DynamicSlabListHandle<Store>   val_) noexcept
        {
            return DynamicSlabListHandle(
                atomicCAS(&(address_->pack_), compare_.pack_, val_.pack_));
        }

        DEVICEQUALIFIER INLINEQUALIFIER
        friend DynamicSlabListHandle<Store> atomicExch(
            DynamicSlabListHandle<Store> * address_,
            DynamicSlabListHandle<Store>   val_) noexcept
        {
            return DynamicSlabListHandle(
                atomicExch(&(address_->pack_), val_.pack_));
        }

        friend Store;
    };

} // namespace detail

/*! \brief value store consisting of same-sized linked slabs of values
 * \warning broken DO NOT USE
 * \tparam Value type to store
 * \tparam SlabSize size of each linked slab of memory
 * \tparam SlabIndexBits number of bits used to enumerate slab IDs
 * \tparam ValueCounterBits number of bits used to count values in a slab list
 */
template<
    class   Value,
    index_t SlabSize = 8,
    index_t SlabIndexBits = 31,
    index_t ValueCounterBits = 31>
class StaticSlabListStore
{
private:
    static_assert(
        checks::is_valid_value_type<Value>(),
        "Value type must be trivially copyable.");

    static_assert(
        (SlabSize > 0),
        "Invalid slab size.");

    static_assert(
        (SlabIndexBits + ValueCounterBits <= 64 - 2),
        "Too many bits for slab index and value counter.");

    using type = StaticSlabListStore<
        Value,
        SlabSize,
        SlabIndexBits,
        ValueCounterBits>;

    friend detail::StaticSlabListHandle<type>;

public:
    using value_type = Value;
    using handle_type = detail::StaticSlabListHandle<type>;
    using status_type = Status;
    using slab_type = detail::StaticSlab<type>;
    using index_type = index_t;
    using tag = tags::static_value_storage;

    /*! \brief get slab size
     * \return slab size
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr index_type slab_size() noexcept { return SlabSize; };

    /*! \brief get number of bits used to enumerate slabs
     * \return number of bits
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr index_type slab_index_bits() noexcept { return SlabIndexBits; };

    /*! \brief get number of bits used to count values in a slab list
     * \return number of bits
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr index_type value_counter_bits() noexcept { return ValueCounterBits; };

private:

    friend slab_type;

    /*! \brief head slab identifier
     *  \return identifier
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr index_type head() noexcept
    {
        return handle_type::max_slab_index();
    }

public:
    /*! \brief get uninitialized handle
     *  \return handle
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr handle_type uninitialized_handle() noexcept
    {
        return handle_type{
            detail::LinkedListState::uninitialized,
            head(),
            0};
    }

    /*! \brief get number of values in slab list
     *  \return value count
     */
    DEVICEQUALIFIER INLINEQUALIFIER
    static constexpr index_type size(handle_type handle) noexcept
    {
        return handle.value_count();
    }

    /*! \brief constructor
     * \param[in] min_capacity minimum number of value slots
     */
    HOSTQUALIFIER INLINEQUALIFIER
    explicit StaticSlabListStore(index_type min_capacity) noexcept :
        status_(status_type::not_initialized()),
        capacity_(SDIV(min_capacity, slab_size()) * slab_size()),
        num_slabs_(capacity_ / slab_size()),
        slabs_(nullptr),
        next_free_slab_(nullptr),
        is_copy_(false)
    {
        if(num_slabs_ <= head())
        {
            const auto total_bytes =
                sizeof(slab_type) * num_slabs_ + sizeof(index_type);

            if(available_gpu_memory() >= total_bytes)
            {
                cudaMalloc(&slabs_, sizeof(slab_type) * num_slabs_);
                cudaMalloc(&next_free_slab_, sizeof(index_type));

                status_ = status_type::none();
                init();
            }
            else
            {
                status_ += status_type::out_of_memory();
            }
        }
        else
        {
            status_ += status_type::invalid_configuration();
        }
    }

    /*! \brief copy-constructor (shallow)
     *  \param[in] object to be copied
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    StaticSlabListStore(const StaticSlabListStore& o) noexcept :
        status_(o.status_),
        capacity_(o.capacity_),
        num_slabs_(o.num_slabs_),
        slabs_(o.slabs_),
        next_free_slab_(o.next_free_slab_),
        is_copy_(true)
    {}

    /*! \brief move-constructor
     *  \param[in] object to be moved
     */
    HOSTQUALIFIER INLINEQUALIFIER
    StaticSlabListStore(StaticSlabListStore&& o) noexcept :
        status_(std::move(o.status_)),
        capacity_(std::move(o.capacity_)),
        num_slabs_(std::move(o.num_slabs_)),
        slabs_(std::move(o.slabs_)),
        next_free_slab_(std::move(o.next_free_slab_)),
        is_copy_(std::move(o.is_copy_))
    {
        o.is = true;
    }

    #ifndef __CUDA_ARCH__
    /*! \brief destructor
     */
    HOSTQUALIFIER INLINEQUALIFIER
    ~StaticSlabListStore() noexcept
    {
        if(!is_copy_)
        {
            if(slabs_ != nullptr) cudaFree(slabs_);
            if(next_free_slab_ != nullptr) cudaFree(next_free_slab_);
        }
    }
    #endif

    /*! \brief (re)initialize the store
     * \param[in] stream CUDA stream in which this operation is executed in
     */
    HOSTQUALIFIER INLINEQUALIFIER
    void init(cudaStream_t stream = 0) noexcept
    {
        if(!status_.has_not_initialized())
        {
            lambda_kernel
            <<<SDIV(num_slabs_, MAXBLOCKSIZE), MAXBLOCKSIZE, 0, stream>>>
            ([=, *this] DEVICEQUALIFIER () mutable
            {
                const index_type tid = blockDim.x * blockIdx.x + threadIdx.x;

                if(tid < num_slabs_)
                {
                    if(tid == 0)
                    {
                        *next_free_slab_ = 0;
                    }

                    slabs_[tid].previous = head();
                }
            });

            status_ = status_type::none();
        }
    }

    // FIXME
    /*! \brief append a value to a slab list
     * \param[in] handle handle to the slab list
     * \param[in] value value to be inserted
     * \return status
     */
    DEVICEQUALIFIER INLINEQUALIFIER
    status_type append(
        handle_type& handle,
        const value_type& value) noexcept
    {
        auto current_handle = cub::ThreadLoad<cub::LOAD_VOLATILE>(&handle);

        if(current_handle.is_uninitialized())
        {
            // block handle
            const auto old_handle = atomicCAS(
                &handle,
                current_handle,
                handle_type{
                    detail::LinkedListState::blocking,
                    0,
                    0});

            // winner allocates first slab
            if(old_handle == current_handle)
            {
                // get index of next free slab
                const index_type alloc = atomicAdd(next_free_slab_, 1);

                // if not full
                if(alloc < num_slabs_)
                {
                    value_type val;
                    val.key = value.key;
                    val.slab_index = alloc;
                    val.value_index = 1;
                    slabs_[alloc].values[0] = val;

                    // write value
                    //slabs_[alloc].values[0] = value;

                    // unblock handle
                    atomicExch(
                        &handle,
                        handle_type{
                            detail::LinkedListState::initialized,
                            alloc,
                            1});

                    return status_type::none();
                }
                else
                {
                    // mark as full
                    atomicExch(
                        &handle,
                        handle_type{
                            detail::LinkedListState::full,
                            0,
                            0});

                    //status_.atomic_join(Status::out_of_memory();
                    return status_type::out_of_memory();
                }
            }
        }

        // try to find a slot until there is no more space
        while(true)
        {
            current_handle = cub::ThreadLoad<cub::LOAD_VOLATILE>(&handle);

            if(current_handle.is_blocking())
            {
                //__nanosleep(100); // why not?
                continue;
            }

            if(current_handle.is_full())
            {
                return status_type::out_of_memory();
            }

            // if the current slab is already full allocate new slab
            if((current_handle.value_count() % slab_size()) == 0)
            {
                const auto old_handle = atomicCAS(
                    &handle,
                    current_handle,
                    handle_type{
                        detail::LinkedListState::blocking,
                        current_handle.slab_index(),
                        current_handle.value_count()});

                // TODO == initialized
                if(old_handle == current_handle)
                {

                    // get index of next free slab
                    const index_type alloc = atomicAdd(next_free_slab_, 1);

                    // if not full
                    if(alloc < num_slabs_)
                    {
                        value_type val;
                        val.key = value.key;
                        val.slab_index = alloc;
                        val.value_index = 1;
                        slabs_[alloc].values[0] = val;

                        // write value
                        //slabs_[alloc].values[0] = value;

                        // establish link between slabs
                        // FIXME
                        slabs_[alloc].previous =
                            current_handle.slab_index();

                        // unblock handle
                        atomicExch(
                            &handle,
                            handle_type{
                                detail::LinkedListState::initialized,
                                alloc,
                                current_handle.value_count() + 1});

                        return status_type::none();
                    }
                    else
                    {
                        // mark as full
                        atomicExch(
                            &handle,
                            handle_type{
                                detail::LinkedListState::full,
                                current_handle.slab_index(),
                                current_handle.value_count()});

                        return status_type::out_of_memory();
                    }
                }
            }
            else
            {
                const auto old_handle = atomicCAS(
                    &handle,
                    current_handle,
                    handle_type{
                        current_handle.linked_list_state(),
                        current_handle.slab_index(),
                        current_handle.value_count() + 1});

                if(old_handle == current_handle)
                {
                    const auto i = current_handle.slab_index();
                    const auto j = current_handle.value_count() % slab_size();

                    value_type val;
                    val.key = value.key;
                    val.slab_index = i;
                    val.value_index = j+1;
                    slabs_[i].values[j] = val;

                    //slabs_[i].values[j] = value;

                    return status_type::none();
                }
            }
        }

        return status_type::unknown_error();
    }

    /*! \brief apply a (lambda-)function on each value inside a slab list
     * \tparam Func function to be executed for each value
     * \param[in] handle handle to the slab list
     * \param[in] f function which takes the value together whith the index of the value inside the list as parameters
     * \param[in] group cooperative group used for hash table probing
     */
    template<class Func>
    DEVICEQUALIFIER INLINEQUALIFIER
    void for_each(
        handle_type handle,
        Func f,
        const cg::thread_group& group = cg::this_thread()) const noexcept
    {
        const index_type rank = group.thread_rank();
        const index_type group_size = group.size();
        index_type local_index = rank;
        index_type global_index = rank;
        const index_type num_tail_values = handle.value_count() % slab_size();

        // return if nothing is to be done
        if(!handle.is_initialized() || handle.slab_index() == head()) return;

        slab_type& current_slab = slabs_[handle.slab_index()];

        // process remaining values residing in tail slab
        while (local_index < num_tail_values)
        {
            f(current_slab.values[local_index], global_index);
            local_index += group_size;
            global_index += group_size;
        }

        local_index -= num_tail_values;

        // as long as processed slabs are full
        while(current_slab.previous != head())
        {
            current_slab = slabs_[current_slab.previous];

            while (local_index < slab_size())
            {
                f(current_slab.values[local_index], global_index);
                local_index += group_size;
                global_index += group_size;
            }

            local_index -= slab_size();
        }
    }

    /*! \brief get status
     * \return status
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    status_type status() const noexcept
    {
        return status_;
    }

    /*! \brief get value capacity
     * \return capacity
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    index_type capacity() const noexcept
    {
        return capacity_;
    }

    /*! \brief get load factor
     * \param[in] stream CUDA stream in which this operation is executed in
     * \return load factor
     */
     HOSTDEVICEQUALIFIER INLINEQUALIFIER
     float load_factor(cudaStream_t stream = 0) const noexcept
     {
         index_type load = 0;

         cudaMemcpyAsync(
             &load, next_free_slab_, sizeof(index_type), D2H, stream);

         cudaStreamSynchronize(stream);

         return float(load) / float(num_slabs_);
     }

    /*! \brief get number of slabs
     * \return number of slabs
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    index_type num_slabs() const noexcept
    {
        return num_slabs_;
    }

    /*! \brief indicates if this object is a shallow copy
     * \return \c bool
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    bool is_copy() const noexcept
    {
        return is_copy_;
    }

private:
    status_type status_; //< status of the store
    const index_type capacity_; //< value capacity
    const index_type num_slabs_; //< number of slabs
    slab_type * slabs_; //< pointer to slab store
    index_type * next_free_slab_; //< index of next non-occupied slab
    bool is_copy_; //< indicates if this object is a shallow copy
};

/*! \brief value store consisting of growing linked slabs of values
 * \tparam Value type to store
 * \tparam SlabIndexBits number of bits used to enumerate slab IDs
 * \tparam ValueCounterBits number of bits used to count values in a slab list
 * \tparam SlabSizeBits number of bits used to hold the value capacity of a slab
 */
template<
    class   Value,
    index_t SlabIndexBits = 30,
    index_t ValueCounterBits = 22,
    index_t SlabSizeBits = 10>
class DynamicSlabListStore
{
private:
    static_assert(
        checks::is_valid_value_type<Value>(),
        "Value type must be trivially copyable.");

    static_assert(
        (SlabIndexBits + ValueCounterBits + SlabSizeBits + 2 <= 64),
        "Too many bits for slab index and value counter and slab size.");

    using type = DynamicSlabListStore<
        Value,
        SlabIndexBits,
        ValueCounterBits,
        SlabSizeBits>;

    friend detail::DynamicSlabListHandle<type>;

public:
    using value_type = Value;
    using handle_type = detail::DynamicSlabListHandle<type>;
    using index_type = index_t;
    using status_type = Status;
    using slab_type = detail::DynamicSlab<type>;
    using tag = tags::dynamic_value_storage;

    /*! \brief get number of bits used to enumerate slabs
     * \return number of bits
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr index_type slab_index_bits() noexcept
    {
        return SlabIndexBits;
    };

    /*! \brief get number of bits used to count values in a slab list
     * \return number of bits
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr index_type value_counter_bits() noexcept
    {
        return ValueCounterBits;
    };

    /*! \brief get number of bits used to hold the value capacity of a slab
     * \return number of bits
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr index_type slab_size_bits() noexcept
    {
        return SlabSizeBits;
    };

private:
    friend slab_type;

    /*! \brief head slab identifier
     *  \return identifier
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr index_type head() noexcept
    {
        return handle_type::max_slab_index();
    }

public:
    /*! \brief get uninitialized handle
     *  \return handle
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr handle_type uninitialized_handle() noexcept
    {
        return handle_type{detail::LinkedListState::uninitialized, head(), 0, 0};
    }

    /*! \brief get number of values in slab list
     *  \return value count
     */
    DEVICEQUALIFIER INLINEQUALIFIER
    static constexpr index_type size(handle_type handle) noexcept
    {
        return handle.value_count();
    }


    /*! \brief constructor
     * \param[in] max_capacity maximum number of value slots
     * \param[in] slab_grow_factor factor which determines the growth of each newly allocated slab
     * \param[in] min_slab_size value capacity of the first slab of a slab list
     * \param[in] max_slab_size value capacity after which no more growth is allowed for newly allocated slabs
     */
    HOSTQUALIFIER INLINEQUALIFIER
    explicit DynamicSlabListStore(
        index_type max_capacity,
        float slab_grow_factor = 2.0,
        index_type min_slab_size = 1,
        index_type max_slab_size = handle_type::max_slab_size()) noexcept :
        status_(Status::not_initialized()),
        capacity_(max_capacity),
        slab_grow_factor_(slab_grow_factor),
        min_slab_size_(min_slab_size),
        max_slab_size_(max_slab_size),
        next_free_slab_(nullptr),
        slabs_(nullptr),
        is_copy_(false)
    {
        if(capacity_ < handle_type::max_slab_index() &&
            slab_grow_factor_ >= 1.0 &&
            min_slab_size_ >= 1 &&
            max_slab_size_ >= min_slab_size_ &&
            max_slab_size_ <= handle_type::max_slab_size())
        {
            const auto total_bytes =
                sizeof(slab_type) * capacity_ + sizeof(index_type);

            if(available_gpu_memory() >= total_bytes)
            {
                cudaMalloc(&slabs_, sizeof(slab_type) * capacity_);
                cudaMalloc(&next_free_slab_, sizeof(index_type));

                status_ = status_type::none();
                init();
            }
            else
            {
                status_ += status_type::out_of_memory();
            }
        }
        else
        {
            status_ += status_type::invalid_configuration();
        }
    }

    /*! \brief copy-constructor (shallow)
     *  \param[in] object to be copied
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    DynamicSlabListStore(const DynamicSlabListStore& o) noexcept :
        status_(o.status_),
        capacity_(o.capacity_),
        slab_grow_factor_(o.slab_grow_factor_),
        min_slab_size_(o.min_slab_size_),
        max_slab_size_(o.max_slab_size_),
        slabs_(o.slabs_),
        next_free_slab_(o.next_free_slab_),
        is_copy_(true)
    {}

    /*! \brief move-constructor
     *  \param[in] object to be moved
     */
    HOSTQUALIFIER INLINEQUALIFIER
    DynamicSlabListStore(DynamicSlabListStore&& o) noexcept :
        status_(std::move(o.status_)),
        capacity_(std::move(o.capacity_)),
        slab_grow_factor_(std::move(o.slab_grow_factor_)),
        min_slab_size_(std::move(o.min_slab_size_)),
        max_slab_size_(std::move(o.max_slab_size_)),
        slabs_(std::move(o.slabs_)),
        next_free_slab_(std::move(o.next_free_slab_)),
        is_copy_(std::move(o.is_copy_))
    {
        o.is_copy_ = true;
    }

    #ifndef __CUDA_ARCH__
    /*! \brief destructor
     */
    HOSTQUALIFIER INLINEQUALIFIER
    ~DynamicSlabListStore() noexcept
    {
        if(!is_copy_)
        {
            if(slabs_ != nullptr) cudaFree(slabs_);
            if(next_free_slab_ != nullptr) cudaFree(next_free_slab_);
        }
    }
    #endif

    /*! \brief (re)initialize storage
     * \param[in] stream CUDA stream in which this operation is executed in
     */
    HOSTQUALIFIER INLINEQUALIFIER
    void init(cudaStream_t stream = 0) noexcept
    {
        if(!status_.has_not_initialized())
        {
            lambda_kernel
            <<<SDIV(capacity_, MAXBLOCKSIZE), MAXBLOCKSIZE, 0, stream>>>
            ([=, *this] DEVICEQUALIFIER () mutable
            {
                const index_type tid = global_thread_id();

                if(tid < capacity_)
                {
                    if(tid == 0)
                    {
                        *next_free_slab_ = 0;
                    }

                    slabs_[tid].previous(head());
                    slabs_[tid].slab_size(min_slab_size_);
                }
            });

            status_ = status_type::none();
        }
    }

    /*! \brief append a value to a slab list
     * \param[in] handle handle to the slab list
     * \param[in] value value to be inserted
     * \return status
     */
    DEVICEQUALIFIER INLINEQUALIFIER
    status_type append(
        handle_type& handle,
        const value_type& value) noexcept
    {
        handle_type current_handle =cub::ThreadLoad<cub::LOAD_VOLATILE>(&handle);

        if(current_handle.is_uninitialized())
        {
            // block handle
            const auto old_handle = atomicCAS(
                &handle,
                current_handle,
                handle_type{
                    detail::LinkedListState::blocking,
                    head(),
                    0,
                    0});

            // winner allocates first slab
            if(old_handle == current_handle)
            {
                const index_type alloc =
                        atomicAdd(next_free_slab_, min_slab_size_);

                if(alloc + min_slab_size_ <= capacity_)
                {
                    slabs_[alloc].value(value);

                    // successfully allocated initial slab
                    atomicExch(
                    &handle,
                    handle_type{
                        detail::LinkedListState::initialized,
                        alloc,
                        1,
                        1});

                    return Status::none();
                }

                // mark as full
                atomicExch(
                    &handle,
                    handle_type{
                        detail::LinkedListState::full,
                        head(),
                        0,
                        0});

                return status_type::out_of_memory();
            }
        }

        // try to find a slot until there is no more space
        while(true)
        {
            current_handle = cub::ThreadLoad<cub::LOAD_VOLATILE>(&handle);

            if(current_handle.is_blocking())
            {
                //__nanosleep(1000); // why not?
                continue;
            }

            if(current_handle.is_full())
            {
                return status_type::out_of_memory();
            }

            if(current_handle.value_count() == handle_type::max_value_count())
            {
                return status_type::index_overflow();
            }

            const auto current_slab = cub::ThreadLoad<cub::LOAD_VOLATILE>(
                slabs_ + current_handle.slab_index());

            const auto current_slab_size =
                (current_handle.value_count() <= min_slab_size_) ?
                    min_slab_size_ : current_slab.slab_size();

            // if the slab is already full allocate a new slab
            if(current_handle.num_values_tail() == current_slab_size)
            {
                const auto old_handle = atomicCAS(
                    &handle,
                    current_handle,
                    handle_type{
                        detail::LinkedListState::blocking,
                        current_handle.slab_index(),
                        current_handle.value_count(),
                        current_handle.num_values_tail()});

                // blocking failed -> reload handle
                if(old_handle != current_handle)
                {
                    continue;
                }

                // compute new slab size
                const index_type new_slab_size = min(
                    float(max_slab_size_),
                    ceilf(float(current_slab_size) * slab_grow_factor_));

                // get index of next free slab in pool
                const index_type alloc =
                    atomicAdd(next_free_slab_, new_slab_size + 1);

                if(alloc + new_slab_size + 1 <= capacity_)
                {
                    slabs_[alloc + 1].value(value);

                    const auto old = slabs_[alloc].atomic_exchange_info(
                        slab_type{current_handle.slab_index(),
                        new_slab_size});

                    if(old.slab_size() != 0)
                    {
                        // slab allocation successful
                        atomicExch(
                        &handle,
                        handle_type{
                            detail::LinkedListState::initialized,
                            alloc,
                            current_handle.value_count() + 1,
                            1});
                    }

                    return Status::none();
                }
                else
                {
                    // mark as full
                    atomicExch(
                        &handle,
                        handle_type{
                            detail::LinkedListState::full,
                            current_handle.slab_index(),
                            current_handle.value_count(),
                            current_handle.num_values_tail()});

                    return status_type::out_of_memory();
                }
            }

            const auto old_handle =
                atomicCAS(
                    &handle,
                    current_handle,
                    handle_type{
                    detail::LinkedListState::initialized,
                    current_handle.slab_index(),
                    current_handle.value_count() + 1,
                    current_handle.num_values_tail() + 1});

            if(old_handle == current_handle)
            {
                const auto i = current_handle.slab_index();
                const auto j =
                    (current_handle.value_count() + 1 <= min_slab_size_) ?
                        current_handle.num_values_tail() :
                        current_handle.num_values_tail() + 1;

                slabs_[i + j].value(value);

                return status_type::none();
            }
        }

        return status_type::unknown_error();
    }

    /*! \brief apply a (lambda-)function on each value inside a slab list
     * \tparam Func function to be executed for each value
     * \param[in] handle handle to the slab list
     * \param[in] f function which takes the value together whith the index of the value inside the list as parameters
     * \param[in] group cooperative group used for hash table probing
     */
    template<class Func>
    DEVICEQUALIFIER INLINEQUALIFIER
    void for_each(
        handle_type handle,
        Func f,
        const cg::thread_group& group = cg::this_thread()) const noexcept
    {
        const index_type rank = group.thread_rank();
        const index_type group_size = group.size();
        index_type local_index = rank;

        // return if nothing is to be done
        if(!handle.is_initialized() || handle.slab_index() == head()) return;

        slab_type * slab_ptr = slabs_ + handle.slab_index();

        const index_type slab_offset =
            (handle.value_count() <= min_slab_size_) ? 0 : 1;

        // process first slab
        while(local_index < handle.num_values_tail())
        {
            f((slab_ptr + local_index + slab_offset)->value(), local_index);
            local_index += group_size;
        }

        index_type global_index = local_index;
        local_index -= handle.num_values_tail();

        // while there are more values left, process them, too
        while(global_index < handle.value_count())
        {
            slab_ptr = slabs_ + slab_ptr->previous();

            // check if we are at the final slab
            const bool last =
                (global_index >= (handle.value_count() - min_slab_size_));
            const auto current_slab_size =
                last ? min_slab_size_ : slab_ptr->slab_size();
            const index_type slab_offset =
                last ? 0 : 1;

            // while there are more values to be processed in the current slab
            while(local_index < current_slab_size)
            {
                f((slab_ptr + local_index + slab_offset)->value(), global_index);

                local_index += group_size;
                global_index += group_size;
            }

            local_index -= slab_ptr->slab_size();
        }
    }

    /*! \brief get status
     * \return status
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    status_type status() const noexcept
    {
        return status_;
    }

    /*! \brief get maximum value capacity
     * \return capacity
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    index_type capacity() const noexcept
    {
        return capacity_;
    }

    /*! \brief get load factor
     * \param[in] stream CUDA stream in which this operation is executed in
     * \return load factor
     */
     HOSTDEVICEQUALIFIER INLINEQUALIFIER
     float load_factor(cudaStream_t stream = 0) const noexcept
     {
         index_type load = 0;

         cudaMemcpyAsync(
             &load, next_free_slab_, sizeof(index_type), D2H, stream);

         cudaStreamSynchronize(stream);

         return float(load) / float(capacity());
     }

     /*! \brief get the number of occupied bytes
     * \param[in] stream CUDA stream in which this operation is executed in
     * \return bytes
     */
     HOSTDEVICEQUALIFIER INLINEQUALIFIER
     index_type bytes_occupied(cudaStream_t stream = 0) const noexcept
     {
         index_type occupied = 0;

         cudaMemcpyAsync(
             &occupied, next_free_slab_, sizeof(index_type), D2H, stream);

         cudaStreamSynchronize(stream);

         return occupied * sizeof(slab_type);
     }

    /*! \brief get slab growth factor
     * \return factor
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    float slab_grow_factor() const noexcept
    {
        return slab_grow_factor_;
    }

    /*! \brief get minimum slab capacity
     * \return capacity
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    index_type min_slab_size() const noexcept
    {
        return min_slab_size_;
    }

    /*! \brief get maximum slab capacity
     * \return capacity
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    index_type max_slab_size() const noexcept
    {
        return max_slab_size_;
    }

    /*! \brief indicates if this object is a shallow copy
     * \return \c bool
     */
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    bool is_copy() const noexcept
    {
        return is_copy_;
    }

private:
    status_type status_; //< status of the store
    const index_type capacity_; //< value capacity
    const float slab_grow_factor_; //< grow factor for allocated slabs
    const index_type min_slab_size_; //< initial slab size
    const index_type max_slab_size_; //< slab size after which no more growth occurs
    slab_type * slabs_; //< pointer to slab store
    index_type * next_free_slab_; //< index of next non-occupied slab
    bool is_copy_; //< indicates if this object is a shallow copy
};

} // namespace multi_value

} // namespace storage

} // namespace warpcore

#endif /* WARPCORE_STORAGE_CUH */