/**
 * @file dynamic_bitset.hpp
 * @brief A drop-in replacement for boost::dynamic_bitset using only standard C++20.
 *
 * Stores bits in a std::vector<uint64_t>. Provides the same API surface that
 * the QEPG backend requires: construction, bit access (with proxy reference),
 * bitwise XOR/AND/OR, popcount, find_first/find_next, block-level access,
 * and swap/reset/comparison.
 *
 * Designed to be SIMD-extensible: the block type can later be templated to
 * use wider SIMD words (128-bit SSE, 256-bit AVX) without changing the API.
 */

#ifndef DYNAMIC_BITSET_HPP
#define DYNAMIC_BITSET_HPP

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace qepg_bits {

class DynamicBitset {
   public:
    using block_type = std::uint64_t;

    /// Sentinel returned by find_first() / find_next() when no set bit is found.
    static constexpr std::size_t npos = static_cast<std::size_t>(-1);

   private:
    static constexpr std::size_t BITS_PER_BLOCK = 64;

    std::size_t num_bits_ = 0;
    std::vector<block_type> blocks_;

    static std::size_t blocks_needed(std::size_t n) noexcept {
        return (n + BITS_PER_BLOCK - 1) / BITS_PER_BLOCK;
    }

    /// Zero out any excess high bits in the last block.
    void sanitize() noexcept {
        if (num_bits_ == 0) return;
        const std::size_t excess = num_bits_ % BITS_PER_BLOCK;
        if (excess != 0) {
            blocks_.back() &= (block_type{1} << excess) - 1;
        }
    }

   public:
    // ---------------------------------------------------------------
    // Construction
    // ---------------------------------------------------------------

    DynamicBitset() = default;

    explicit DynamicBitset(std::size_t num_bits)
        : num_bits_(num_bits), blocks_(blocks_needed(num_bits), block_type{0}) {}

    /// Construct from raw uint64_t words (copies n_words from src).
    DynamicBitset(std::size_t num_bits, const block_type* src, std::size_t n_words)
        : num_bits_(num_bits), blocks_(src, src + n_words) {
        sanitize();
    }

    // ---------------------------------------------------------------
    // Size
    // ---------------------------------------------------------------

    std::size_t size() const noexcept { return num_bits_; }
    std::size_t num_blocks() const noexcept { return blocks_.size(); }

    // ---------------------------------------------------------------
    // Element access (read)
    // ---------------------------------------------------------------

    bool test(std::size_t pos) const {
        if (pos >= num_bits_) throw std::out_of_range("bit index out of range");
        return (blocks_[pos / BITS_PER_BLOCK] >> (pos % BITS_PER_BLOCK)) & 1;
    }

    bool operator[](std::size_t pos) const {
        return (blocks_[pos / BITS_PER_BLOCK] >> (pos % BITS_PER_BLOCK)) & 1;
    }

    // ---------------------------------------------------------------
    // Proxy reference for operator[] write access
    // ---------------------------------------------------------------

    class reference {
        block_type& block_;
        block_type mask_;

       public:
        reference(block_type& block, std::size_t bit_pos) noexcept
            : block_(block), mask_(block_type{1} << bit_pos) {}

        operator bool() const noexcept { return (block_ & mask_) != 0; }

        reference& operator=(bool val) noexcept {
            if (val)
                block_ |= mask_;
            else
                block_ &= ~mask_;
            return *this;
        }

        reference& operator=(const reference& rhs) noexcept {
            return operator=(static_cast<bool>(rhs));
        }
    };

    reference operator[](std::size_t pos) {
        return reference(blocks_[pos / BITS_PER_BLOCK], pos % BITS_PER_BLOCK);
    }

    // ---------------------------------------------------------------
    // Modifiers
    // ---------------------------------------------------------------

    void set(std::size_t pos) {
        blocks_[pos / BITS_PER_BLOCK] |= block_type{1} << (pos % BITS_PER_BLOCK);
    }

    void reset() noexcept {
        std::fill(blocks_.begin(), blocks_.end(), block_type{0});
    }

    void reset(std::size_t pos) {
        blocks_[pos / BITS_PER_BLOCK] &= ~(block_type{1} << (pos % BITS_PER_BLOCK));
    }

    void swap(DynamicBitset& other) noexcept {
        std::swap(num_bits_, other.num_bits_);
        blocks_.swap(other.blocks_);
    }

    // ---------------------------------------------------------------
    // Bitwise operations
    // ---------------------------------------------------------------

    DynamicBitset& operator^=(const DynamicBitset& rhs) {
        if (num_bits_ != rhs.num_bits_) throw std::invalid_argument("bitset sizes differ");
        for (std::size_t i = 0; i < blocks_.size(); ++i)
            blocks_[i] ^= rhs.blocks_[i];
        return *this;
    }

    DynamicBitset& operator&=(const DynamicBitset& rhs) {
        if (num_bits_ != rhs.num_bits_) throw std::invalid_argument("bitset sizes differ");
        for (std::size_t i = 0; i < blocks_.size(); ++i)
            blocks_[i] &= rhs.blocks_[i];
        return *this;
    }

    DynamicBitset& operator|=(const DynamicBitset& rhs) {
        if (num_bits_ != rhs.num_bits_) throw std::invalid_argument("bitset sizes differ");
        for (std::size_t i = 0; i < blocks_.size(); ++i)
            blocks_[i] |= rhs.blocks_[i];
        return *this;
    }

    DynamicBitset operator&(const DynamicBitset& rhs) const {
        DynamicBitset result(*this);
        result &= rhs;
        return result;
    }

    DynamicBitset operator^(const DynamicBitset& rhs) const {
        DynamicBitset result(*this);
        result ^= rhs;
        return result;
    }

    // ---------------------------------------------------------------
    // Counting
    // ---------------------------------------------------------------

    std::size_t count() const noexcept {
        std::size_t total = 0;
        for (auto w : blocks_)
            total += static_cast<std::size_t>(std::popcount(w));
        return total;
    }

    // ---------------------------------------------------------------
    // Searching
    // ---------------------------------------------------------------

    std::size_t find_first() const noexcept {
        for (std::size_t i = 0; i < blocks_.size(); ++i) {
            if (blocks_[i] != 0) {
                std::size_t bit = i * BITS_PER_BLOCK +
                                  static_cast<std::size_t>(std::countr_zero(blocks_[i]));
                return bit < num_bits_ ? bit : npos;
            }
        }
        return npos;
    }

    std::size_t find_next(std::size_t pos) const noexcept {
        if (pos + 1 >= num_bits_) return npos;

        std::size_t next = pos + 1;
        std::size_t block_idx = next / BITS_PER_BLOCK;
        std::size_t bit_idx = next % BITS_PER_BLOCK;

        // Check the remainder of the current block
        block_type masked = blocks_[block_idx] >> bit_idx;
        if (masked != 0) {
            std::size_t bit = block_idx * BITS_PER_BLOCK + bit_idx +
                              static_cast<std::size_t>(std::countr_zero(masked));
            return bit < num_bits_ ? bit : npos;
        }

        // Scan subsequent blocks
        for (std::size_t i = block_idx + 1; i < blocks_.size(); ++i) {
            if (blocks_[i] != 0) {
                std::size_t bit = i * BITS_PER_BLOCK +
                                  static_cast<std::size_t>(std::countr_zero(blocks_[i]));
                return bit < num_bits_ ? bit : npos;
            }
        }
        return npos;
    }

    // ---------------------------------------------------------------
    // Comparison
    // ---------------------------------------------------------------

    bool operator==(const DynamicBitset& rhs) const noexcept {
        return num_bits_ == rhs.num_bits_ && blocks_ == rhs.blocks_;
    }

    bool operator!=(const DynamicBitset& rhs) const noexcept {
        return !(*this == rhs);
    }

    // ---------------------------------------------------------------
    // Block-level access (replaces boost::to_block_range)
    // ---------------------------------------------------------------

    const block_type* data() const noexcept { return blocks_.data(); }
    block_type* data() noexcept { return blocks_.data(); }

    template <typename OutputIt>
    void to_block_range(OutputIt out) const {
        std::copy(blocks_.begin(), blocks_.end(), out);
    }
};

}  // namespace qepg_bits

#endif  // DYNAMIC_BITSET_HPP
