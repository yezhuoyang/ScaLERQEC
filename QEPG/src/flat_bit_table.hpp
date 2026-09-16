/**
 * @file flat_bit_table.hpp
 * @brief Contiguous 2D bit matrix with cache-line aligned rows and SIMD XOR.
 *
 * FlatBitTable stores a 2D bit matrix in a single contiguous allocation
 * with each row padded to 64-byte (cache line) alignment. This layout
 * eliminates pointer chasing and enables hardware prefetching and SIMD
 * operations on rows.
 */

#ifndef FLAT_BIT_TABLE_HPP
#define FLAT_BIT_TABLE_HPP

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <limits>
#include <stdexcept>
#include <vector>
#include "simd_util.hpp"

namespace qepg_bits {

class FlatBitTable {
public:
    FlatBitTable() = default;

    /**
     * @brief Construct a zero-initialized 2D bit matrix.
     * @param n_rows Number of rows.
     * @param n_cols Number of columns (bits per row).
     */
    FlatBitTable(std::size_t n_rows, std::size_t n_cols)
        : n_rows_(n_rows), n_cols_(n_cols)
    {
        if (n_cols > std::numeric_limits<std::size_t>::max() - 63)
            throw std::length_error("bit table column count overflows");
        // Round up to whole uint64_t words
        words_per_row_ = (n_cols + 63) / 64;
        // Pad to cache line (64 bytes = 8 words of uint64_t)
        stride_words_ = (words_per_row_ + 7) & ~std::size_t{7};

        if (stride_words_ && n_rows > std::numeric_limits<std::size_t>::max() / sizeof(std::uint64_t) / stride_words_)
            throw std::length_error("bit table allocation size overflows");
        if (stride_words_ == 0 || n_rows == 0) return;
        data_ = simd::aligned_alloc_u64(stride_words_ * n_rows);
        if (!data_) throw std::bad_alloc();
        std::memset(data_, 0, stride_words_ * n_rows * sizeof(std::uint64_t));
    }

    ~FlatBitTable() {
        simd::aligned_free(data_);
    }

    // Move semantics
    FlatBitTable(FlatBitTable&& other) noexcept
        : n_rows_(other.n_rows_), n_cols_(other.n_cols_),
          words_per_row_(other.words_per_row_), stride_words_(other.stride_words_),
          data_(other.data_)
    {
        other.data_ = nullptr;
        other.n_rows_ = other.n_cols_ = other.words_per_row_ = other.stride_words_ = 0;
    }

    FlatBitTable& operator=(FlatBitTable&& other) noexcept {
        if (this != &other) {
            simd::aligned_free(data_);
            n_rows_ = other.n_rows_;
            n_cols_ = other.n_cols_;
            words_per_row_ = other.words_per_row_;
            stride_words_ = other.stride_words_;
            data_ = other.data_;
            other.data_ = nullptr;
            other.n_rows_ = other.n_cols_ = other.words_per_row_ = other.stride_words_ = 0;
        }
        return *this;
    }

    // No copy (large data)
    FlatBitTable(const FlatBitTable&) = delete;
    FlatBitTable& operator=(const FlatBitTable&) = delete;

    // --- Accessors ---

    std::size_t n_rows() const noexcept { return n_rows_; }
    std::size_t n_cols() const noexcept { return n_cols_; }
    std::size_t words_per_row() const noexcept { return words_per_row_; }
    std::size_t stride_words() const noexcept { return stride_words_; }

    /// Get pointer to the start of row r (64-byte aligned).
    std::uint64_t* row_ptr(std::size_t r) noexcept {
        return data_ + r * stride_words_;
    }
    const std::uint64_t* row_ptr(std::size_t r) const noexcept {
        return data_ + r * stride_words_;
    }

    /// Set a single bit.
    void set(std::size_t row, std::size_t col) noexcept {
        data_[row * stride_words_ + col / 64] |= std::uint64_t{1} << (col % 64);
    }

    /// Test a single bit.
    bool test(std::size_t row, std::size_t col) const noexcept {
        return (data_[row * stride_words_ + col / 64] >> (col % 64)) & 1;
    }

    /// XOR row `src_row` into a destination buffer using SIMD.
    void xor_row_into(std::size_t src_row, std::uint64_t* dst) const noexcept {
        simd::xor_words(dst, row_ptr(src_row), words_per_row_);
    }

    /// Zero out a destination buffer of words_per_row_ words.
    void zero_buf(std::uint64_t* dst) const noexcept {
        std::memset(dst, 0, words_per_row_ * sizeof(std::uint64_t));
    }

private:
    std::size_t n_rows_ = 0;
    std::size_t n_cols_ = 0;
    std::size_t words_per_row_ = 0;   ///< Actual words needed (no padding).
    std::size_t stride_words_ = 0;    ///< Words per row including cache-line padding.
    std::uint64_t* data_ = nullptr;   ///< Contiguous 64-byte aligned storage.
};

} // namespace qepg_bits

#endif // FLAT_BIT_TABLE_HPP
