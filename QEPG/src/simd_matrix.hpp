#ifndef SIMD_MATRIX_HPP
#define SIMD_MATRIX_HPP
#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <boost/dynamic_bitset.hpp>

// Platform-specific includes for aligned allocation
#ifdef _MSC_VER
    #include <malloc.h>  // _aligned_malloc, _aligned_free
#else
    #include <cstdlib>   // aligned_alloc, free
#endif

// SIMD intrinsics - available on both MSVC and GCC/Clang
#if defined(_MSC_VER) || defined(__AVX2__)
    #include <immintrin.h>
#endif

namespace SIMD {

// Alignment for AVX2 (32 bytes = 256 bits)
constexpr size_t SIMD_ALIGN = 32;

// Number of uint64_t words per AVX2 register
constexpr size_t WORDS_PER_AVX2 = 4;

/**
 * @brief Aligned memory allocation wrapper (cross-platform)
 */
inline uint64_t* aligned_alloc_u64(size_t count, size_t alignment) {
#ifdef _MSC_VER
    return static_cast<uint64_t*>(_aligned_malloc(count * sizeof(uint64_t), alignment));
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, count * sizeof(uint64_t)) != 0) {
        return nullptr;
    }
    return static_cast<uint64_t*>(ptr);
#endif
}

/**
 * @brief Aligned memory deallocation wrapper (cross-platform)
 */
inline void aligned_free_u64(uint64_t* ptr) {
#ifdef _MSC_VER
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

/**
 * @brief Row-major bit matrix with SIMD-aligned storage for fast XOR operations.
 *
 * This class stores a bit matrix in a contiguous, aligned memory layout
 * optimized for SIMD XOR operations. Each row is padded to a multiple of
 * SIMD_ALIGN bytes to enable aligned loads/stores.
 */
class SIMDMatrix {
public:
    /**
     * @brief Construct an empty SIMD matrix
     */
    SIMDMatrix() : rows_(0), cols_(0), words_per_row_(0), data_(nullptr) {}

    /**
     * @brief Construct a SIMD matrix with given dimensions
     * @param rows Number of rows
     * @param cols Number of columns (bits per row)
     */
    SIMDMatrix(size_t rows, size_t cols);

    /**
     * @brief Destructor - frees aligned memory
     */
    ~SIMDMatrix();

    // Disable copy (move-only for efficiency)
    SIMDMatrix(const SIMDMatrix&) = delete;
    SIMDMatrix& operator=(const SIMDMatrix&) = delete;

    // Enable move
    SIMDMatrix(SIMDMatrix&& other) noexcept;
    SIMDMatrix& operator=(SIMDMatrix&& other) noexcept;

    /**
     * @brief Copy data from a vector of boost::dynamic_bitset
     * @param src Source bitset matrix
     */
    void copy_from_bitset_matrix(const std::vector<boost::dynamic_bitset<>>& src);

    /**
     * @brief SIMD-accelerated XOR: dst ^= row[row_idx]
     * @param dst Destination buffer (must be aligned to SIMD_ALIGN)
     * @param row_idx Row index to XOR into dst
     */
    void xor_row_into(uint64_t* dst, size_t row_idx) const;

    /**
     * @brief SIMD-accelerated batch XOR: dst ^= row[idx0] ^ row[idx1] ^ ...
     * @param dst Destination buffer (must be aligned to SIMD_ALIGN)
     * @param row_indices Array of row indices
     * @param count Number of rows to XOR
     */
    void xor_rows_into(uint64_t* dst, const size_t* row_indices, size_t count) const;

    /**
     * @brief Zero-initialize an aligned result buffer
     * @param dst Destination buffer (must be aligned to SIMD_ALIGN)
     */
    void zero_buffer(uint64_t* dst) const;

    /**
     * @brief Allocate an aligned buffer for results
     * @return Pointer to aligned buffer (caller must free with aligned_free_u64)
     */
    uint64_t* allocate_result_buffer() const;

    // Accessors
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    size_t words_per_row() const { return words_per_row_; }
    const uint64_t* data() const { return data_; }
    const uint64_t* row_ptr(size_t row_idx) const { return data_ + row_idx * words_per_row_; }

private:
    size_t rows_;           // Number of rows
    size_t cols_;           // Number of columns (bits)
    size_t words_per_row_;  // uint64_t words per row (padded for alignment)
    uint64_t* data_;        // Aligned storage (row-major)
};

}  // namespace SIMD

#endif  // SIMD_MATRIX_HPP
