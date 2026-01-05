#include "simd_matrix.hpp"
#include <cstring>  // memset, memcpy
#include <stdexcept>

namespace SIMD {

SIMDMatrix::SIMDMatrix(size_t rows, size_t cols)
    : rows_(rows), cols_(cols), data_(nullptr)
{
    if (rows == 0 || cols == 0) {
        words_per_row_ = 0;
        return;
    }

    // Calculate words needed per row (ceil division)
    size_t min_words = (cols + 63) / 64;

    // Pad to multiple of WORDS_PER_AVX2 (4) for aligned SIMD operations
    words_per_row_ = ((min_words + WORDS_PER_AVX2 - 1) / WORDS_PER_AVX2) * WORDS_PER_AVX2;

    // Allocate aligned memory
    size_t total_words = rows_ * words_per_row_;
    data_ = aligned_alloc_u64(total_words, SIMD_ALIGN);

    if (!data_) {
        throw std::bad_alloc();
    }

    // Zero-initialize
    std::memset(data_, 0, total_words * sizeof(uint64_t));
}

SIMDMatrix::~SIMDMatrix() {
    if (data_) {
        aligned_free_u64(data_);
        data_ = nullptr;
    }
}

SIMDMatrix::SIMDMatrix(SIMDMatrix&& other) noexcept
    : rows_(other.rows_)
    , cols_(other.cols_)
    , words_per_row_(other.words_per_row_)
    , data_(other.data_)
{
    other.rows_ = 0;
    other.cols_ = 0;
    other.words_per_row_ = 0;
    other.data_ = nullptr;
}

SIMDMatrix& SIMDMatrix::operator=(SIMDMatrix&& other) noexcept {
    if (this != &other) {
        if (data_) {
            aligned_free_u64(data_);
        }
        rows_ = other.rows_;
        cols_ = other.cols_;
        words_per_row_ = other.words_per_row_;
        data_ = other.data_;

        other.rows_ = 0;
        other.cols_ = 0;
        other.words_per_row_ = 0;
        other.data_ = nullptr;
    }
    return *this;
}

void SIMDMatrix::copy_from_bitset_matrix(const std::vector<boost::dynamic_bitset<>>& src) {
    if (src.empty()) return;

    // Verify dimensions match
    if (src.size() != rows_) {
        throw std::invalid_argument("Row count mismatch in copy_from_bitset_matrix");
    }

    // Copy each row
    for (size_t r = 0; r < rows_; ++r) {
        const auto& bitset = src[r];
        uint64_t* dst_row = data_ + r * words_per_row_;

        // Zero the row first (in case of padding)
        std::memset(dst_row, 0, words_per_row_ * sizeof(uint64_t));

        // Copy blocks from bitset
        // boost::dynamic_bitset stores data in blocks (typically uint64_t on 64-bit systems)
        size_t num_blocks = bitset.num_blocks();
        if (num_blocks > 0) {
            // Use boost's to_block_range to extract the raw blocks
            boost::to_block_range(bitset, dst_row);
        }
    }
}

void SIMDMatrix::xor_row_into(uint64_t* dst, size_t row_idx) const {
    const uint64_t* src = data_ + row_idx * words_per_row_;

#if defined(__AVX2__) || (defined(_MSC_VER) && defined(__AVX2__))
    // AVX2 path: process 4 uint64_t (256 bits) at a time
    size_t i = 0;
    for (; i + WORDS_PER_AVX2 <= words_per_row_; i += WORDS_PER_AVX2) {
        __m256i d = _mm256_load_si256(reinterpret_cast<const __m256i*>(dst + i));
        __m256i s = _mm256_load_si256(reinterpret_cast<const __m256i*>(src + i));
        _mm256_store_si256(reinterpret_cast<__m256i*>(dst + i), _mm256_xor_si256(d, s));
    }
    // Handle remaining words (shouldn't happen if properly padded, but safe)
    for (; i < words_per_row_; ++i) {
        dst[i] ^= src[i];
    }
#else
    // Scalar fallback
    for (size_t i = 0; i < words_per_row_; ++i) {
        dst[i] ^= src[i];
    }
#endif
}

void SIMDMatrix::xor_rows_into(uint64_t* dst, const size_t* row_indices, size_t count) const {
    for (size_t i = 0; i < count; ++i) {
        xor_row_into(dst, row_indices[i]);
    }
}

void SIMDMatrix::zero_buffer(uint64_t* dst) const {
#if defined(__AVX2__) || (defined(_MSC_VER) && defined(__AVX2__))
    __m256i zero = _mm256_setzero_si256();
    size_t i = 0;
    for (; i + WORDS_PER_AVX2 <= words_per_row_; i += WORDS_PER_AVX2) {
        _mm256_store_si256(reinterpret_cast<__m256i*>(dst + i), zero);
    }
    for (; i < words_per_row_; ++i) {
        dst[i] = 0;
    }
#else
    std::memset(dst, 0, words_per_row_ * sizeof(uint64_t));
#endif
}

uint64_t* SIMDMatrix::allocate_result_buffer() const {
    return aligned_alloc_u64(words_per_row_, SIMD_ALIGN);
}

}  // namespace SIMD
