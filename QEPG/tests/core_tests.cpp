#include "dynamic_bitset.hpp"
#include "flat_bit_table.hpp"
#include "sampler.hpp"
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>

void require(bool condition) {
    if (!condition) throw std::runtime_error("native regression failed");
}

template <typename Exception, typename F>
void must_throw(F&& fn) {
    try { fn(); } catch (const Exception&) { return; }
    throw std::runtime_error("expected native exception was not thrown");
}

int main() {
    using qepg_bits::DynamicBitset;
    for (std::size_t n : {0, 1, 63, 64, 65, 127, 128, 129}) {
        DynamicBitset a(n), b(n);
        if (n) { a.set(n-1); b.set(n-1); }
        require((a ^ b).count() == 0);
        require((a & b).count() == (n ? 1 : 0));
        must_throw<std::out_of_range>([&] { a.test(n); });
        DynamicBitset other(n+1);
        must_throw<std::invalid_argument>([&] { a ^= other; });
        must_throw<std::invalid_argument>([&] { a &= other; });
        must_throw<std::invalid_argument>([&] { a |= other; });
        qepg_bits::FlatBitTable empty(0, n);
        require(empty.n_rows() == 0);
    }
    qepg_bits::FlatBitTable table(3, 129);
    table.set(0, 0); table.set(0, 64); table.set(0, 128);
    auto* buffer = simd::aligned_alloc_u64(table.stride_words());
    require(buffer != nullptr);
    simd::zero_words(buffer, table.stride_words());
    table.xor_row_into(0, buffer);
    require(buffer[0] == 1 && buffer[1] == 1 && buffer[2] == 1);
    table.xor_row_into(0, buffer);
    require(buffer[0] == 0 && buffer[1] == 0 && buffer[2] == 0);
    simd::aligned_free(buffer);
    qepg_bits::FlatBitTable moved(std::move(table));
    require(moved.n_rows() == 3 && table.n_rows() == 0);
    must_throw<std::length_error>([] { qepg_bits::FlatBitTable too_large(SIZE_MAX, 128); });
    SAMPLE::Xoshiro256pp rng(12345);
    std::array<int, 3> counts{};
    for (int i = 0; i < 300000; ++i) ++counts[rng.bounded(3)];
    for (auto count : counts) require(std::abs(count - 100000) < 2000);
    std::cout << "Native bitset, matrix, and RNG checks passed\n";
}
