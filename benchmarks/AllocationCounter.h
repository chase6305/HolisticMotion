#pragma once

// Single-translation-unit benchmark support. Measure allocation calls separately
// from timing; malloc wrapping is enabled only for supported static builds.
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <new>

namespace hm_benchmark {
inline std::atomic<bool> counting{false};
inline std::atomic<std::size_t> allocations{0};
#ifdef HOLISTICMOTION_COUNT_MALLOC
inline constexpr const char *allocation_kind = "malloc";
#else
inline constexpr const char *allocation_kind = "cpp_new";
#endif
inline void StartCounting() {
    allocations.store(0, std::memory_order_relaxed);
    counting.store(true, std::memory_order_relaxed);
}
inline std::size_t StopCounting() {
    counting.store(false, std::memory_order_relaxed);
    return allocations.load(std::memory_order_relaxed);
}
} // namespace hm_benchmark

#ifdef HOLISTICMOTION_COUNT_MALLOC
extern "C" void *__real_malloc(std::size_t);
extern "C" void *__wrap_malloc(std::size_t size) {
    if (hm_benchmark::counting.load(std::memory_order_relaxed))
        hm_benchmark::allocations.fetch_add(1, std::memory_order_relaxed);
    return __real_malloc(size);
}
#endif
void *operator new(std::size_t size) {
#ifndef HOLISTICMOTION_COUNT_MALLOC
    if (hm_benchmark::counting.load(std::memory_order_relaxed))
        hm_benchmark::allocations.fetch_add(1, std::memory_order_relaxed);
#endif
    if (void *result = std::malloc(std::max(size, std::size_t{1})))
        return result;
    throw std::bad_alloc();
}
void operator delete(void *pointer) noexcept { std::free(pointer); }
void operator delete(void *pointer, std::size_t) noexcept { std::free(pointer); }
