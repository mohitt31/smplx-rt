#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <optional>
#include <type_traits>

namespace smplxrt {
// Bounded, allocation-free SPSC queue. Capacity must be a power of two.
// Producer and consumer own different cursors, so hot-path operations are lock-free.
template <typename T, std::size_t Capacity>
class SpscRing {
  static_assert(Capacity >= 2 && (Capacity & (Capacity - 1)) == 0, "Capacity must be a power of two");
 public:
  bool try_push(T value) noexcept(std::is_nothrow_move_assignable<T>::value) {
    const auto write = write_.load(std::memory_order_relaxed);
    const auto next = (write + 1) & kMask;
    if (next == read_.load(std::memory_order_acquire)) return false;
    storage_[write] = std::move(value);
    write_.store(next, std::memory_order_release);
    return true;
  }
  std::optional<T> try_pop() noexcept(std::is_nothrow_move_constructible<T>::value) {
    const auto read = read_.load(std::memory_order_relaxed);
    if (read == write_.load(std::memory_order_acquire)) return std::nullopt;
    T value = std::move(storage_[read]);
    read_.store((read + 1) & kMask, std::memory_order_release);
    return value;
  }
  bool empty() const noexcept { return read_.load(std::memory_order_acquire) == write_.load(std::memory_order_acquire); }
 private:
  static constexpr std::size_t kMask = Capacity - 1;
  alignas(64) std::array<T, Capacity> storage_{};
  alignas(64) std::atomic<std::size_t> write_{0};
  alignas(64) std::atomic<std::size_t> read_{0};
};
}  // namespace smplxrt
