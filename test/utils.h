#ifndef __MINI_BENCH_UTILS_H__
#define __MINI_BENCH_UTILS_H__

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <sstream>

#define STREAM_STR(__stream)                                                   \
  (static_cast<std::ostringstream &>(std::ostringstream() __stream).str())

#define RZ_STR_(x) #x
#define RZ_STR(x) RZ_STR_(x)
#define RZ_CAT_(a, b) a##b
#define RZ_CAT(a, b) RZ_CAT_(a, b)

#define RZ_CHECK(__cond, __printf_args...)                                     \
  ((__cond)                                                                    \
       ? (void)0                                                               \
       : (fprintf(stderr, __FILE__ ":" RZ_STR(__LINE__) ": check \"" RZ_STR(   \
                              __cond) "\" failed:" __printf_args),             \
          fprintf(stderr, "\n"),                                               \
          throw std::runtime_error(                                            \
              STREAM_STR(<< "check failed with errno: " << strerror(errno)))))

template <typename T> T default_rand_gen() { return static_cast<T>(rand()) / RAND_MAX * 10; }
template <typename T>
void host_fill_rand(T *ptr, size_t size,
                    std::function<T(void)> rand_gen = default_rand_gen<T>) {
  for (int i = 0; i < size; ++i) {
    ptr[i] = rand_gen();
  }
}

template <typename T>
bool test_allclose(const T *actual, const T *desired, size_t size,
                   T rtol = 1e-3, T atol = 1e-3, bool print_diff = true) {
  T max_diff = 0;
  bool ok = true;
  for (int i = 0; i < size; ++i) {
    if (std::fabs(actual[i] - desired[i]) >
        std::fabs(desired[i]) * rtol + atol) {
      ok = false;
      max_diff = std::max(max_diff, std::fabs(actual[i] - desired[i]));
    }
  }
  if (!ok && print_diff)
    std::cerr << "max_diff: " << max_diff << std::endl;
  return ok;
}

template <typename T>
std::ostream& dump_array(std::ostream& os, const T* arr, size_t size) {
  os << "[";
  for (int i = 0; i < size; ++i) {
    if (i > 0)
      os << ", ";
    os << arr[i];
  }
  os << "]";
  return os;
}

class DeferGuard {
public:
  template <typename F> DeferGuard(F func) : func(func) {}

  ~DeferGuard() { func(); }

private:
  std::function<void(void)> func;
};

#define DEFER(__actions...)                                                    \
  DeferGuard RZ_CAT(__guard, __LINE__) = [&]() -> void { __actions; };

#endif