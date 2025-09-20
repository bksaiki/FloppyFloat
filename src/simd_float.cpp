#include "simd_float.h"

#include <experimental/simd>

namespace stdx = std::experimental;

// TODO: Remove once SIMD supports the standard float types.
using f32 = float;
using f64 = double;

template <typename T>
using fvec = stdx::native_simd<T>;

template <typename T>
using fmask = stdx::native_simd_mask<T>;

template <typename T>
using nl = std::numeric_limits<T>;

fvec<f32> vqnan32;
const fvec<f32> vmin32([](auto i [[maybe_unused]]) { return nl<f32>::min(); });

fvec<f64> vqnan64;
const fvec<f64> vmin64([](auto i [[maybe_unused]]) { return nl<f64>::min(); });

template <typename FT>
constexpr auto VGetMin();

template <>
constexpr auto VGetMin<f32>() {
  return vmin32;
}

template <>
constexpr auto VGetMin<f64>() {
  return vmin64;
}

template <typename FT>
constexpr auto VGetQnan();

template <>
constexpr auto VGetQnan<f32>() {
  return vqnan32;
}

template <>
constexpr auto VGetQnan<f64>() {
  return vqnan64;
}

template <typename T>
constexpr auto VIsNan(T a) {
  return a != a;
}
template <typename FT>
constexpr auto VIsInf(FT a) {
  static_assert(stdx::is_simd_v<FT>);
  FT res = a - a;
  return (a == a) && (res != res);
}

template <typename T>
constexpr auto VIsInfOrNan(T& a) {
  T res = a - a;
  return res != res;
}

// Returns "false" for NaN.
template <typename T>
constexpr auto VIsNonZero(T a) {
  return (a != -a) && (a == a);
}

// Returns "false" for NaN.
template <typename T>
constexpr auto VIsZero(T a) {
  return a == -a;
}

template <typename T>
fmask<typename T::value_type> VIsSnan(T& a);

fmask<f32> VIsSnan(fvec<f32>& a) {
  fmask<f32> m;
  for (size_t i = 0; i < fmask<f32>::size(); ++i) {
    FfUtils::f32 af{a[i]};
    m[i] = FfUtils::IsSnan(af);
  }
  return m;
}

fmask<f64> VIsSnan(fvec<f64>& a) {
  fmask<f64> m;
  for (size_t i = 0; i < fmask<f64>::size(); ++i) {
    FfUtils::f64 af{a[i]};
    m[i] = FfUtils::IsSnan(af);
  }
  return m;
}

// Vectorized 2Sum algorithm which determines the exact residual of an addition.
// May not work in cases that cause intermediate overflows (e.g., 65504.f16 + -48.f16).
// Prefer the Fast2Sum algorithm for these cases.
template <typename FT>
constexpr FT VTwoSum(FT a, FT b, FT c) {
  FT ad = c - b;
  FT bd = c - ad;
  FT da = ad - a;
  FT db = bd - b;
  FT r = da + db;
  return r;
}

// Vectorized Fast2Sum algorithm.
template <typename FT>
constexpr FT VFastTwoSum(FT a, FT b, FT c) {
  FT a_abs = a;
  FT b_abs = b;
  stdx::where(a < 0, a_abs) = -a_abs;
  stdx::where(b < 0, b_abs) = -b_abs;
  FT x = a;
  FT y = b;
  stdx::where(a_abs < b_abs, x) = b;
  stdx::where(a_abs < b_abs, y) = a;
  FT r = (c - x) - y;
  return r;
}

stdx::rebind_simd_t<f64, fvec<f32>> VUpMul(fvec<f32>& a, fvec<f32>& b, fvec<f32>& c) {
  auto a64 = stdx::simd_cast<stdx::rebind_simd_t<f64, fvec<f32>>>(a);
  auto b64 = stdx::simd_cast<stdx::rebind_simd_t<f64, fvec<f32>>>(b);
  auto c64 = stdx::simd_cast<stdx::rebind_simd_t<f64, fvec<f32>>>(c);
  auto r = a64 * b64 - c64;
  return r;
}

// Note: The algorithm does not work
// for r < 2**-968 = 2**(edmin + pd + 1) = 4.008...
fvec<f64> VUpMul(fvec<f64>& a, fvec<f64>& b, fvec<f64>& c) {
  auto r = fma(a, b, -c);
  return r;
}

template <>
void SimdFloat::SetQnan<f32>(typename FfUtils::FloatToUint<f32>::type val) {
  FloppyFloat::SetQnan<f32>(val);
  for (size_t i = 0; i < stdx::native_simd<f32>::size(); ++i) {
    vqnan32[i] = std::bit_cast<f32>(qnan32_);
  }
}

template <>
void SimdFloat::SetQnan<f64>(typename FfUtils::FloatToUint<f64>::type val) {
  FloppyFloat::SetQnan<f64>(val);
  for (size_t i = 0; i < stdx::native_simd<f64>::size(); ++i) {
    vqnan64[i] = std::bit_cast<f64>(qnan64_);
  }
}

template <typename FT>
constexpr fvec<FT> vfma(fvec<FT>& a) {
  fvec<FT> r{};
  for (size_t i = 0; i < fvec<FT>::size(); ++i)
    r[i] = std::fma(a[i]);
  return r;
}

template <typename FT>
constexpr fvec<FT> vsqrt(fvec<FT>& a) {
  fvec<FT> r{};
  for (size_t i = 0; i < fvec<FT>::size(); ++i)
    r[i] = std::sqrt(a[i]);
  return r;
}

SimdFloat::SimdFloat() : FloppyFloat() {
  SimdFloat::SetQnan<f32>(std::bit_cast<FfUtils::u32>(qnan32_));
  SimdFloat::SetQnan<f64>(std::bit_cast<FfUtils::u64>(qnan64_));
}

template void SimdFloat::VAdd<float>(float* pa, float* pb, float* dest, size_t len);
template void SimdFloat::VAdd<f64>(f64* pa, f64* pb, f64* dest, size_t len);

template <typename FT>
void SimdFloat::VAdd(FT* pa, FT* pb, FT* dest, size_t len) {
  if (rounding_mode != kRoundTiesToEven) [[unlikely]] {
    for (size_t i = 0; i < len; ++i) {
      dest[i] = FloppyFloat::Add<FT>(pa[i], pb[i]);
    }
    return;
  }

  size_t ind = 0;
  while ((ind + stdx::native_simd<f32>::size()) <= len) {
    fvec<FT> a, b, c;
    a.copy_from(&pa[ind], stdx::element_aligned);
    b.copy_from(&pb[ind], stdx::element_aligned);

    c = a + b;

    if (stdx::any_of(VIsInfOrNan(c))) [[unlikely]] {
      if (stdx::any_of(VIsInf(c) && !VIsInf(a) && !VIsInf(b))) {
        SetOverflow();
        SetInexact();
      }
      if (stdx::any_of(VIsNan(c) && VIsInf(a) && VIsInf(b)))
        SetInvalid();
      if (stdx::any_of(VIsSnan(a) || VIsSnan(b)))
        SetInvalid();
      stdx::where(c != c, c) = VGetQnan<FT>();
    }
    if (!inexact) [[unlikely]] {
      // If one input is NaN or ±infinity, the residual "r" will be a
      // qNaN, and no inexact flag is set.
      auto r = VFastTwoSum<fvec<FT>>(a, b, c);
      if (stdx::any_of(VIsNonZero(r)))
        SetInexact();
    }
    c.copy_to(&dest[ind], stdx::element_aligned);
    ind += fvec<FT>::size();
  }

  for (; ind < len; ind++)
    dest[ind] = FloppyFloat::Add<FT, kRoundTiesToEven>(pa[ind], pb[ind]);
}

template void SimdFloat::VSub<f32>(f32* pa, f32* pb, f32* dest, size_t len);
template void SimdFloat::VSub<f64>(f64* pa, f64* pb, f64* dest, size_t len);

template <typename FT>
void SimdFloat::VSub(FT* pa, FT* pb, FT* dest, size_t len) {
  if (rounding_mode != kRoundTiesToEven) [[unlikely]] {
    for (size_t i = 0; i < len; ++i) {
      dest[i] = FloppyFloat::Sub<FT>(pa[i], pb[i]);
    }
    return;
  }

  size_t ind = 0;
  while ((ind + stdx::native_simd<FT>::size()) <= len) {
    fvec<FT> a, b, c;
    a.copy_from(&pa[ind], stdx::element_aligned);
    b.copy_from(&pb[ind], stdx::element_aligned);

    c = a - b;
    if (stdx::any_of(VIsInfOrNan(c))) [[unlikely]] {
      if (stdx::any_of(VIsInf(c) && !VIsInf(a) && !VIsInf(b))) {
        SetOverflow();
        SetInexact();
      }
      if (stdx::any_of(VIsNan(c) && VIsInf(a) && VIsInf(b)))
        SetInvalid();
      if (stdx::any_of(VIsSnan(a) || VIsSnan(b)))
        SetInvalid();
      stdx::where(c != c, c) = VGetQnan<FT>();
    }
    if (!inexact) [[unlikely]] {
      // If one input is NaN or ±infinity, the residual "r" will be a
      // qNaN, and no inexact flag is set.
      auto r = VTwoSum<fvec<FT>>(a, -b, c);
      if (stdx::any_of(VIsNonZero(r)))
        SetInexact();
    }
    c.copy_to(&dest[ind], stdx::element_aligned);
    ind += fvec<FT>::size();
  }

  for (; ind < len; ind++)
    dest[ind] = FloppyFloat::Sub<FT, kRoundTiesToEven>(pa[ind], pb[ind]);
}

template void SimdFloat::VMul<f32>(f32* pa, f32* pb, f32* dest, size_t len);
// template void SimdFloat::VMul<f64>(f64* pa, f64* pb, f64* dest, size_t len);

template <typename FT>
void SimdFloat::VMul(FT* pa, FT* pb, FT* dest, size_t len) {
  if (rounding_mode != kRoundTiesToEven) [[unlikely]] {
    for (size_t i = 0; i < len; ++i) {
      dest[i] = FloppyFloat::Mul<FT>(pa[i], pb[i]);
    }
    return;
  }

  size_t ind = 0;
  while ((ind + stdx::native_simd<f32>::size()) <= len) {
    fvec<FT> a, b, c;
    a.copy_from(&pa[ind], stdx::element_aligned);
    b.copy_from(&pb[ind], stdx::element_aligned);

    c = a * b;
    if (stdx::any_of(VIsInfOrNan(c))) [[unlikely]] {
      if (stdx::any_of(VIsInf(c) && !VIsInf(a) && !VIsInf(b))) {
        SetOverflow();
        SetInexact();
      }
      if (stdx::any_of(VIsSnan(a) || VIsSnan(b)))
        SetInvalid();
      if (stdx::any_of(VIsZero(a) && VIsInf(b)))
        SetInvalid();
      if (stdx::any_of(VIsInf(a) && VIsZero(b)))
        SetInvalid();
      stdx::where(c != c, c) = VGetQnan<FT>();
    }

    // If one input is NaN or ±infinity, the residual "r" will be a qNaN,
    // and no inexact or underflow flag is set.
    if (!inexact) [[unlikely]] {
      auto r = VUpMul(a, b, c);
      if (stdx::any_of(VIsNonZero(r)))
        SetInexact();
    }
    if (!underflow) {
      auto is_small = c < VGetMin<FT>() && c > -VGetMin<FT>();
      if (stdx::any_of(is_small)) {
        auto r = VUpMul(a, b, c);
        auto tmp = stdx::__proposed::static_simd_cast<stdx::rebind_simd_t<f64, decltype(is_small)>>(is_small);
        if (stdx::any_of(VIsNonZero(r) && tmp))
          SetUnderflow();
      }
    }
    c.copy_to(&dest[ind], stdx::element_aligned);
    ind += fvec<FT>::size();
  }

  for (; ind < len; ++ind)
    dest[ind] = FloppyFloat::Mul<FT, kRoundTiesToEven>(pa[ind], pb[ind]);
}

template void SimdFloat::VDiv<f32>(f32* pa, f32* pb, f32* dest, size_t len);
template void SimdFloat::VDiv<f64>(f64* pa, f64* pb, f64* dest, size_t len);

template <typename FT>
void SimdFloat::VDiv(FT* pa, FT* pb, FT* dest, size_t len) {
  if (rounding_mode != kRoundTiesToEven) [[unlikely]] {
    for (size_t i = 0; i < len; ++i) {
      dest[i] = FloppyFloat::Div<FT>(pa[i], pb[i]);
    }
    return;
  }

  size_t ind = 0;
  while ((ind + stdx::native_simd<FT>::size()) <= len) {
    fvec<FT> a, b, c;
    a.copy_from(&pa[ind], stdx::element_aligned);
    b.copy_from(&pb[ind], stdx::element_aligned);

    c = a / b;
    if (stdx::any_of(VIsInfOrNan(c))) [[unlikely]] {
      if (stdx::any_of(VIsInf(c) && !VIsInf(a) && VIsZero(b)))
        SetDivisionByZero();
      if (stdx::any_of(VIsInf(c) && !VIsInf(a) && !VIsInf(b) && !VIsZero(b))) {
        SetOverflow();
        SetInexact();
      }
      if (stdx::any_of(VIsSnan(a) || VIsSnan(b)))
        SetInvalid();
      stdx::where(c != c, c) = VGetQnan<FT>();
    }

    if (!inexact) [[unlikely]] {
      for (size_t i = 0; i < fvec<FT>::size(); ++i)
        FloppyFloat::Div(pa[ind + i], pb[ind + i]);
    }
    if (!underflow) {
      auto is_small = c < VGetMin<FT>() && c > -VGetMin<FT>();
      if (stdx::any_of(is_small)) {
        for (size_t i = 0; i < fvec<FT>::size(); ++i)
          FloppyFloat::Div(pa[ind + i], pb[ind + i]);
      }
    }
    c.copy_to(&dest[ind], stdx::element_aligned);
    ind += fvec<FT>::size();
  }

  for (; ind < len; ++ind)
    dest[ind] = FloppyFloat::Div<FT, kRoundTiesToEven>(pa[ind], pb[ind]);
}

template void SimdFloat::VSqrt<f32>(f32* pa, f32* dest, size_t len);
template void SimdFloat::VSqrt<f64>(f64* pa, f64* dest, size_t len);

template <typename FT>
void SimdFloat::VSqrt(FT* pa, FT* dest, size_t len) {
  if (rounding_mode != kRoundTiesToEven) [[unlikely]] {
    for (size_t i = 0; i < len; ++i) {
      dest[i] = FloppyFloat::Sqrt<FT>(pa[i]);
    }
    return;
  }

  size_t ind = 0;
  while ((ind + stdx::native_simd<FT>::size()) <= len) {
    fvec<FT> a, b;
    a.copy_from(&pa[ind], stdx::element_aligned);

    b = vsqrt(a);
    if (stdx::any_of(VIsNan(b))) [[unlikely]] {
      if (stdx::any_of(VIsSnan(a)) || stdx::any_of(a < 0))
        SetInvalid();
      stdx::where(b != b, b) = VGetQnan<FT>();
    }

    if (!inexact) [[unlikely]] {
      for (size_t i = 0; i < fvec<FT>::size(); ++i)
        FloppyFloat::Sqrt(pa[ind + i]);
    }
    b.copy_to(&dest[ind], stdx::element_aligned);
    ind += fvec<FT>::size();
  }

  for (; ind < len; ++ind)
    dest[ind] = FloppyFloat::Sqrt(pa[ind]);
}

template <typename FT>
void SimdFloat::VFma(FT* pa, FT* pb, FT* pc, FT* dest, size_t len) {
  if (rounding_mode != kRoundTiesToEven) [[unlikely]] {
    for (size_t i = 0; i < len; ++i) {
      dest[i] = FloppyFloat::Fma<FT>(pa[i], pb[i], pc[i]);
    }
    return;
  }

  size_t ind = 0;
  while ((ind + stdx::native_simd<FT>::size()) <= len) {
    fvec<FT> a, b, c, d;
    a.copy_from(&pa[ind], stdx::element_aligned);
    b.copy_from(&pb[ind], stdx::element_aligned);
    c.copy_from(&pc[ind], stdx::element_aligned);

    d = vfma(a, b, c);
    if (unlikely(stdx::any_of(VIsInfOrNan(d)))) {
      if (stdx::any_of(VIsInf(d) && !VIsInf(a) && !VIsInf(b) && !VIsInf(c))) {
        SetOverflow();
        SetInexact();
      }
      if (stdx::any_of(VIsSnan(a) || VIsSnan(b) || VIsSnan(c)))
        SetInvalid();
      if (stdx::any_of((VIsNan(d) && !VIsNan(a)) && !VIsNan(b) && !VIsNan(c)))
        SetInvalid();
      stdx::where(d != d, d) = VGetQnan<FT>();
    }

    if (!inexact) [[unlikely]] {
      for (size_t i = 0; i < fvec<FT>::size(); ++i)
        FloppyFloat::Fma(pa[ind + i], pb[ind + i], pc[ind + i]);
    }
    if (!underflow) {
      auto is_small = d < VGetMin<FT>() && d > -VGetMin<FT>();
      if (stdx::any_of(is_small)) {
        for (size_t i = 0; i < fvec<FT>::size(); ++i)
          FloppyFloat::Fma(pa[ind + i], pb[ind + i], pc[ind + i]);
      }
    }
    d.copy_to(&dest[ind], stdx::element_aligned);
    ind += fvec<FT>::size();
  }

  for (; ind < len; ++ind)
    dest[ind] = FloppyFloat::Fma(pa[ind], pb[ind], pc[ind]);
}

void SimdFloat::SetupToRiscv() {
  FloppyFloat::SetupToRiscv();
  SimdFloat::SetQnan<f32>(std::bit_cast<FfUtils::u32>(qnan32_));
  SimdFloat::SetQnan<f64>(std::bit_cast<FfUtils::u64>(qnan64_));
}
