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

template <typename FT>
constexpr FT VTwoSum(FT a, FT b, FT c) {
  FT ad = c - b;
  FT bd = c - ad;
  FT da = ad - a;
  FT db = bd - b;
  FT r = da + db;
  return r;
}

template <typename FT>
constexpr FT VFastTwoSum(FT a, FT b, FT c) {
  const bool no_swap = std::fabs(a) > std::fabs(b);
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

void SimdFloat::Vadd(f32* pa, f32* pb, f32* dest, size_t len) {
  if (rounding_mode != kRoundTiesToEven) [[unlikely]] {
    for (size_t i = 0; i < len; ++i) {
      dest[i] = FloppyFloat::Add<f32>(pa[i], pb[i]);
    }
    return;
  }

  size_t ind = 0;
  while ((ind + stdx::native_simd<f32>::size()) <= len) {
    fvec<::f32> a, b, c;
    a.copy_from(&pa[ind], stdx::element_aligned);
    b.copy_from(&pb[ind], stdx::element_aligned);

    c = a + b;

    if (stdx::any_of(VIsInfOrNan(c))) [[unlikely]] {
      if (stdx::any_of(VIsInf(c) && !VIsInf(a) && !VIsInf(b))) {
        overflow = true;
        inexact = true;
      }
      if (stdx::any_of(VIsNan(c) && VIsInf(a) && VIsInf(b)))
        invalid = true;
      if (stdx::any_of(VIsSnan(a) || VIsSnan(b)))
        invalid = true;
      stdx::where(c != c, c) = vqnan32;
    }
    if (!inexact) [[unlikely]] {
      // If one input is NaN or ±infinity, the residual "r" will be a
      // qNaN, and no inexact flag is set.
      auto r = VTwoSum<fvec<::f32>>(a, b, c);
      if (stdx::any_of(VIsNonZero(r)))
        inexact = true;
    }
    c.copy_to(&dest[ind], stdx::element_aligned);
    ind += fvec<f32>::size();
  }

  for (; ind < len; ind++)
    dest[ind] = FloppyFloat::Add<f32, kRoundTiesToEven>(pa[ind], pb[ind]);
}

void SimdFloat::VSub(f32* pa, f32* pb, f32* dest, size_t len) {
  if (rounding_mode != kRoundTiesToEven) [[unlikely]] {
    for (size_t i = 0; i < len; ++i) {
      dest[i] = FloppyFloat::Sub<f32>(pa[i], pb[i]);
    }
    return;
  }

  size_t ind = 0;
  while ((ind + stdx::native_simd<f32>::size()) <= len) {
    fvec<::f32> a, b, c;
    a.copy_from(&pa[ind], stdx::element_aligned);
    b.copy_from(&pb[ind], stdx::element_aligned);

    c = a - b;
    if (stdx::any_of(VIsInfOrNan(c))) [[unlikely]] {
      if (stdx::any_of(VIsInf(c) && !VIsInf(a) && !VIsInf(b))) {
        overflow = true;
        inexact = true;
      }
      if (stdx::any_of(VIsNan(c) && VIsInf(a) && VIsInf(b)))
        invalid = true;
      if (stdx::any_of(VIsSnan(a) || VIsSnan(b)))
        invalid = true;
      stdx::where(c != c, c) = vqnan32;
    }
    if (!inexact) [[unlikely]] {
      // If one input is NaN or ±infinity, the residual "r" will be a
      // qNaN, and no inexact flag is set.
      auto r = VTwoSum<fvec<::f32>>(a, -b, c);
      if (stdx::any_of(VIsNonZero(r)))
        inexact = true;
    }
    c.copy_to(&dest[ind], stdx::element_aligned);
    ind += fvec<f32>::size();
  }

  for (; ind < len; ind++)
    dest[ind] = FloppyFloat::Sub<f32, kRoundTiesToEven>(pa[ind], pb[ind]);
}

void SimdFloat::VMul(f32* pa, f32* pb, f32* dest, size_t len) {
  if (rounding_mode != kRoundTiesToEven) [[unlikely]] {
    for (size_t i = 0; i < len; ++i) {
      dest[i] = FloppyFloat::Mul<f32>(pa[i], pb[i]);
    }
    return;
  }

  size_t ind = 0;
  while ((ind + stdx::native_simd<f32>::size()) <= len) {
    fvec<::f32> a, b, c;
    a.copy_from(&pa[ind], stdx::element_aligned);
    b.copy_from(&pb[ind], stdx::element_aligned);

    c = a * b;
    if (stdx::any_of(VIsInfOrNan(c))) [[unlikely]] {
      if (stdx::any_of(VIsInf(c) && !VIsInf(a) && !VIsInf(b))) {
        overflow = true;
        inexact = true;
      }
      if (stdx::any_of(VIsNan(a) || VIsNan(b)))
        invalid = true;
      if (stdx::any_of(VIsNonZero(a) && VIsInf(b)))
        invalid = true;
      stdx::where(c != c, c) = vqnan32;
    }

    // If one input is NaN or ±infinity, the residual "r" will be a qNaN,
    // and no inexact or underflow flag is set.
    if (!inexact) [[unlikely]] {
      auto r = VUpMul(a, b, c);
      if (stdx::any_of(VIsNonZero(r)))
        inexact = true;
    }
    if (!underflow) {
      auto is_small = c < vmin32 && c > -vmin32;
      if (stdx::any_of(is_small)) {
        auto r = VUpMul(a, b, c);
        auto tmp = stdx::__proposed::static_simd_cast<stdx::rebind_simd_t<f64, decltype(is_small)>>(is_small);
        if (stdx::any_of(VIsNonZero(r) && tmp))
          underflow = true;
      }
    }
    c.copy_to(&dest[ind], stdx::element_aligned);
    ind += fvec<f32>::size();
  }

  for (; ind < len; ++ind)
    dest[ind] = FloppyFloat::Mul<f32, kRoundTiesToEven>(pa[ind], pb[ind]);
}