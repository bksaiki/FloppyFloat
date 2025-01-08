/*******************************************************************************
 * Apache License, Version 2.0
 * Copyright (c) 2024 chciken/Niko Zurstraßen
 ******************************************************************************/

#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <functional>
#include <iostream>
#include <type_traits>

#include "float_rng.h"
#include "simd_float.h"

extern "C" {
#include "softfloat.h"
}

using namespace std::placeholders;
using namespace FfUtils;

constexpr i32 kNumIterations = 200000;
constexpr i32 kRngSeed = 42;
constexpr size_t kSimdLength = 10;

SimdFloat ff;

std::array<std::pair<uint_fast8_t, SoftFloat::RoundingMode>, 5> rounding_modes{
    {{::softfloat_round_near_even, SoftFloat::RoundingMode::kRoundTiesToEven},
     {::softfloat_round_near_maxMag, SoftFloat::RoundingMode::kRoundTiesToAway},
     {::softfloat_round_max, SoftFloat::RoundingMode::kRoundTowardPositive},
     {::softfloat_round_min, SoftFloat::RoundingMode::kRoundTowardNegative},
     {::softfloat_round_minMag, SoftFloat::RoundingMode::kRoundTowardZero}}};

template <typename T>
struct FFloatToSFloat;

template <>
struct FFloatToSFloat<float> {
  using type = float32_t;
};
template <>
struct FFloatToSFloat<double> {
  using type = float64_t;
};

template <typename T>
auto ToComparableType(T a) {
  if constexpr (std::is_floating_point<decltype(a)>::value) {
    return std::bit_cast<typename FloatToUint<T>::type>(a);
  } else if constexpr (std::is_same_v<decltype(a), float16_t>) {
    return a.v;
  } else if constexpr (std::is_same_v<decltype(a), float32_t>) {
    return a.v;
  } else if constexpr (std::is_same_v<decltype(a), float64_t>) {
    return a.v;
  } else {
    return a;
  }
}

template <typename T1, typename T2>
void CheckResult(T1 ff_result_u, T2 sf_result_u, size_t i) {
  ASSERT_EQ(ff_result_u, sf_result_u)
    << "Iteration: " << i << ", FF result:" << ff_result_u << ", SF result:" << sf_result_u;
  ASSERT_EQ(ff.invalid, static_cast<bool>(::softfloat_exceptionFlags & ::softfloat_flag_invalid))
    << "Iteration: " << i << ", FF result:" << ff_result_u << ", SF result:" << sf_result_u;
  ASSERT_EQ(ff.division_by_zero, static_cast<bool>(::softfloat_exceptionFlags & ::softfloat_flag_infinite))
    << "Iteration: " << i << ", FF result:" << ff_result_u << ", SF result:" << sf_result_u;
  ASSERT_EQ(ff.overflow, static_cast<bool>(::softfloat_exceptionFlags & ::softfloat_flag_overflow))
    << "Iteration: " << i << ", FF result:" << ff_result_u << ", SF result:" << sf_result_u;
  ASSERT_EQ(ff.underflow, static_cast<bool>(::softfloat_exceptionFlags & ::softfloat_flag_underflow))
    << "Iteration: " << i << ", FF result:" << ff_result_u << ", SF result:" << sf_result_u;
  ASSERT_EQ(ff.inexact, static_cast<bool>(::softfloat_exceptionFlags & ::softfloat_flag_inexact))
    << "Iteration: " << i << ", FF result:" << ff_result_u << ", SF result:" << sf_result_u;
}

template <typename FT, typename FFFUNC, typename SFFUNC, int num_args>
void DoTest(FFFUNC ff_func, SFFUNC sf_func) {

#if defined(ARCH_RISCV)
  ff.SetupToRiscv();
#elif defined(ARCH_X86)
  ff.SetupToX86();
#elif defined(ARCH_ARM)
  ff.SetupToArm();
#endif

  ::softfloat_exceptionFlags = 0;
  ff.ClearFlags();

  FloatRng<FT> float_rng(kRngSeed);
  FT av[kSimdLength] = {};
  std::fill_n(av, kSimdLength, 1.f);
  FT bv[kSimdLength] = {};
  std::fill_n(bv, kSimdLength, 1.f);
  FT cv[kSimdLength] = {0.f};
  FT dv[kSimdLength];
  av[0] = float_rng.Gen();
  typename FFloatToSFloat<FT>::type valuesfa{std::bit_cast<typename FloatToUint<FT>::type>(av[0])};
  bv[0] = float_rng.Gen();
  typename FFloatToSFloat<FT>::type valuesfb{std::bit_cast<typename FloatToUint<FT>::type>(bv[0])};
  cv[0] = float_rng.Gen();
  typename FFloatToSFloat<FT>::type valuesfc{std::bit_cast<typename FloatToUint<FT>::type>(cv[0])};


  for (i32 i = 0; i < kNumIterations; ++i) {
    if constexpr (num_args == 1) {
      auto ff_result = ff_func(av, bv, kSimdLength);
      auto sf_result = sf_func(valuesfa);
      auto ff_result_u = ToComparableType(ff_result);
      auto sf_result_u = ToComparableType(sf_result);
      CheckResult(ff_result_u, sf_result_u, i);
    }
    if constexpr (num_args == 2) {
      //ff_func(av, bv, cv, kSimdLength);
      //ff_func((f32*)&av[0], (f32*)&bv[0], (f32*)&cv[0], kSimdLength);
      ff_func(&av[0], &bv[0], &cv[0], kSimdLength);
      auto sf_result = sf_func(valuesfa, valuesfb);
      auto ff_result_u = ToComparableType(cv[0]);
      auto sf_result_u = ToComparableType(sf_result);
      CheckResult(ff_result_u, sf_result_u, i);
    }
    if constexpr (num_args == 3) {
      auto ff_result = ff_func(av, bv, cv, dv, kSimdLength);
      auto sf_result = sf_func(valuesfa, valuesfb, valuesfc);
      auto ff_result_u = ToComparableType(ff_result);
      auto sf_result_u = ToComparableType(sf_result);
      CheckResult(ff_result_u, sf_result_u, i);
    }

    ::softfloat_exceptionFlags = 0;
    ff.ClearFlags();

    valuesfc = valuesfb;
    cv[0] = bv[0];
    valuesfb = valuesfa;
    bv[0] = av[0];
    av[0]= float_rng.Gen();
    valuesfa.v = std::bit_cast<typename FloatToUint<FT>::type>(av[0]);
  }
}

#if defined(ARCH_RISCV)
  #define TEST_SUITE_NAME SoftFloatSimdFloatRiscvTests
#elif defined(ARCH_X86)
  #define TEST_SUITE_NAME SoftFloatSimdFloatX86Tests
#elif defined(ARCH_ARM)
  #define TEST_SUITE_NAME SoftFloatSimdFloatArmTests
#else
  static_assert(false, "Unknown architecture");
#endif

#define TEST_MACRO_BASE(name, ff_op, sf_op, type, rm, rm_name, nargs, ...)       \
  TEST(TEST_SUITE_NAME, name##rm_name) {                                         \
    ::softfloat_roundingMode = rounding_modes[rm].first;                         \
    ff.rounding_mode = rounding_modes[rm].second;                                \
    auto ff_func = std::bind(ff_op, &ff, _1, _2, _3, _4);                           \
    auto sf_func = std::bind(&::sf_op, __VA_ARGS__);                             \
    DoTest<type, decltype(ff_func), decltype(sf_func), nargs>(ff_func, sf_func); \
  }


#define TEST_MACRO_1(name, ff_op, sf_op, type, rm, rm_name) TEST_MACRO_BASE(name, ff_op, sf_op, type, rm, rm_name, 1, _1)
#define TEST_MACRO_2(name, ff_op, sf_op, type, rm, rm_name) TEST_MACRO_BASE(name, ff_op, sf_op, type, rm, rm_name, 2, _1, _2)
#define TEST_MACRO_3(name, ff_op, sf_op, type, rm, rm_name) TEST_MACRO_BASE(name, ff_op, sf_op, type, rm, rm_name, 3, _1, _2, _3)

TEST_MACRO_2(Addf32, &SimdFloat::VAdd<f32>, f32_add, f32, 0, RoundTiesToEven)
TEST_MACRO_2(Addf32, &SimdFloat::VAdd<f32>, f32_add, f32, 1, RoundTiesToAway)
TEST_MACRO_2(Addf32, &SimdFloat::VAdd<f32>, f32_add, f32, 2, RoundTowardPositive)
TEST_MACRO_2(Addf32, &SimdFloat::VAdd<f32>, f32_add, f32, 3, RoundTowardNegative)
TEST_MACRO_2(Addf32, &SimdFloat::VAdd<f32>, f32_add, f32, 4, RoundTowardZero)
TEST_MACRO_2(Addf64, &SimdFloat::VAdd<f64>, f64_add, f64, 0, RoundTiesToEven)
TEST_MACRO_2(Addf64, &SimdFloat::VAdd<f64>, f64_add, f64, 1, RoundTiesToAway)
TEST_MACRO_2(Addf64, &SimdFloat::VAdd<f64>, f64_add, f64, 2, RoundTowardPositive)
TEST_MACRO_2(Addf64, &SimdFloat::VAdd<f64>, f64_add, f64, 3, RoundTowardNegative)
TEST_MACRO_2(Addf64, &SimdFloat::VAdd<f64>, f64_add, f64, 4, RoundTowardZero)

TEST_MACRO_2(Subf32, &SimdFloat::VSub<f32>, f32_sub, f32, 0, RoundTiesToEven)
TEST_MACRO_2(Subf32, &SimdFloat::VSub<f32>, f32_sub, f32, 1, RoundTiesToAway)
TEST_MACRO_2(Subf32, &SimdFloat::VSub<f32>, f32_sub, f32, 2, RoundTowardPositive)
TEST_MACRO_2(Subf32, &SimdFloat::VSub<f32>, f32_sub, f32, 3, RoundTowardNegative)
TEST_MACRO_2(Subf32, &SimdFloat::VSub<f32>, f32_sub, f32, 4, RoundTowardZero)
TEST_MACRO_2(Subf64, &SimdFloat::VSub<f64>, f64_sub, f64, 0, RoundTiesToEven)
TEST_MACRO_2(Subf64, &SimdFloat::VSub<f64>, f64_sub, f64, 1, RoundTiesToAway)
TEST_MACRO_2(Subf64, &SimdFloat::VSub<f64>, f64_sub, f64, 2, RoundTowardPositive)
TEST_MACRO_2(Subf64, &SimdFloat::VSub<f64>, f64_sub, f64, 3, RoundTowardNegative)
TEST_MACRO_2(Subf64, &SimdFloat::VSub<f64>, f64_sub, f64, 4, RoundTowardZero)

TEST_MACRO_2(Mulf32, &SimdFloat::VMul<f32>, f32_mul, f32, 0, RoundTiesToEven)
TEST_MACRO_2(Mulf32, &SimdFloat::VMul<f32>, f32_mul, f32, 1, RoundTiesToAway)
TEST_MACRO_2(Mulf32, &SimdFloat::VMul<f32>, f32_mul, f32, 2, RoundTowardPositive)
TEST_MACRO_2(Mulf32, &SimdFloat::VMul<f32>, f32_mul, f32, 3, RoundTowardNegative)
TEST_MACRO_2(Mulf32, &SimdFloat::VMul<f32>, f32_mul, f32, 4, RoundTowardZero)
// TEST_MACRO_2(Mulf64, &SimdFloat::VMul<f64>, f64_mul, f64, 0, RoundTiesToEven)
// TEST_MACRO_2(Mulf64, &SimdFloat::VMul<f64>, f64_mul, f64, 1, RoundTiesToAway)
// TEST_MACRO_2(Mulf64, &SimdFloat::VMul<f64>, f64_mul, f64, 2, RoundTowardPositive)
// TEST_MACRO_2(Mulf64, &SimdFloat::VMul<f64>, f64_mul, f64, 3, RoundTowardNegative)
// TEST_MACRO_2(Mulf64, &SimdFloat::VMul<f64>, f64_mul, f64, 4, RoundTowardZero)

TEST_MACRO_2(Divf32, &SimdFloat::VDiv<f32>, f32_div, f32, 0, RoundTiesToEven)
TEST_MACRO_2(Divf32, &SimdFloat::VDiv<f32>, f32_div, f32, 1, RoundTiesToAway)
TEST_MACRO_2(Divf32, &SimdFloat::VDiv<f32>, f32_div, f32, 2, RoundTowardPositive)
TEST_MACRO_2(Divf32, &SimdFloat::VDiv<f32>, f32_div, f32, 3, RoundTowardNegative)
TEST_MACRO_2(Divf32, &SimdFloat::VDiv<f32>, f32_div, f32, 4, RoundTowardZero)
TEST_MACRO_2(Divf64, &SimdFloat::VDiv<f64>, f64_div, f64, 0, RoundTiesToEven)
TEST_MACRO_2(Divf64, &SimdFloat::VDiv<f64>, f64_div, f64, 1, RoundTiesToAway)
TEST_MACRO_2(Divf64, &SimdFloat::VDiv<f64>, f64_div, f64, 2, RoundTowardPositive)
TEST_MACRO_2(Divf64, &SimdFloat::VDiv<f64>, f64_div, f64, 3, RoundTowardNegative)
TEST_MACRO_2(Divf64, &SimdFloat::VDiv<f64>, f64_div, f64, 4, RoundTowardZero)

// TEST_MACRO_1(Sqrtf32, &SimdFloat::VSqrt<f32>, f32_sqrt, f32, 0, RoundTiesToEven)
// TEST_MACRO_1(Sqrtf32, &SimdFloat::VSqrt<f32>, f32_sqrt, f32, 1, RoundTiesToAway)
// TEST_MACRO_1(Sqrtf32, &SimdFloat::VSqrt<f32>, f32_sqrt, f32, 2, RoundTowardPositive)
// TEST_MACRO_1(Sqrtf32, &SimdFloat::VSqrt<f32>, f32_sqrt, f32, 3, RoundTowardNegative)
// TEST_MACRO_1(Sqrtf32, &SimdFloat::VSqrt<f32>, f32_sqrt, f32, 4, RoundTowardZero)
// TEST_MACRO_1(Sqrtf64, &SimdFloat::VSqrt<f64>, f64_sqrt, f64, 0, RoundTiesToEven)
// TEST_MACRO_1(Sqrtf64, &SimdFloat::VSqrt<f64>, f64_sqrt, f64, 1, RoundTiesToAway)
// TEST_MACRO_1(Sqrtf64, &SimdFloat::VSqrt<f64>, f64_sqrt, f64, 2, RoundTowardPositive)
// TEST_MACRO_1(Sqrtf64, &SimdFloat::VSqrt<f64>, f64_sqrt, f64, 3, RoundTowardNegative)
// TEST_MACRO_1(Sqrtf64, &SimdFloat::VSqrt<f64>, f64_sqrt, f64, 4, RoundTowardZero)

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}