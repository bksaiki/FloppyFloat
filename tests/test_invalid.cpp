/*******************************************************************************
 * Apache License, Version 2.0
 * Copyright (c) 2025 chciken/Niko Zurstraßen
 *
 * Tests against golden references for functions that are not supported
 * by Berkeley SoftFloat.
 ******************************************************************************/

#include <gtest/gtest.h>

#include <array>
#include <limits>

#include "floppy_float.h"
#include "utils.h"

using namespace FfUtils;

TEST(InvalidTests, RoundingMode) {
  FloppyFloat fpu;
  fpu.SetupToRiscv();
  f32 a(5.0f), b(3.0f), c{1.0f};
  i32 d(5);
  fpu.rounding_mode = (Vfpu::RoundingMode)-1;

  ASSERT_THROW(fpu.Add<f32>(a, b), std::runtime_error);
  ASSERT_THROW(fpu.Sub<f32>(a, b), std::runtime_error);
  ASSERT_THROW(fpu.Mul<f32>(a, b), std::runtime_error);
  ASSERT_THROW(fpu.Div<f32>(a, b), std::runtime_error);
  ASSERT_THROW(fpu.Sqrt<f32>(a), std::runtime_error);
  ASSERT_THROW(fpu.Fma<f32>(a, b, c), std::runtime_error);
  ASSERT_NO_THROW(fpu.EqQuiet<f32>(a, b));
  ASSERT_NO_THROW(fpu.LeQuiet<f32>(a, b));
  ASSERT_NO_THROW(fpu.LtQuiet<f32>(a, b));
  ASSERT_NO_THROW(fpu.EqSignaling<f32>(a, b));
  ASSERT_NO_THROW(fpu.LeSignaling<f32>(a, b));
  ASSERT_NO_THROW(fpu.LtSignaling<f32>(a, b));
  ASSERT_NO_THROW(fpu.MinimumNumber<f32>(a, b));
  ASSERT_NO_THROW(fpu.MaximumNumber<f32>(a, b));
  ASSERT_NO_THROW(fpu.Minx86<f32>(a, b));
  ASSERT_NO_THROW(fpu.Maxx86<f32>(a, b));
  ASSERT_NO_THROW(fpu.F16ToF32((f16)a));
  ASSERT_NO_THROW(fpu.F16ToF64((f16)a));
  ASSERT_THROW(fpu.F32ToI32(a), std::runtime_error);
  ASSERT_THROW(fpu.F32ToI64(a), std::runtime_error);
  ASSERT_THROW(fpu.F32ToU32(a), std::runtime_error);
  ASSERT_THROW(fpu.F32ToU64(a), std::runtime_error);
  ASSERT_THROW(fpu.F32ToF16(a), std::runtime_error);
  ASSERT_THROW(fpu.F64ToI32(a), std::runtime_error);
  ASSERT_THROW(fpu.F64ToI64(a), std::runtime_error);
  ASSERT_THROW(fpu.F64ToU32(a), std::runtime_error);
  ASSERT_THROW(fpu.F64ToU64(a), std::runtime_error);
  ASSERT_THROW(fpu.F64ToF16(a), std::runtime_error);
  ASSERT_THROW(fpu.F64ToF32(a), std::runtime_error);
  ASSERT_THROW(fpu.I32ToF16(d), std::runtime_error);
  ASSERT_THROW(fpu.I32ToF32(d), std::runtime_error);
  ASSERT_NO_THROW(fpu.I32ToF64(d));
  ASSERT_THROW(fpu.U32ToF32(d), std::runtime_error);
  ASSERT_NO_THROW(fpu.U32ToF64(d));
  // ASSERT_THROW(fpu.I64ToF32(d), std::runtime_error); // TODO
  // ASSERT_THROW(fpu.I64ToF64(d), std::runtime_error); // TODO
  ASSERT_THROW(fpu.U64ToF32(d), std::runtime_error);
  // ASSERT_THROW(fpu.U64ToF64(d), std::runtime_error); // TODO
}

TEST(InvalidTests, NanPropagation) {
  FloppyFloat fpu;
  fpu.SetupToRiscv();
  fpu.nan_propagation_scheme = (Vfpu::NanPropagationSchemes)-1;
  const f32 qnanff = CreateQnanWithPayload<f32>(0xff);

  ASSERT_THROW(fpu.Sqrt<f32>(qnanff), std::runtime_error);
  ASSERT_THROW(fpu.Add<f32>(qnanff, qnanff), std::runtime_error);
  ASSERT_THROW(fpu.Fma<f32>(qnanff, qnanff, qnanff), std::runtime_error);
  ASSERT_THROW(fpu.F16ToF32(qnanff), std::runtime_error);
  ASSERT_THROW(fpu.F16ToF64(qnanff), std::runtime_error);
  ASSERT_THROW(fpu.F32ToF64(qnanff), std::runtime_error);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}