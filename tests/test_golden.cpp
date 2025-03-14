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

TEST(GoldenTests, Minx86f32) {
  FloppyFloat fpu;
  fpu.SetupToX86();
  const f32 qnanff = CreateQnanWithPayload<f32>(0xff);
  const f32 snanff = CreateSnanWithPayload<f32>(0xff);
  const f32 infinity = std::numeric_limits<f32>::infinity();
  ASSERT_EQ(fpu.Minx86<f32>(1.0f32, 0.0f32), 0.f32);
  ASSERT_EQ(fpu.Minx86<f32>(0.0f32, 1.0f32), 0.f32);
  ASSERT_EQ(fpu.Minx86<f32>(-1.0f32, 0.0f32), -1.0f32);
  ASSERT_EQ(fpu.Minx86<f32>(infinity, -infinity), -infinity);
  ASSERT_EQ(fpu.Minx86<f32>(-infinity, infinity), -infinity);
  ASSERT_EQ(fpu.Minx86<f32>(-infinity, -infinity), -infinity);
  ASSERT_EQ(fpu.Minx86<f32>(infinity, infinity), infinity);
  ASSERT_EQ(std::bit_cast<u32>(fpu.Minx86<f32>(-0.0f32, +0.0f32)), std::bit_cast<u32>(+0.0f32));
  ASSERT_EQ(std::bit_cast<u32>(fpu.Minx86<f32>(+0.0f32, -0.0f32)), std::bit_cast<u32>(-0.0f32));
  ASSERT_EQ(std::bit_cast<u32>(fpu.Minx86<f32>(-0.0f32, -0.0f32)), std::bit_cast<u32>(-0.0f32));
  ASSERT_EQ(std::bit_cast<u32>(fpu.Minx86<f32>(+0.0f32, +0.0f32)), std::bit_cast<u32>(+0.0f32));
  ASSERT_EQ(std::bit_cast<u32>(fpu.Minx86<f32>(qnanff, +5.0f32)), std::bit_cast<u32>(+5.0f32));
  ASSERT_EQ(std::bit_cast<u32>(fpu.Minx86<f32>(+5.0f32, qnanff)), std::bit_cast<u32>(qnanff));
  ASSERT_EQ(std::bit_cast<u32>(fpu.Minx86<f32>(qnanff, qnanff)), std::bit_cast<u32>(qnanff));
  ASSERT_EQ(fpu.invalid, true);
  fpu.invalid = false;
  ASSERT_EQ(std::bit_cast<u32>(fpu.Minx86<f32>(snanff, snanff)), std::bit_cast<u32>(snanff));
  ASSERT_EQ(fpu.invalid, true);
}

TEST(GoldenTests, Maxx86f32) {
  FloppyFloat fpu;
  fpu.SetupToX86();
  const f32 qnanff = CreateQnanWithPayload<f32>(0xff);
  const f32 snanff = CreateSnanWithPayload<f32>(0xff);
  const f32 infinity = std::numeric_limits<f32>::infinity();
  ASSERT_EQ(fpu.Maxx86<f32>(1.0f32, 0.0f32), 1.f32);
  ASSERT_EQ(fpu.Maxx86<f32>(0.0f32, 1.0f32), 1.f32);
  ASSERT_EQ(fpu.Maxx86<f32>(-1.0f32, 0.0f32), 0.0f32);
  ASSERT_EQ(fpu.Maxx86<f32>(infinity, -infinity), infinity);
  ASSERT_EQ(fpu.Maxx86<f32>(-infinity, infinity), infinity);
  ASSERT_EQ(fpu.Maxx86<f32>(-infinity, -infinity), -infinity);
  ASSERT_EQ(fpu.Maxx86<f32>(infinity, infinity), infinity);
  ASSERT_EQ(std::bit_cast<u32>(fpu.Maxx86<f32>(-0.0f32, +0.0f32)), std::bit_cast<u32>(+0.0f32));
  ASSERT_EQ(std::bit_cast<u32>(fpu.Maxx86<f32>(+0.0f32, -0.0f32)), std::bit_cast<u32>(-0.0f32));
  ASSERT_EQ(std::bit_cast<u32>(fpu.Maxx86<f32>(-0.0f32, -0.0f32)), std::bit_cast<u32>(-0.0f32));
  ASSERT_EQ(std::bit_cast<u32>(fpu.Maxx86<f32>(+0.0f32, +0.0f32)), std::bit_cast<u32>(+0.0f32));
  ASSERT_EQ(std::bit_cast<u32>(fpu.Maxx86<f32>(qnanff, +5.0f32)), std::bit_cast<u32>(+5.0f32));
  ASSERT_EQ(std::bit_cast<u32>(fpu.Maxx86<f32>(+5.0f32, qnanff)), std::bit_cast<u32>(qnanff));
  ASSERT_EQ(std::bit_cast<u32>(fpu.Maxx86<f32>(qnanff, qnanff)), std::bit_cast<u32>(qnanff));
  ASSERT_EQ(fpu.invalid, true);
  fpu.invalid = false;
  ASSERT_EQ(std::bit_cast<u32>(fpu.Maxx86<f32>(snanff, snanff)), std::bit_cast<u32>(snanff));
  ASSERT_EQ(fpu.invalid, true);
}

TEST(GoldenTests, MininumNumberRiscvf32) {
  FloppyFloat fpu;
  fpu.SetupToRiscv();
  const f32 qnanff = CreateQnanWithPayload<f32>(0xff);
  const f32 snanff = CreateSnanWithPayload<f32>(0xff);
  const f32 infinity = std::numeric_limits<f32>::infinity();
  ASSERT_EQ(fpu.MinimumNumber<f32>(1.0f32, 0.0f32), 0.f32);
  ASSERT_EQ(fpu.MinimumNumber<f32>(0.0f32, 1.0f32), 0.f32);
  ASSERT_EQ(fpu.MinimumNumber<f32>(-1.0f32, 0.0f32), -1.0f32);
  ASSERT_EQ(fpu.MinimumNumber<f32>(infinity, -infinity), -infinity);
  ASSERT_EQ(fpu.MinimumNumber<f32>(-infinity, infinity), -infinity);
  ASSERT_EQ(fpu.MinimumNumber<f32>(-infinity, -infinity), -infinity);
  ASSERT_EQ(fpu.MinimumNumber<f32>(infinity, infinity), infinity);
  ASSERT_EQ(std::bit_cast<u32>(fpu.MinimumNumber<f32>(-0.0f32, +0.0f32)), std::bit_cast<u32>(-0.0f32));
  ASSERT_EQ(std::bit_cast<u32>(fpu.MinimumNumber<f32>(+0.0f32, -0.0f32)), std::bit_cast<u32>(-0.0f32));
  ASSERT_EQ(std::bit_cast<u32>(fpu.MinimumNumber<f32>(-0.0f32, -0.0f32)), std::bit_cast<u32>(-0.0f32));
  ASSERT_EQ(std::bit_cast<u32>(fpu.MinimumNumber<f32>(+0.0f32, +0.0f32)), std::bit_cast<u32>(+0.0f32));
  ASSERT_EQ(std::bit_cast<u32>(fpu.MinimumNumber<f32>(qnanff, +5.0f32)), std::bit_cast<u32>(+5.0f32));
  ASSERT_EQ(std::bit_cast<u32>(fpu.MinimumNumber<f32>(+5.0f32, qnanff)), std::bit_cast<u32>(+5.0f32));
  ASSERT_EQ(std::bit_cast<u32>(fpu.MinimumNumber<f32>(qnanff, qnanff)), std::bit_cast<u32>(fpu.GetQnan<f32>()));
  ASSERT_EQ(fpu.invalid, false);
  ASSERT_EQ(std::bit_cast<u32>(fpu.MinimumNumber<f32>(snanff, snanff)), std::bit_cast<u32>(fpu.GetQnan<f32>()));
  ASSERT_EQ(fpu.invalid, true);
}

TEST(GoldenTests, MaximumNumberRiscvf32) {
  FloppyFloat fpu;
  fpu.SetupToRiscv();
  const f32 qnanff = CreateQnanWithPayload<f32>(0xff);
  const f32 snanff = CreateSnanWithPayload<f32>(0xff);
  const f32 infinity = std::numeric_limits<f32>::infinity();
  ASSERT_EQ(fpu.MaximumNumber<f32>(1.0f32, 0.0f32), 1.f32);
  ASSERT_EQ(fpu.MaximumNumber<f32>(0.0f32, 1.0f32), 1.f32);
  ASSERT_EQ(fpu.MaximumNumber<f32>(-1.0f32, 0.0f32), 0.0f32);
  ASSERT_EQ(fpu.MaximumNumber<f32>(infinity, -infinity), infinity);
  ASSERT_EQ(fpu.MaximumNumber<f32>(-infinity, infinity), infinity);
  ASSERT_EQ(fpu.MaximumNumber<f32>(-infinity, -infinity), -infinity);
  ASSERT_EQ(fpu.MaximumNumber<f32>(infinity, infinity), infinity);
  ASSERT_EQ(std::bit_cast<u32>(fpu.MaximumNumber<f32>(-0.0f32, +0.0f32)), std::bit_cast<u32>(+0.0f32));
  ASSERT_EQ(std::bit_cast<u32>(fpu.MaximumNumber<f32>(+0.0f32, -0.0f32)), std::bit_cast<u32>(+0.0f32));
  ASSERT_EQ(std::bit_cast<u32>(fpu.MaximumNumber<f32>(-0.0f32, -0.0f32)), std::bit_cast<u32>(-0.0f32));
  ASSERT_EQ(std::bit_cast<u32>(fpu.MaximumNumber<f32>(+0.0f32, +0.0f32)), std::bit_cast<u32>(+0.0f32));
  ASSERT_EQ(std::bit_cast<u32>(fpu.MaximumNumber<f32>(qnanff, +5.0f32)), std::bit_cast<u32>(+5.0f32));
  ASSERT_EQ(std::bit_cast<u32>(fpu.MaximumNumber<f32>(+5.0f32, qnanff)), std::bit_cast<u32>(+5.0f32));
  ASSERT_EQ(std::bit_cast<u32>(fpu.MaximumNumber<f32>(qnanff, qnanff)), std::bit_cast<u32>(fpu.GetQnan<f32>()));
  ASSERT_EQ(fpu.invalid, false);
  ASSERT_EQ(std::bit_cast<u32>(fpu.MaximumNumber<f32>(snanff, snanff)), std::bit_cast<u32>(fpu.GetQnan<f32>()));
  ASSERT_EQ(fpu.invalid, true);
}

TEST(GoldenTests, MininumNumberRiscvf64) {
  FloppyFloat fpu;
  fpu.SetupToRiscv();
  const f64 qnanff = CreateQnanWithPayload<f64>(0xff);
  const f64 snanff = CreateSnanWithPayload<f64>(0xff);
  const f64 infinity = std::numeric_limits<f64>::infinity();
  ASSERT_EQ(fpu.MinimumNumber<f64>(1.0f64, 0.0f64), 0.f64);
  ASSERT_EQ(fpu.MinimumNumber<f64>(0.0f64, 1.0f64), 0.f64);
  ASSERT_EQ(fpu.MinimumNumber<f64>(-1.0f64, 0.0f64), -1.0f64);
  ASSERT_EQ(fpu.MinimumNumber<f64>(infinity, -infinity), -infinity);
  ASSERT_EQ(fpu.MinimumNumber<f64>(-infinity, infinity), -infinity);
  ASSERT_EQ(fpu.MinimumNumber<f64>(-infinity, -infinity), -infinity);
  ASSERT_EQ(fpu.MinimumNumber<f64>(infinity, infinity), infinity);
  ASSERT_EQ(std::bit_cast<u64>(fpu.MinimumNumber<f64>(-0.0f64, +0.0f64)), std::bit_cast<u64>(-0.0f64));
  ASSERT_EQ(std::bit_cast<u64>(fpu.MinimumNumber<f64>(+0.0f64, -0.0f64)), std::bit_cast<u64>(-0.0f64));
  ASSERT_EQ(std::bit_cast<u64>(fpu.MinimumNumber<f64>(-0.0f64, -0.0f64)), std::bit_cast<u64>(-0.0f64));
  ASSERT_EQ(std::bit_cast<u64>(fpu.MinimumNumber<f64>(+0.0f64, +0.0f64)), std::bit_cast<u64>(+0.0f64));
  ASSERT_EQ(std::bit_cast<u64>(fpu.MinimumNumber<f64>(qnanff, +5.0f64)), std::bit_cast<u64>(+5.0f64));
  ASSERT_EQ(std::bit_cast<u64>(fpu.MinimumNumber<f64>(+5.0f64, qnanff)), std::bit_cast<u64>(+5.0f64));
  ASSERT_EQ(std::bit_cast<u64>(fpu.MinimumNumber<f64>(qnanff, qnanff)), std::bit_cast<u64>(fpu.GetQnan<f64>()));
  ASSERT_EQ(fpu.invalid, false);
  ASSERT_EQ(std::bit_cast<u64>(fpu.MinimumNumber<f64>(snanff, snanff)), std::bit_cast<u64>(fpu.GetQnan<f64>()));
  ASSERT_EQ(fpu.invalid, true);
}

TEST(GoldenTests, MaximumNumberRiscvf64) {
  FloppyFloat fpu;
  fpu.SetupToRiscv();
  const f64 qnanff = CreateQnanWithPayload<f64>(0xff);
  const f64 snanff = CreateSnanWithPayload<f64>(0xff);
  const f64 infinity = std::numeric_limits<f64>::infinity();
  ASSERT_EQ(fpu.MaximumNumber<f64>(1.0f64, 0.0f64), 1.f64);
  ASSERT_EQ(fpu.MaximumNumber<f64>(0.0f64, 1.0f64), 1.f64);
  ASSERT_EQ(fpu.MaximumNumber<f64>(-1.0f64, 0.0f64), 0.0f64);
  ASSERT_EQ(fpu.MaximumNumber<f64>(infinity, -infinity), infinity);
  ASSERT_EQ(fpu.MaximumNumber<f64>(-infinity, infinity), infinity);
  ASSERT_EQ(fpu.MaximumNumber<f64>(-infinity, -infinity), -infinity);
  ASSERT_EQ(fpu.MaximumNumber<f64>(infinity, infinity), infinity);
  ASSERT_EQ(std::bit_cast<u64>(fpu.MaximumNumber<f64>(-0.0f64, +0.0f64)), std::bit_cast<u64>(+0.0f64));
  ASSERT_EQ(std::bit_cast<u64>(fpu.MaximumNumber<f64>(+0.0f64, -0.0f64)), std::bit_cast<u64>(+0.0f64));
  ASSERT_EQ(std::bit_cast<u64>(fpu.MaximumNumber<f64>(-0.0f64, -0.0f64)), std::bit_cast<u64>(-0.0f64));
  ASSERT_EQ(std::bit_cast<u64>(fpu.MaximumNumber<f64>(+0.0f64, +0.0f64)), std::bit_cast<u64>(+0.0f64));
  ASSERT_EQ(std::bit_cast<u64>(fpu.MaximumNumber<f64>(qnanff, +5.0f64)), std::bit_cast<u64>(+5.0f64));
  ASSERT_EQ(std::bit_cast<u64>(fpu.MaximumNumber<f64>(+5.0f64, qnanff)), std::bit_cast<u64>(+5.0f64));
  ASSERT_EQ(std::bit_cast<u64>(fpu.MaximumNumber<f64>(qnanff, qnanff)), std::bit_cast<u64>(fpu.GetQnan<f64>()));
  ASSERT_EQ(fpu.invalid, false);
  ASSERT_EQ(std::bit_cast<u64>(fpu.MaximumNumber<f64>(snanff, snanff)), std::bit_cast<u64>(fpu.GetQnan<f64>()));
  ASSERT_EQ(fpu.invalid, true);
}

TEST(GoldenTests, ClassRiscvf32) {
  FloppyFloat fpu;
  fpu.SetupToRiscv();
  const f32 qnanff = CreateQnanWithPayload<f32>(0xff);
  const f32 snanff = CreateSnanWithPayload<f32>(0xff);
  const f32 infinity = std::numeric_limits<f32>::infinity();
  ASSERT_EQ(fpu.Class<f32>(+0.0f32), 1 << Vfpu::ClassIndex::kPosZero);
  ASSERT_EQ(fpu.Class<f32>(-0.0f32), 1 << Vfpu::ClassIndex::kNegZero);
  ASSERT_EQ(fpu.Class<f32>(std::nextafterf(+0.0f32, 1.0f32)), 1 << Vfpu::ClassIndex::kPosSubnormal);
  ASSERT_EQ(fpu.Class<f32>(std::nextafterf(-0.0f32, -1.0f32)), 1 << Vfpu::ClassIndex::kNegSubnormal);
  ASSERT_EQ(fpu.Class<f32>(1.0f32), 1 << Vfpu::ClassIndex::kPosNormal);
  ASSERT_EQ(fpu.Class<f32>(-1.0f32), 1 << Vfpu::ClassIndex::kNegNormal);
  ASSERT_EQ(fpu.Class<f32>(infinity), 1 << Vfpu::ClassIndex::kPosInfinity);
  ASSERT_EQ(fpu.Class<f32>(-infinity), 1 << Vfpu::ClassIndex::kNegInfinity);
  ASSERT_EQ(fpu.Class<f32>(qnanff), 1 << Vfpu::ClassIndex::kQNan);
  ASSERT_EQ(fpu.Class<f32>(snanff), 1 << Vfpu::ClassIndex::kSNan);
  ASSERT_EQ(fpu.division_by_zero, false);
  ASSERT_EQ(fpu.inexact, false);
  ASSERT_EQ(fpu.invalid, false);
  ASSERT_EQ(fpu.overflow, false);
  ASSERT_EQ(fpu.underflow, false);
}

TEST(GoldenTests, ClassRiscvf64) {
  FloppyFloat fpu;
  fpu.SetupToRiscv();
  const f64 qnanff = CreateQnanWithPayload<f64>(0xff);
  const f64 snanff = CreateSnanWithPayload<f64>(0xff);
  const f64 infinity = std::numeric_limits<f64>::infinity();
  ASSERT_EQ(fpu.Class<f64>(+0.0f64), 1 << Vfpu::ClassIndex::kPosZero);
  ASSERT_EQ(fpu.Class<f64>(-0.0f64), 1 << Vfpu::ClassIndex::kNegZero);
  ASSERT_EQ(fpu.Class<f64>(std::nextafter(+0.0f64, 1.0f64)), 1 << Vfpu::ClassIndex::kPosSubnormal);
  ASSERT_EQ(fpu.Class<f64>(std::nextafter(-0.0f64, -1.0f64)), 1 << Vfpu::ClassIndex::kNegSubnormal);
  ASSERT_EQ(fpu.Class<f64>(1.0f64), 1 << Vfpu::ClassIndex::kPosNormal);
  ASSERT_EQ(fpu.Class<f64>(-1.0f64), 1 << Vfpu::ClassIndex::kNegNormal);
  ASSERT_EQ(fpu.Class<f64>(infinity), 1 << Vfpu::ClassIndex::kPosInfinity);
  ASSERT_EQ(fpu.Class<f64>(-infinity), 1 << Vfpu::ClassIndex::kNegInfinity);
  ASSERT_EQ(fpu.Class<f64>(qnanff), 1 << Vfpu::ClassIndex::kQNan);
  ASSERT_EQ(fpu.Class<f64>(snanff), 1 << Vfpu::ClassIndex::kSNan);
  ASSERT_EQ(fpu.division_by_zero, false);
  ASSERT_EQ(fpu.inexact, false);
  ASSERT_EQ(fpu.invalid, false);
  ASSERT_EQ(fpu.overflow, false);
  ASSERT_EQ(fpu.underflow, false);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}