#pragma once
/**************************************************************************************************
 * Apache License, Version 2.0
 * Copyright (c) 2024 chciken/Niko Zurstraßen
 **************************************************************************************************/

#include "floppy_float.h"
#include "utils.h"

class SimdFloat : public FloppyFloat {
  using f32 = FfUtils::f32;

 public:
  SimdFloat() {};

  void Vadd(f32* pa, f32* pb, f32* dest, size_t len);
  void VSub(f32* pa, f32* pb, f32* dest, size_t len);
  void VMul(f32* pa, f32* pb, f32* dest, size_t len);
};