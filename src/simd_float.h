#pragma once
/**************************************************************************************************
 * Apache License, Version 2.0
 * Copyright (c) 2024 chciken/Niko Zurstraßen
 **************************************************************************************************/

#include "floppy_float.h"
#include "utils.h"

class SimdFloat : public FloppyFloat {
 public:
  SimdFloat();

  template <typename FT>
  void SetQnan(typename FfUtils::FloatToUint<FT>::type val);

  template <typename FT>
  void VAdd(FT* pa, FT* pb, FT* dest, size_t len);

  template <typename FT>
  void VSub(FT* pa, FT* pb, FT* dest, size_t len);

  template <typename FT>
  void VMul(FT* pa, FT* pb, FT* dest, size_t len);

  template <typename FT>
  void VDiv(FT* pa, FT* pb, FT* dest, size_t len);

  template <typename FT>
  void VSqrt(FT* pa, FT* dest, size_t len);

  template <typename FT>
  void VFma(FT* pa, FT* pb, FT* pc, FT* dest, size_t len);

  void SetupToRiscv();

private:
  void SetupToArm();  // Currently not implemented/working.
  void SetupToX86();  // Currently not implemented/working.
};