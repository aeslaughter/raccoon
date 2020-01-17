//* This file is part of the RACCOON application
//* being developed at Dolbow lab at Duke University
//* http://dolbow.pratt.duke.edu

#pragma once

#include "ADComputeStressBase.h"

#define usingCNHPK1StressMembers                                                                   \
  usingComputeStressBaseMembers;                                                                   \
  using CNHPK1Stress<compute_stage>::_elasticity_tensor;                                           \
  using CNHPK1Stress<compute_stage>::_F;

template <ComputeStage>
class CNHPK1Stress;

declareADValidParams(CNHPK1Stress);

template <ComputeStage compute_stage>
class CNHPK1Stress : public ADComputeStressBase<compute_stage>
{
public:
  static InputParameters validParams();

  CNHPK1Stress(const InputParameters & parameters);

protected:
  virtual void computeQpStress() override;

  /// Elasticity tensor material property
  const ADMaterialProperty(RankFourTensor) & _elasticity_tensor;

  /// deformation gradient
  const ADMaterialProperty(RankTwoTensor) & _F;

  usingComputeStressBaseMembers;
};