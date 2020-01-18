//* This file is part of the RACCOON application
//* being developed at Dolbow lab at Duke University
//* http://dolbow.pratt.duke.edu

#pragma once

#include "ADDegradedElasticStressBase.h"

template <ComputeStage>
class SmallStrainDegradedElasticPK2Stress_StrainSpectral;

declareADValidParams(SmallStrainDegradedElasticPK2Stress_StrainSpectral);

template <ComputeStage compute_stage>
class SmallStrainDegradedElasticPK2Stress_StrainSpectral : public ADDegradedElasticStressBase<compute_stage>
{
public:
  static InputParameters validParams();

  SmallStrainDegradedElasticPK2Stress_StrainSpectral(const InputParameters & parameters);

protected:
  virtual void computeQpStress() override;

private:
  /// positive eigenvalues
  ADRankTwoTensor D_pos;

  usingDegradedStressBaseMembers
};
