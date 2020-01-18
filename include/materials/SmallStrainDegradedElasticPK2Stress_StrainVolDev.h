//* This file is part of the RACCOON application
//* being developed at Dolbow lab at Duke University
//* http://dolbow.pratt.duke.edu

#pragma once

#include "ADDegradedElasticStressBase.h"

template <ComputeStage>
class SmallStrainDegradedElasticPK2Stress_StrainVolDev;

declareADValidParams(SmallStrainDegradedElasticPK2Stress_StrainVolDev);

template <ComputeStage compute_stage>
class SmallStrainDegradedElasticPK2Stress_StrainVolDev : public ADDegradedElasticStressBase<compute_stage>
{
public:
  static InputParameters validParams();

  SmallStrainDegradedElasticPK2Stress_StrainVolDev(const InputParameters & parameters);

protected:
  virtual void computeQpStress() override;

  usingDegradedStressBaseMembers
};
