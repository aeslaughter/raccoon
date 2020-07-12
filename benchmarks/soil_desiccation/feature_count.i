[Problem]
  solve = false
[]

[UserObjects]
  [./solution]
    type = SolutionUserObject
    mesh = 'visualize.e'
  [../]
[]

[Mesh]
  [./fmg]
    type = FileMeshGenerator
    file = 'visualize.e'
  [../]
[]

[AuxVariables]
  [./d]
  [../]
  [./unique_grains]
    order = CONSTANT
    family = MONOMIAL
  [../]
  [./halos]
    order = CONSTANT
    family = MONOMIAL
  [../]
[]

[AuxKernels]
  [./read_d]
    type = SolutionAux
    variable = 'd'
    from_variable = 'd'
    solution = 'solution'
    execute_on = 'INITIAL TIMESTEP_BEGIN'
  [../]
  [./unique_grains]
    type = FeatureFloodCountAux
    variable = 'unique_grains'
    flood_counter = 'feature_counter'
    execute_on = 'INITIAL TIMESTEP_BEGIN'
    field_display = UNIQUE_REGION
  [../]
  [./halos]
    type = FeatureFloodCountAux
    variable = 'halos'
    flood_counter = 'feature_counter'
    execute_on = 'INITIAL TIMESTEP_BEGIN'
    field_display = HALOS
  [../]
[]

[VectorPostprocessors]
  [./feature_volumes]
    type = FeatureVolumeVectorPostprocessor
    flood_counter = feature_counter
    execute_on = 'initial timestep_end'
  [../]
[]

[Postprocessors]
  [./feature_counter]
    type = FeatureFloodCount
    variable = 'd'
    compute_var_to_feature_map = true
    use_less_than_threshold_comparison = false
    compute_halo_maps = true
    execute_on = 'INITIAL TIMESTEP_END'
  [../]
  [./volume]
    type = VolumePostprocessor
    execute_on = 'INITIAL'
  [../]
  [./volume_fraction]
    type = FeatureVolumeFraction
    mesh_volume = volume
    feature_volumes = feature_volumes
    execute_on = 'INITIAL TIMESTEP_END'
  [../]
[]

[Executioner]
  type = Transient
  dt = 1e-3
  end_time = 0.04
[]

[Outputs]
  exodus = true
  csv = true
[]
