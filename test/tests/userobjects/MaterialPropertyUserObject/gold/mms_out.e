CDF      
      
len_string     !   len_line   Q   four      	time_step          len_name   !   num_dim       	num_nodes         num_elem      
num_el_blk        num_node_sets         num_side_sets         num_el_in_blk1        num_nod_per_el1       num_side_ss1      num_side_ss2      num_side_ss3      num_side_ss4      num_nod_ns1       num_nod_ns2       num_nod_ns3       num_nod_ns4       num_nod_var       num_glo_var       num_info  m         api_version       @�
=   version       @�
=   floating_point_word_size            	file_size               int64_status             title         
mms_out.e      maximum_name_length                 !   
time_whole                            T   	eb_status                             	0   eb_prop1               name      ID              	4   	ns_status         	                    	8   ns_prop1      	         name      ID              	H   	ss_status         
                    	X   ss_prop1      
         name      ID              	h   coordx                             	x   coordy                             	�   eb_names                       $      	�   ns_names      	                 �      	�   ss_names      
                 �      
`   
coor_names                         D      
�   node_num_map                          (   connect1                  	elem_type         QUAD4               8   elem_num_map                          H   elem_ss1                          L   side_ss1                          P   elem_ss2                          T   side_ss2                          X   elem_ss3                          \   side_ss3                          `   elem_ss4                          d   side_ss4                          h   node_ns1                          l   node_ns2                          t   node_ns3                          |   node_ns4                          �   vals_nod_var1                                 \   name_nod_var                       $      �   name_glo_var                       $      �   vals_glo_var                             |   info_records                      s�      �                                                                 ?�      ?�                              ?�      ?�                                          bottom                           right                            top                              left                             bottom                           top                              left                             right                                                                                                                                                                           u                                   l2_error                            ####################                                                             # Created by MOOSE #                                                             ####################                                                             ### Command Line Arguments ###                                                    raccoon-opt -i mms.i### Version Info ###                                                                                                                         Framework Information:                                                           MOOSE Version:           git commit b5f2007 on 2020-01-13                        LibMesh Version:         81589c1acb86765a0a1981b7fef5328e91fb92ab                PETSc Version:           3.11.4                                                  SLEPc Version:           3.11.0                                                  Current Time:            Tue Jan 14 14:20:25 2020                                Executable Timestamp:    Tue Jan 14 14:11:23 2020                                                                                                                                                                                                  ### Input File ###                                                                                                                                                []                                                                                 inactive                       = (no_default)                                    initial_from_file_timestep     = LATEST                                          initial_from_file_var          = INVALID                                         element_order                  = AUTO                                            order                          = AUTO                                            side_order                     = AUTO                                            type                           = GAUSS                                         []                                                                                                                                                                [BCs]                                                                                                                                                               [./left]                                                                           boundary                     = left                                              control_tags                 = INVALID                                           displacements                = INVALID                                           enable                       = 1                                                 extra_matrix_tags            = INVALID                                           extra_vector_tags            = INVALID                                           implicit                     = 1                                                 inactive                     = (no_default)                                      isObjectAction               = 1                                                 matrix_tags                  = system                                            type                         = FunctionDirichletBC                               use_displaced_mesh           = 0                                                 variable                     = u                                                 vector_tags                  = nontime                                           diag_save_in                 = INVALID                                           function                     = solution                                          save_in                      = INVALID                                           seed                         = 0                                               [../]                                                                                                                                                             [./right]                                                                          boundary                     = right                                             control_tags                 = INVALID                                           displacements                = INVALID                                           enable                       = 1                                                 extra_matrix_tags            = INVALID                                           extra_vector_tags            = INVALID                                           implicit                     = 1                                                 inactive                     = (no_default)                                      isObjectAction               = 1                                                 matrix_tags                  = system                                            type                         = FunctionDirichletBC                               use_displaced_mesh           = 0                                                 variable                     = u                                                 vector_tags                  = nontime                                           diag_save_in                 = INVALID                                           function                     = solution                                          save_in                      = INVALID                                           seed                         = 0                                               [../]                                                                          []                                                                                                                                                                [Executioner]                                                                      auto_preconditioning           = 1                                               inactive                       = (no_default)                                    isObjectAction                 = 1                                               type                           = Steady                                          accept_on_max_picard_iteration = 0                                               automatic_scaling              = INVALID                                         compute_initial_residual_before_preset_bcs = 0                                   compute_scaling_once           = 1                                               contact_line_search_allowed_lambda_cuts = 2                                      contact_line_search_ltol       = INVALID                                         control_tags                   = (no_default)                                    disable_picard_residual_norm_check = 0                                           enable                         = 1                                               l_abs_tol                      = 1e-50                                           l_max_its                      = 10000                                           l_tol                          = 1e-05                                           line_search                    = default                                         line_search_package            = petsc                                           max_xfem_update                = 4294967295                                      mffd_type                      = wp                                              nl_abs_step_tol                = 1e-50                                           nl_abs_tol                     = 1e-50                                           nl_div_tol                     = -1                                              nl_max_funcs                   = 10000                                           nl_max_its                     = 50                                              nl_rel_step_tol                = 1e-50                                           nl_rel_tol                     = 1e-08                                           num_grids                      = 1                                               petsc_options                  = INVALID                                         petsc_options_iname            = -pc_type                                        petsc_options_value            = lu                                              picard_abs_tol                 = 1e-50                                           picard_force_norms             = 0                                               picard_max_its                 = 1                                               picard_rel_tol                 = 1e-08                                           relaxation_factor              = 1                                               relaxed_variables              = (no_default)                                    restart_file_base              = (no_default)                                    skip_exception_check           = 0                                               snesmf_reuse_base              = 1                                               solve_type                     = NEWTON                                          splitting                      = INVALID                                         time                           = 0                                               update_xfem_at_timestep_begin  = 0                                               verbose                        = 0                                             []                                                                                                                                                                [Functions]                                                                                                                                                         [./solution]                                                                       inactive                     = (no_default)                                      isObjectAction               = 1                                                 type                         = ParsedFunction                                    control_tags                 = Functions                                         enable                       = 1                                                 vals                         = INVALID                                           value                        = exp(2*x)                                          vars                         = INVALID                                         [../]                                                                          []                                                                                                                                                                [Kernels]                                                                                                                                                           [./diff]                                                                           inactive                     = (no_default)                                      isObjectAction               = 1                                                 type                         = Diffusion                                         block                        = INVALID                                           control_tags                 = Kernels                                           diag_save_in                 = INVALID                                           displacements                = INVALID                                           enable                       = 1                                                 extra_matrix_tags            = INVALID                                           extra_vector_tags            = INVALID                                           implicit                     = 1                                                 matrix_tags                  = system                                            save_in                      = INVALID                                           seed                         = 0                                                 use_displaced_mesh           = 0                                                 variable                     = u                                                 vector_tags                  = nontime                                         [../]                                                                                                                                                             [./react]                                                                          inactive                     = (no_default)                                      isObjectAction               = 1                                                 type                         = MaterialPropertyUserObjectReaction                block                        = INVALID                                           control_tags                 = Kernels                                           diag_save_in                 = INVALID                                           displacements                = INVALID                                           enable                       = 1                                                 extra_matrix_tags            = INVALID                                           extra_vector_tags            = INVALID                                           implicit                     = 1                                                 matrix_tags                  = system                                            save_in                      = INVALID                                           seed                         = 0                                                 uo_name                      = c                                                 use_displaced_mesh           = 0                                                 variable                     = u                                                 vector_tags                  = nontime                                         [../]                                                                          []                                                                                                                                                                [Materials]                                                                                                                                                         [./c]                                                                              inactive                     = (no_default)                                      isObjectAction               = 1                                                 type                         = GenericConstantMaterial                           block                        = INVALID                                           boundary                     = INVALID                                           compute                      = 1                                                 constant_on                  = NONE                                              control_tags                 = Materials                                         enable                       = 1                                                 implicit                     = 1                                                 output_properties            = INVALID                                           outputs                      = none                                              prop_names                   = c                                                 prop_values                  = 4                                                 seed                         = 0                                                 use_displaced_mesh           = 0                                               [../]                                                                          []                                                                                                                                                                [Mesh]                                                                             displacements                  = INVALID                                         inactive                       = (no_default)                                    use_displaced_mesh             = 1                                               include_local_in_ghosting      = 0                                               output_ghosting                = 0                                               block_id                       = INVALID                                         block_name                     = INVALID                                         boundary_id                    = INVALID                                         boundary_name                  = INVALID                                         construct_side_list_from_node_list = 0                                           ghosted_boundaries             = INVALID                                         ghosted_boundaries_inflation   = INVALID                                         isObjectAction                 = 1                                               second_order                   = 0                                               skip_partitioning              = 0                                               type                           = GeneratedMesh                                   uniform_refine                 = 0                                               allow_renumbering              = 1                                               bias_x                         = 1                                               bias_y                         = 1                                               bias_z                         = 1                                               centroid_partitioner_direction = INVALID                                         construct_node_list_from_side_list = 1                                           control_tags                   = (no_default)                                    dim                            = 2                                               elem_type                      = QUAD4                                           enable                         = 1                                               gauss_lobatto_grid             = 0                                               ghosting_patch_size            = INVALID                                         max_leaf_size                  = 10                                              nemesis                        = 0                                               nx                             = 1                                               ny                             = 1                                               nz                             = 1                                               parallel_type                  = DEFAULT                                         partitioner                    = default                                         patch_size                     = 40                                              patch_update_strategy          = never                                           xmax                           = 1                                               xmin                           = 0                                               ymax                           = 1                                               ymin                           = 0                                               zmax                           = 1                                               zmin                           = 0                                             []                                                                                                                                                                [Mesh]                                                                           []                                                                                                                                                                [Mesh]                                                                           []                                                                                                                                                                [Outputs]                                                                          append_date                    = 0                                               append_date_format             = INVALID                                         checkpoint                     = 0                                               color                          = 1                                               console                        = 1                                               controls                       = 0                                               csv                            = 0                                               dofmap                         = 0                                               execute_on                     = 'INITIAL TIMESTEP_END'                          exodus                         = 1                                               file_base                      = INVALID                                         gmv                            = 0                                               gnuplot                        = 0                                               hide                           = INVALID                                         inactive                       = (no_default)                                    interval                       = 1                                               nemesis                        = 0                                               output_if_base_contains        = INVALID                                         perf_graph                     = 0                                               print_linear_residuals         = 1                                               print_mesh_changed_info        = 0                                               print_perf_log                 = 0                                               show                           = INVALID                                         solution_history               = 0                                               sync_times                     = (no_default)                                    tecplot                        = 0                                               vtk                            = 0                                               xda                            = 0                                               xdr                            = 0                                             []                                                                                                                                                                [Postprocessors]                                                                                                                                                    [./l2_error]                                                                       inactive                     = (no_default)                                      isObjectAction               = 1                                                 type                         = ElementL2Error                                    allow_duplicate_execution_on_initial = 0                                         block                        = INVALID                                           control_tags                 = Postprocessors                                    enable                       = 1                                                 execute_on                   = TIMESTEP_END                                      function                     = solution                                          implicit                     = 1                                                 outputs                      = INVALID                                           seed                         = 0                                                 use_displaced_mesh           = 0                                                 variable                     = u                                               [../]                                                                          []                                                                                                                                                                [Preconditioning]                                                                                                                                                   [./smp]                                                                            inactive                     = (no_default)                                      isObjectAction               = 1                                                 type                         = SMP                                               control_tags                 = Preconditioning                                   coupled_groups               = INVALID                                           enable                       = 1                                                 full                         = 1                                                 ksp_norm                     = unpreconditioned                                  mffd_type                    = wp                                                off_diag_column              = INVALID                                           off_diag_row                 = INVALID                                           pc_side                      = default                                           petsc_options                = INVALID                                           petsc_options_iname          = INVALID                                           petsc_options_value          = INVALID                                           solve_type                   = INVALID                                         [../]                                                                          []                                                                                                                                                                [UserObjects]                                                                                                                                                       [./c]                                                                              inactive                     = (no_default)                                      isObjectAction               = 1                                                 type                         = MaterialPropertyUserObject                        allow_duplicate_execution_on_initial = 0                                         block                        = INVALID                                           control_tags                 = UserObjects                                       enable                       = 1                                                 execute_on                   = 'INITIAL LINEAR NONLINEAR'                        implicit                     = 1                                                 mat_prop                     = c                                                 seed                         = 0                                                 use_displaced_mesh           = 0                                               [../]                                                                          []                                                                                                                                                                [Variables]                                                                                                                                                         [./u]                                                                              family                       = LAGRANGE                                          inactive                     = (no_default)                                      isObjectAction               = 1                                                 order                        = FIRST                                             scaling                      = INVALID                                           type                         = MooseVariableBase                                 initial_from_file_timestep   = LATEST                                            initial_from_file_var        = INVALID                                           block                        = INVALID                                           components                   = 1                                                 control_tags                 = Variables                                         eigen                        = 0                                                 enable                       = 1                                                 initial_condition            = INVALID                                           outputs                      = INVALID                                         [../]                                                                          []                                                                                                                                  ?�      ?�      @�d��ݮ@�d��ݮ?�      ?�p����t