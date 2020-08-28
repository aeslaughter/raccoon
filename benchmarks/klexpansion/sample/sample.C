//* This file is part of the RACCOON application
//* being developed at Dolbow lab at Duke University
//* http://dolbow.pratt.duke.edu

// libMesh include files.
#include "libmesh/libmesh.h"
#include "libmesh/mesh.h"
#include "libmesh/exodusII_io.h"
#include "libmesh/exodusII_io_helper.h"

#include <boost/math/distributions.hpp>
#include <ctime>
#include <random>

// Bring in everything from the libMesh namespace
using namespace libMesh;

std::vector<Real> sample_gaussian(const std::vector<Real> & eigvals,
                                  const std::vector<std::vector<Real>> & eigvecs,
                                  std::default_random_engine & generator);

void gaussian_to_gamma(const std::vector<Real> & Xi, std::vector<Real> & P, Real mean, Real cv);

int
main(int argc, char ** argv)
{
  // Check for proper usage.
  if (argc % 3 != 1)
    libmesh_error_msg("\nUsage: " << argv[0] << " field_name field_mean field_cv ...");

  // Initialize libMesh and the dependent libraries.
  LibMeshInit init(argc, argv);

  if (init.comm().size() > 1)
    libmesh_error_msg("Parallel sampling is not supported, use 1 processor only.");

  // Read the mesh
  Mesh mesh(init.comm());
  mesh.read("basis.e");

  // Read the eigenpairs
  std::vector<Real> eigvals;
  std::vector<std::vector<Real>> eigvecs;

  ExodusII_IO basis(mesh);
  basis.read("basis.e");
  ExodusII_IO_Helper & basis_helper = basis.get_exio_helper();
  for (int i = 1; i < basis.get_num_time_steps(); i++)
  {
    // read eigenvalue
    std::vector<Real> eigval;
    basis.read_global_variable({"d"}, i, eigval);
    eigvals.push_back(eigval[0]);

    // read eigenvector
    basis_helper.read_nodal_var_values("v", i);
    eigvecs.push_back(basis_helper.nodal_var_values);
  }

  // random engine
  std::default_random_engine generator;
  generator.seed(std::time(NULL));

  // number of fields to sample
  int num_fields = (argc - 1) / 3;
  std::vector<std::vector<Real>> fields(num_fields);
  std::vector<std::string> field_names(num_fields);

  // sample each field
  for (int i = 0; i < num_fields; i++)
  {
    // read in arguments
    field_names[i] = argv[3 * i + 1];
    Real mean = strtod(argv[3 * i + 2], nullptr);
    Real cv = strtod(argv[3 * i + 3], nullptr);
    libMesh::out << "Sampling field " << field_names[i] << ", mean = " << mean << ", CV = " << cv
                 << std::endl;

    // sample Gaussian fields
    std::vector<Real> Xi = sample_gaussian(eigvals, eigvecs, generator);

    // transform to marginal Gamma fields
    gaussian_to_gamma(Xi, fields[i], mean, cv);
  }

  // write random field
  ExodusII_IO mesh_with_random_fields(mesh);
  mesh_with_random_fields.write("fields.e");
  ExodusII_IO_Helper & fields_helper = mesh_with_random_fields.get_exio_helper();
  fields_helper.initialize_nodal_variables(field_names);
  for (int i = 0; i < num_fields; i++)
    fields_helper.write_nodal_values(i + 1, fields[i], 1);

  return EXIT_SUCCESS;
}

std::vector<Real>
sample_gaussian(const std::vector<Real> & eigvals,
                const std::vector<std::vector<Real>> & eigvecs,
                std::default_random_engine & generator)
{
  unsigned int ndof = eigvecs[0].size();
  std::vector<Real> Xi(ndof);
  std::normal_distribution<Real> distribution(0.0, 1.0);

  for (unsigned int i = 0; i < eigvals.size(); i++)
  {
    Real eta = distribution(generator);
    for (unsigned int j = 0; j < ndof; j++)
      Xi[j] += std::sqrt(eigvals[i]) * eta * eigvecs[i][j];
  }

  return Xi;
}

void
gaussian_to_gamma(const std::vector<Real> & Xi, std::vector<Real> & P, Real mean, Real cv)
{
  unsigned int ndof = Xi.size();

  // Normal distribution
  auto normal = boost::math::normal_distribution<Real>(0, 1);

  // Gamma distribution
  Real std = cv * mean;
  Real var = std * std;
  Real theta = var / mean;
  Real k = mean / theta;

  // Transform into the field
  P.resize(ndof);
  for (unsigned int i = 0; i < ndof; i++)
    P[i] = theta * boost::math::gamma_p_inv<Real, Real>(k, boost::math::cdf(normal, Xi[i]));
}
