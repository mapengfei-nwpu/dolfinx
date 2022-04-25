// Copyright (C) 2022 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <basix/e-lagrange.h>
#include <basix/e-nedelec.h>
#include <cmath>
#include <dolfinx/common/Scatter.h>
#include <dolfinx/common/log.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/io/ADIOS2Writers.h>
#include <dolfinx/io/VTKFile.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/cell_types.h>
#include <dolfinx/mesh/generation.h>
#include <filesystem>
#include <mpi.h>

using namespace dolfinx;

/// This program shows how to create finite element spaces without FFCx
/// generated code
int main(int argc, char* argv[])
{
  dolfinx::init_logging(argc, argv);
  MPI_Init(&argc, &argv);

  // The main body of the function is scoped with the curly braces to
  // ensure that all objects that depend on an MPI communicator are
  // destroyed before MPI is finalised at the end of this function.
  {
    // Create a mesh. For what comes later in this demo we need to
    // ensure that a boundary between cells is located at x0=0.5
    auto mesh = std::make_shared<mesh::Mesh>(mesh::create_rectangle(
        MPI_COMM_WORLD, {{{0.0, 0.0}, {1.0, 1.0}}}, {32, 32},
        mesh::CellType::triangle, mesh::GhostMode::none));

    // Interpolate a function in a scalar Lagrange space and output the
    // result to file for visualisation
    // Create a Basix continuous Lagrange element of degree 1
    basix::FiniteElement e = basix::element::create_lagrange(
        mesh::cell_type_to_basix_type(mesh::CellType::triangle), 1,
        basix::element::lagrange_variant::equispaced, false);

    // Create a scalar function space
    auto V = std::make_shared<fem::FunctionSpace>(
        fem::create_functionspace(mesh, e, 2));

    // Create a finite element Function
    auto u = std::make_shared<fem::Function<double>>(V);

    int rank = dolfinx::MPI::rank(MPI_COMM_WORLD);

    auto vector = u->x()->mutable_array();
    auto map = V->dofmap()->index_map;
    int bs = V->dofmap()->index_map_bs();

    la::Vector<double> vec(map, bs);

    vec.set(rank);

    vec.scatter_fwd_begin();
    vec.scatter_fwd_end();
    std::int32_t n = map->size_local() * bs;
    xtl::span<const double> remote_data(vec.array().data() + n,
                                        map->num_ghosts() * bs);

  }

  MPI_Finalize();

  return 0;
}
