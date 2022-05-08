// Copyright (C) 2018-2021 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Constant.h"
#include "DirichletBC.h"
#include "DofMap.h"
#include "Form.h"
#include "FunctionSpace.h"
#include "utils.h"
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/utils.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <functional>
#include <memory>
#include <vector>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xtensor.hpp>

namespace dolfinx::fem::impl
{

/// Implementation of vector assembly

/// Implementation of bc application
/// @tparam T The scalar type
/// @tparam _bs0 The block size of the form test function dof map. If
/// less than zero the block size is determined at runtime. If `_bs0` is
/// positive the block size is used as a compile-time constant, which
/// has performance benefits.
/// @tparam _bs1 The block size of the trial function dof map.
template <typename T, int _bs0 = -1, int _bs1 = -1>
void _lift_bc_cells(
    xtl::span<T> b, const mesh::Geometry& geometry,
    const std::function<void(T*, const T*, const T*, const double*, const int*,
                             const std::uint8_t*)>& kernel,
    const xtl::span<const std::int32_t>& cells,
    const std::function<void(const xtl::span<T>&,
                             const xtl::span<const std::uint32_t>&,
                             std::int32_t, int)>& dof_transform,
    const graph::AdjacencyList<std::int32_t>& dofmap0, int bs0,
    const std::function<void(const xtl::span<T>&,
                             const xtl::span<const std::uint32_t>&,
                             std::int32_t, int)>& dof_transform_to_transpose,
    const graph::AdjacencyList<std::int32_t>& dofmap1, int bs1,
    const xtl::span<const T>& constants, const xtl::span<const T>& coeffs,
    int cstride, const xtl::span<const std::uint32_t>& cell_info,
    const xtl::span<const T>& bc_values1,
    const xtl::span<const std::int8_t>& bc_markers1,
    const xtl::span<const T>& x0, double scale)
{ }

/// @tparam T The scalar type
/// @tparam _bs0 The block size of the form test function dof map. If
/// less than zero the block size is determined at runtime. If `_bs0` is
/// positive the block size is used as a compile-time constant, which
/// has performance benefits.
/// @tparam _bs1 The block size of the trial function dof map.
template <typename T, int _bs = -1>
void _lift_bc_exterior_facets(
    xtl::span<T> b, const mesh::Mesh& mesh,
    const std::function<void(T*, const T*, const T*, const double*, const int*,
                             const std::uint8_t*)>& kernel,
    const xtl::span<const std::pair<std::int32_t, int>>& facets,
    const std::function<void(const xtl::span<T>&,
                             const xtl::span<const std::uint32_t>&,
                             std::int32_t, int)>& dof_transform,
    const graph::AdjacencyList<std::int32_t>& dofmap0, int bs0,
    const std::function<void(const xtl::span<T>&,
                             const xtl::span<const std::uint32_t>&,
                             std::int32_t, int)>& dof_transform_to_transpose,
    const graph::AdjacencyList<std::int32_t>& dofmap1, int bs1,
    const xtl::span<const T>& constants, const xtl::span<const T>& coeffs,
    int cstride, const xtl::span<const std::uint32_t>& cell_info,
    const xtl::span<const T>& bc_values1,
    const xtl::span<const std::int8_t>& bc_markers1,
    const xtl::span<const T>& x0, double scale)
{ }

/// @tparam T The scalar type
/// @tparam _bs0 The block size of the form test function dof map. If
/// less than zero the block size is determined at runtime. If `_bs0` is
/// positive the block size is used as a compile-time constant, which
/// has performance benefits.
/// @tparam _bs1 The block size of the trial function dof map.
template <typename T, int _bs = -1>
void _lift_bc_interior_facets(
    xtl::span<T> b, const mesh::Mesh& mesh,
    const std::function<void(T*, const T*, const T*, const double*, const int*,
                             const std::uint8_t*)>& kernel,
    const xtl::span<const std::tuple<std::int32_t, int, std::int32_t, int>>&
        facets,
    const std::function<void(const xtl::span<T>&,
                             const xtl::span<const std::uint32_t>&,
                             std::int32_t, int)>& dof_transform,
    const graph::AdjacencyList<std::int32_t>& dofmap0, int bs0,
    const std::function<void(const xtl::span<T>&,
                             const xtl::span<const std::uint32_t>&,
                             std::int32_t, int)>& dof_transform_to_transpose,
    const graph::AdjacencyList<std::int32_t>& dofmap1, int bs1,
    const xtl::span<const T>& constants, const xtl::span<const T>& coeffs,
    int cstride, const xtl::span<const std::uint32_t>& cell_info,
    const std::function<std::uint8_t(std::size_t)>& get_perm,
    const xtl::span<const T>& bc_values1,
    const xtl::span<const std::int8_t>& bc_markers1,
    const xtl::span<const T>& x0, double scale)
{ }
/// Execute kernel over cells and accumulate result in vector
/// @tparam T The scalar type
/// @tparam _bs The block size of the form test function dof map. If
/// less than zero the block size is determined at runtime. If `_bs` is
/// positive the block size is used as a compile-time constant, which
/// has performance benefits.
template <typename T, int _bs = -1>
void assemble_cells(
    const std::function<void(const xtl::span<T>&,
                             const xtl::span<const std::uint32_t>&,
                             std::int32_t, int)>& dof_transform,
    xtl::span<T> b, const mesh::Geometry& geometry,
    const xtl::span<const std::int32_t>& cells,
    const graph::AdjacencyList<std::int32_t>& dofmap, int bs,
    const std::function<void(T*, const T*, const T*, const double*, const int*,
                             const std::uint8_t*)>& kernel,
    const xtl::span<const T>& constants, const xtl::span<const T>& coeffs,
    int cstride, const xtl::span<const std::uint32_t>& cell_info)
{
  assert(_bs < 0 or _bs == bs);

  if (cells.empty())
    return;

  // Prepare cell geometry
  const graph::AdjacencyList<std::int32_t>& x_dofmap = geometry.dofmap();
  const std::size_t num_dofs_g = geometry.cmap().dim();
  xtl::span<const double> x_g = geometry.x();

  // FIXME: Add proper interface for num_dofs
  // Create data structures used in assembly
  const int num_dofs = dofmap.links(0).size();
  std::vector<double> coordinate_dofs(3 * num_dofs_g);
  std::vector<T> be(bs * num_dofs);
  const xtl::span<T> _be(be);

  // Iterate over active cells
  for (std::size_t index = 0; index < cells.size(); ++index)
  {
    std::int32_t c = cells[index];

    // Get cell coordinates/geometry
    auto x_dofs = x_dofmap.links(c);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
    {
      common::impl::copy_N<3>(std::next(x_g.begin(), 3 * x_dofs[i]),
                              std::next(coordinate_dofs.begin(), 3 * i));
    }

    // Tabulate vector for cell
    std::fill(be.begin(), be.end(), 0);
    kernel(be.data(), coeffs.data() + index * cstride, constants.data(),
           coordinate_dofs.data(), nullptr, nullptr);
    // dof_transform(_be, cell_info, c, 1);

    // Scatter cell vector to 'global' vector array
    auto dofs = dofmap.links(c);
    if constexpr (_bs > 0)
    {
      for (int i = 0; i < num_dofs; ++i)
        for (int k = 0; k < _bs; ++k)
          b[_bs * dofs[i] + k] += be[_bs * i + k];
    }
    else
    {
      for (int i = 0; i < num_dofs; ++i)
        for (int k = 0; k < bs; ++k)
          b[bs * dofs[i] + k] += be[bs * i + k];
    }
  }
}

/// Execute kernel over cells and accumulate result in vector
/// @tparam T The scalar type
/// @tparam _bs The block size of the form test function dof map. If
/// less than zero the block size is determined at runtime. If `_bs` is
/// positive the block size is used as a compile-time constant, which
/// has performance benefits.
template <typename T, int _bs = -1>
void assemble_exterior_facets(
    const std::function<void(const xtl::span<T>&,
                             const xtl::span<const std::uint32_t>&,
                             std::int32_t, int)>& dof_transform,
    xtl::span<T> b, const mesh::Mesh& mesh,
    const xtl::span<const std::pair<std::int32_t, int>>& facets,
    const graph::AdjacencyList<std::int32_t>& dofmap, int bs,
    const std::function<void(T*, const T*, const T*, const double*, const int*,
                             const std::uint8_t*)>& fn,
    const xtl::span<const T>& constants, const xtl::span<const T>& coeffs,
    int cstride, const xtl::span<const std::uint32_t>& cell_info)
{ }

/// Assemble linear form interior facet integrals into an vector
/// @tparam T The scalar type
/// @tparam _bs The block size of the form test function dof map. If
/// less than zero the block size is determined at runtime. If `_bs` is
/// positive the block size is used as a compile-time constant, which
/// has performance benefits.
template <typename T, int _bs = -1>
void assemble_interior_facets(
    const std::function<void(const xtl::span<T>&,
                             const xtl::span<const std::uint32_t>&,
                             std::int32_t, int)>& dof_transform,
    xtl::span<T> b, const mesh::Mesh& mesh,
    const xtl::span<const std::tuple<std::int32_t, int, std::int32_t, int>>&
        facets,
    const fem::DofMap& dofmap,
    const std::function<void(T*, const T*, const T*, const double*, const int*,
                             const std::uint8_t*)>& fn,
    const xtl::span<const T>& constants, const xtl::span<const T>& coeffs,
    int cstride, const xtl::span<const std::uint32_t>& cell_info,
    const std::function<std::uint8_t(std::size_t)>& get_perm)
{ }

/// Modify RHS vector to account for boundary condition such that:
///
/// b <- b - scale * A (x_bc - x0)
///
/// @param[in,out] b The vector to be modified
/// @param[in] a The bilinear form that generates A
/// @param[in] constants Constants that appear in `a`
/// @param[in] coefficients Coefficients that appear in `a`
/// @param[in] bc_values1 The boundary condition 'values'
/// @param[in] bc_markers1 The indices (columns of A, rows of x) to
/// which bcs belong
/// @param[in] x0 The array used in the lifting, typically a 'current
/// solution' in a Newton method
/// @param[in] scale Scaling to apply
template <typename T>
void lift_bc(xtl::span<T> b, const Form<T>& a,
             const xtl::span<const T>& constants,
             const std::map<std::pair<IntegralType, int>,
                            std::pair<xtl::span<const T>, int>>& coefficients,
             const xtl::span<const T>& bc_values1,
             const xtl::span<const std::int8_t>& bc_markers1,
             const xtl::span<const T>& x0, double scale)
{ }

/// Modify b such that:
///
///   b <- b - scale * A_j (g_j - x0_j)
///
/// where j is a block (nest) row index. For a non-blocked problem j = 0.
/// The boundary conditions bc1 are on the trial spaces V_j. The forms
/// in [a] must have the same test space as L (from which b was built),
/// but the trial space may differ. If x0 is not supplied, then it is
/// treated as zero.
/// @param[in,out] b The vector to be modified
/// @param[in] a The bilinear forms, where a[j] is the form that
/// generates A_j
/// @param[in] constants Constants that appear in `a`
/// @param[in] coeffs Coefficients that appear in `a`
/// @param[in] bcs1 List of boundary conditions for each block, i.e.
/// bcs1[2] are the boundary conditions applied to the columns of a[2] /
/// x0[2] block
/// @param[in] x0 The vectors used in the lifting
/// @param[in] scale Scaling to apply
template <typename T>
void apply_lifting(
    xtl::span<T> b, const std::vector<std::shared_ptr<const Form<T>>> a,
    const std::vector<xtl::span<const T>>& constants,
    const std::vector<std::map<std::pair<IntegralType, int>,
                               std::pair<xtl::span<const T>, int>>>& coeffs,
    const std::vector<std::vector<std::shared_ptr<const DirichletBC<T>>>>& bcs1,
    const std::vector<xtl::span<const T>>& x0, double scale)
{ }

/// Assemble linear form into a vector
/// @param[in,out] b The vector to be assembled. It will not be zeroed
/// before assembly.
/// @param[in] L The linear forms to assemble into b
/// @param[in] constants Packed constants that appear in `L`
/// @param[in] coefficients Packed coefficients that appear in `L`
template <typename T>
void assemble_vector(
    xtl::span<T> b, const Form<T>& L, const xtl::span<const T>& constants,
    const std::map<std::pair<IntegralType, int>,
                   std::pair<xtl::span<const T>, int>>& coefficients)
{
  std::shared_ptr<const mesh::Mesh>         mesh = L.mesh();
  std::shared_ptr<const fem::FiniteElement> element = L.function_spaces().at(0)->element();
  const graph::AdjacencyList<std::int32_t>& dofs = L.function_spaces().at(0)->dofmap()->list();
  const int bs = L.function_spaces().at(0)->dofmap().bs();

  const std::function<void(const xtl::span<T>&,
                           const xtl::span<const std::uint32_t>&, std::int32_t,
                           int)>
      dof_transform;

  // const bool needs_transformation_data
  //     = element->needs_dof_transformations() or L.needs_facet_permutations();
  xtl::span<const std::uint32_t> cell_info;
  // if (needs_transformation_data)
  // {
  //   mesh->topology_mutable().create_entity_permutations();
  //   cell_info = xtl::span(mesh->topology().get_cell_permutation_info());
  // }

  for (int i : L.integral_ids(IntegralType::cell))
  {
    const auto& fn = L.kernel(IntegralType::cell, i);
    const auto& [coeffs, cstride] = coefficients.at({IntegralType::cell, i});
    const std::vector<std::int32_t>& cells = L.cell_domains(i);
    if (bs == 1)
    {
      impl::assemble_cells<T, 1>(dof_transform, b, mesh->geometry(), cells,
                                 dofs, bs, fn, constants, coeffs, cstride,
                                 cell_info);
    }
    // else if (bs == 3)
    // {
    //   impl::assemble_cells<T, 3>(dof_transform, b, mesh->geometry(), cells,
    //                              dofs, bs, fn, constants, coeffs, cstride,
    //                              cell_info);
    // }
    // else
    // {
    //   impl::assemble_cells(dof_transform, b, mesh->geometry(), cells, dofs, bs,
    //                        fn, constants, coeffs, cstride, cell_info);
    // }
  }
}
} // namespace dolfinx::fem::impl
