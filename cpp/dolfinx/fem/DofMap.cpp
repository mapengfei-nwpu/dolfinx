// Copyright (C) 2007-2018 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "DofMap.h"
#include "ElementDofLayout.h"
#include "dofmapbuilder.h"
#include "utils.h"
#include <cstdint>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/sort.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/mesh/Topology.h>
#include <memory>
#include <utility>

using namespace dolfinx;
using namespace dolfinx::fem;

namespace
{
//-----------------------------------------------------------------------------
template <typename T>
std::vector<std::int32_t>
remap_dofs(const std::vector<std::int32_t>& old_to_new,
           const graph::AdjacencyList<T>& dofs_old)
{
  const std::vector<T>& _dofs_old = dofs_old.array();
  std::vector<std::int32_t> dofmap(_dofs_old.size());
  for (std::size_t i = 0; i < dofmap.size(); ++i)
    dofmap[i] = old_to_new[_dofs_old[i]];
  return dofmap;
}
//-----------------------------------------------------------------------------
// Build a collapsed DofMap from a dofmap view. Extracts dofs and
// doesn't build a new re-ordered dofmap
fem::DofMap build_collapsed_dofmap(MPI_Comm comm, const DofMap& dofmap_view,
                                   const mesh::Topology& topology)
{
  if (dofmap_view.element_dof_layout().block_size() > 1)
  {
    throw std::runtime_error(
        "Cannot collapse dofmap with block size greater "
        "than 1 from parent with block size of 1. Create new dofmap first.");
  }

  // Get topological dimension
  const int tdim = topology.dim();
  auto cells = topology.connectivity(tdim, 0);
  assert(cells);

  // Build set of dofs that are in the new dofmap
  std::vector<std::int32_t> dofs_view = dofmap_view.list().array();
  dolfinx::radix_sort(xtl::span(dofs_view));
  dofs_view.erase(std::unique(dofs_view.begin(), dofs_view.end()),
                  dofs_view.end());

  // Get block sizes
  const int bs_view = dofmap_view.index_map_bs();

  // Compute sizes
  const std::int32_t num_owned_view = dofmap_view.index_map->size_local();
  const auto it_unowned0 = std::lower_bound(dofs_view.begin(), dofs_view.end(),
                                            num_owned_view * bs_view);
  const std::size_t num_owned = std::distance(dofs_view.begin(), it_unowned0);
  const std::size_t num_unowned = std::distance(it_unowned0, dofs_view.end());

  // FIXME: We can avoid the MPI_Exscan by counting the offsets for the
  // owned mesh entities

  // Get process offset for new dofmap
  std::size_t offset = 0;
  MPI_Exscan(&num_owned, &offset, 1, dolfinx::MPI::mpi_type<std::size_t>(),
             MPI_SUM, comm);

  // For owned dofs, compute new global index
  std::vector<std::int64_t> global_index(dofmap_view.index_map->size_local(),
                                         -1);
  for (auto it = dofs_view.begin(); it != it_unowned0; ++it)
  {
    const std::size_t block = std::distance(dofs_view.begin(), it);
    const std::int32_t block_parent = *it / bs_view;
    global_index[block_parent] = block + offset;
  }

  // Send new global indices for owned dofs to non-owning process, and
  // receive new global indices from owner
  std::vector<std::int64_t> global_index_remote(
      dofmap_view.index_map->num_ghosts());
  dofmap_view.index_map->scatter_fwd(
      xtl::span<const std::int64_t>(global_index),
      xtl::span<std::int64_t>(global_index_remote), 1);

  // Get owning ranks (neighbour) for each ghost and map from
  // neighbourhood rank to global rank
  const std::vector<int> ghost_owner_old
      = dofmap_view.index_map->ghost_owners();
  const std::vector<int> neighbor_ranks = dolfinx::MPI::neighbors(
      dofmap_view.index_map->comm(common::IndexMap::Direction::forward))[0];

  // Compute ghosts for collapsed dofmap
  std::vector<std::int64_t> ghosts(num_unowned);
  std::vector<int> ghost_owners(num_unowned);
  for (auto it = it_unowned0; it != dofs_view.end(); ++it)
  {
    const std::int32_t index = std::distance(it_unowned0, it);
    const std::int32_t index_old = *it / bs_view - num_owned_view;
    assert(global_index_remote[index_old] >= 0);
    ghosts[index] = global_index_remote[index_old];
    ghost_owners[index] = neighbor_ranks[ghost_owner_old[index_old]];
  }

  // Create new index map
  std::vector<int> src_ranks = ghost_owners;
  std::sort(src_ranks.begin(), src_ranks.end());
  src_ranks.erase(std::unique(src_ranks.begin(), src_ranks.end()),
                  src_ranks.end());
  auto dest_ranks = dolfinx::MPI::compute_graph_edges_nbx(comm, src_ranks);
  auto index_map = std::make_shared<common::IndexMap>(
      comm, num_owned, dest_ranks, ghosts, ghost_owners);

  // Create array from dofs in view to new dof indices
  std::vector<std::int32_t> old_to_new(dofs_view.back() + 1, -1);
  std::int32_t count = 0;
  for (auto& dof : dofs_view)
    old_to_new[dof] = count++;

  // Build new dofmap
  const graph::AdjacencyList<std::int32_t>& dof_array_view = dofmap_view.list();
  std::vector<std::int32_t> dofmap = remap_dofs(old_to_new, dof_array_view);

  // Dimension sanity checks
  assert((int)dofmap.size()
         == (cells->num_nodes() * dofmap_view.element_dof_layout().num_dofs()));

  const int cell_dimension = dofmap_view.element_dof_layout().num_dofs();
  assert(dofmap.size() % cell_dimension == 0);

  // Copy dof layout, discarding parent data
  ElementDofLayout element_dof_layout = dofmap_view.element_dof_layout().copy();

  // Create new dofmap and return
  return fem::DofMap(
      std::move(element_dof_layout), index_map, 1,
      graph::regular_adjacency_list(std::move(dofmap), cell_dimension), 1);
}

} // namespace

//-----------------------------------------------------------------------------
graph::AdjacencyList<std::int32_t>
fem::transpose_dofmap(const graph::AdjacencyList<std::int32_t>& dofmap,
                      std::int32_t num_cells)
{
  // Count number of cell contributions to each global index
  const std::int32_t max_index = *std::max_element(
      dofmap.array().begin(),
      std::next(dofmap.array().begin(), dofmap.offsets()[num_cells]));

  std::vector<int> num_local_contributions(max_index + 1, 0);
  for (int c = 0; c < num_cells; ++c)
  {
    for (auto dof : dofmap.links(c))
      num_local_contributions[dof]++;
  }

  // Compute offset for each global index
  std::vector<int> index_offsets(num_local_contributions.size() + 1, 0);
  std::partial_sum(num_local_contributions.begin(),
                   num_local_contributions.end(), index_offsets.begin() + 1);

  std::vector<std::int32_t> data(index_offsets.back());
  std::vector<int> pos = index_offsets;
  int cell_offset = 0;
  for (int c = 0; c < num_cells; ++c)
  {
    for (auto dof : dofmap.links(c))
      data[pos[dof]++] = cell_offset++;
  }

  // Sort the source indices for each global index
  // This could improve linear memory access
  // FIXME: needs profiling
  for (int index = 0; index < max_index; ++index)
  {
    std::sort(data.begin() + index_offsets[index],
              data.begin() + index_offsets[index + 1]);
  }

  return graph::AdjacencyList<std::int32_t>(std::move(data),
                                            std::move(index_offsets));
}
//-----------------------------------------------------------------------------
/// Equality operator
bool DofMap::operator==(const DofMap& map) const
{
  return this->_index_map_bs == map._index_map_bs
         and this->_dofmap == map._dofmap and this->_bs == map._bs;
}
//-----------------------------------------------------------------------------
int DofMap::bs() const noexcept { return _bs; }
//-----------------------------------------------------------------------------
DofMap DofMap::extract_sub_dofmap(const std::vector<int>& component) const
{
  assert(!component.empty());

  // Get components in parent map that correspond to sub-dofs
  const std::vector sub_element_map_view
      = this->element_dof_layout().sub_view(component);

  // Build dofmap by extracting from parent
  const int num_cells = this->_dofmap.num_nodes();
  // FIXME X: how does sub_element_map_view hand block sizes?
  const std::int32_t dofs_per_cell = sub_element_map_view.size();
  std::vector<std::int32_t> dofmap(num_cells * dofs_per_cell);
  const int bs_parent = this->bs();
  for (int c = 0; c < num_cells; ++c)
  {
    auto cell_dmap_parent = this->_dofmap.links(c);
    for (std::int32_t i = 0; i < dofs_per_cell; ++i)
    {
      const std::div_t pos = std::div(sub_element_map_view[i], bs_parent);
      dofmap[c * dofs_per_cell + i]
          = bs_parent * cell_dmap_parent[pos.quot] + pos.rem;
    }
  }

  // FIXME X

  // Set element dof layout and cell dimension
  ElementDofLayout sub_element_dof_layout
      = _element_dof_layout.sub_layout(component);
  return DofMap(
      std::move(sub_element_dof_layout), this->index_map, this->index_map_bs(),
      graph::regular_adjacency_list(std::move(dofmap), dofs_per_cell), 1);
}
//-----------------------------------------------------------------------------
std::pair<DofMap, std::vector<std::int32_t>> DofMap::collapse(
    MPI_Comm comm, const mesh::Topology& topology,
    const std::function<std::vector<int>(
        const graph::AdjacencyList<std::int32_t>&)>& reorder_fn) const
{
  // Create new dofmap
  auto create_submap = [](MPI_Comm comm, auto index_map_bs, auto& layout,
                          auto& topology, auto& reorder_fn, auto& dmap)
  {
    if (index_map_bs == 1 and layout.block_size() > 1)
    {
      // Parent does not have block structure but sub-map does, so build
      // new submap to get block structure for collapsed dofmap.

      // Create new element dof layout and reset parent
      ElementDofLayout collapsed_dof_layout = layout.copy();

      auto [_index_map, bs, dofmap] = fem::build_dofmap_data(
          comm, topology, collapsed_dof_layout, reorder_fn);
      auto index_map
          = std::make_shared<common::IndexMap>(std::move(_index_map));
      return DofMap(layout, index_map, bs, std::move(dofmap), bs);
    }
    else
    {
      // Collapse dof map, without build and re-ordering from scratch
      return build_collapsed_dofmap(comm, dmap, topology);
    }
  };

  DofMap dofmap_new = create_submap(comm, index_map_bs(), _element_dof_layout,
                                    topology, reorder_fn, *this);

  // Build map from collapsed dof index to original dof index
  auto index_map_new = dofmap_new.index_map;
  const std::int32_t size
      = (index_map_new->size_local() + index_map_new->num_ghosts())
        * dofmap_new.index_map_bs();
  std::vector<std::int32_t> collapsed_map(size);

  const int tdim = topology.dim();
  auto cells = topology.connectivity(tdim, 0);
  assert(cells);
  const int bs = dofmap_new.bs();
  for (int c = 0; c < cells->num_nodes(); ++c)
  {
    xtl::span<const std::int32_t> cell_dofs_view = this->cell_dofs(c);
    xtl::span<const std::int32_t> cell_dofs = dofmap_new.cell_dofs(c);
    for (std::size_t i = 0; i < cell_dofs.size(); ++i)
    {
      for (int k = 0; k < bs; ++k)
      {
        assert(bs * cell_dofs[i] + k < (int)collapsed_map.size());
        assert(bs * i + k < cell_dofs_view.size());
        collapsed_map[bs * cell_dofs[i] + k] = cell_dofs_view[bs * i + k];
      }
    }
  }

  return {std::move(dofmap_new), std::move(collapsed_map)};
}
//-----------------------------------------------------------------------------
const graph::AdjacencyList<std::int32_t>& DofMap::list() const
{
  return _dofmap;
}
//-----------------------------------------------------------------------------
int DofMap::index_map_bs() const { return _index_map_bs; }
//-----------------------------------------------------------------------------
