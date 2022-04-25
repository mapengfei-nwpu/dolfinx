// Copyright (C) 2015-2019 Chris Richardson, Garth N. Wells and Igor Baratta
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "IndexMapNew.h"
#include "IndexMap.h"
#include <algorithm>
#include <dolfinx/common/sort.h>
#include <functional>
#include <numeric>

using namespace dolfinx;
using namespace dolfinx::common;

//-----------------------------------------------------------------------------
common::IndexMap common::create_old(const IndexMapNew& map)
{
  return common::IndexMap(
      map.comm(), map.size_local(),
      dolfinx::MPI::compute_graph_edges_nbx(map.comm(), map.ghost_owners()),
      map.ghosts(), map.ghost_owners());
}
//-----------------------------------------------------------------------------
// std::vector<std::int32_t> common::halo_owned_indices(const IndexMapNew&)
// {
//   return std::vector<std::int32_t>();
// }

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
IndexMapNew::IndexMapNew(MPI_Comm comm, std::int32_t local_size)
    : IndexMapNew(comm, local_size, {}, {})
{
  // Do nothing
}
//-----------------------------------------------------------------------------
IndexMapNew::IndexMapNew(MPI_Comm comm, std::int32_t local_size,
                         const xtl::span<const std::int64_t>& ghosts,
                         const xtl::span<const int>& src_ranks)
    : _comm(comm), _ghosts(ghosts.begin(), ghosts.end()),
      _ghost_owners(src_ranks.begin(), src_ranks.end())
{
  assert(size_t(ghosts.size()) == src_ranks.size());

  // Get global offset (index), using partial exclusive reduction
  std::int64_t offset = 0;
  const std::int64_t local_size_tmp = (std::int64_t)local_size;
  MPI_Request request_scan;
  MPI_Iexscan(&local_size_tmp, &offset, 1, MPI_INT64_T, MPI_SUM, comm,
              &request_scan);

  // Send local size to sum reduction to get global size
  MPI_Request request;
  MPI_Iallreduce(&local_size_tmp, &_size_global, 1, MPI_INT64_T, MPI_SUM, comm,
                 &request);

  // Create communicators with directed edges: (0) owner -> ghost,

  // Wait for MPI_Iexscan to complete (get offset)
  MPI_Wait(&request_scan, MPI_STATUS_IGNORE);
  _local_range = {offset, offset + local_size};

  // Wait for the MPI_Iallreduce to complete
  MPI_Wait(&request, MPI_STATUS_IGNORE);
}
//-----------------------------------------------------------------------------
std::array<std::int64_t, 2> IndexMapNew::local_range() const noexcept
{
  return _local_range;
}
//-----------------------------------------------------------------------------
std::int32_t IndexMapNew::num_ghosts() const noexcept { return _ghosts.size(); }
//-----------------------------------------------------------------------------
std::int32_t IndexMapNew::size_local() const noexcept
{
  return _local_range[1] - _local_range[0];
}
//-----------------------------------------------------------------------------
std::int64_t IndexMapNew::size_global() const noexcept { return _size_global; }
//-----------------------------------------------------------------------------
const std::vector<std::int64_t>& IndexMapNew::ghosts() const noexcept
{
  return _ghosts;
}
//-----------------------------------------------------------------------------
void IndexMapNew::local_to_global(const xtl::span<const std::int32_t>& local,
                                  const xtl::span<std::int64_t>& global) const
{
  assert(local.size() <= global.size());
  const std::int32_t local_size = _local_range[1] - _local_range[0];
  std::transform(
      local.cbegin(), local.cend(), global.begin(),
      [local_size, local_range = _local_range[0], &ghosts = _ghosts](auto local)
      {
        if (local < local_size)
          return local_range + local;
        else
        {
          assert((local - local_size) < (int)ghosts.size());
          return ghosts[local - local_size];
        }
      });
}
//-----------------------------------------------------------------------------
void IndexMapNew::global_to_local(const xtl::span<const std::int64_t>& global,
                                  const xtl::span<std::int32_t>& local) const
{
  const std::int32_t local_size = _local_range[1] - _local_range[0];

  std::vector<std::pair<std::int64_t, std::int32_t>> global_local_ghosts(
      _ghosts.size());
  for (std::size_t i = 0; i < _ghosts.size(); ++i)
    global_local_ghosts[i] = {_ghosts[i], i + local_size};
  std::map<std::int64_t, std::int32_t> global_to_local(
      global_local_ghosts.begin(), global_local_ghosts.end());

  std::transform(global.cbegin(), global.cend(), local.begin(),
                 [range = _local_range,
                  &global_to_local](std::int64_t index) -> std::int32_t
                 {
                   if (index >= range[0] and index < range[1])
                     return index - range[0];
                   else
                   {
                     auto it = global_to_local.find(index);
                     return it != global_to_local.end() ? it->second : -1;
                   }
                 });
}
//-----------------------------------------------------------------------------
std::vector<std::int64_t> IndexMapNew::global_indices() const
{
  const std::int32_t local_size = _local_range[1] - _local_range[0];
  const std::int32_t num_ghosts = _ghosts.size();
  const std::int64_t global_offset = _local_range[0];
  std::vector<std::int64_t> global(local_size + num_ghosts);
  std::iota(global.begin(), std::next(global.begin(), local_size),
            global_offset);
  std::copy(_ghosts.cbegin(), _ghosts.cend(),
            std::next(global.begin(), local_size));
  return global;
}
//-----------------------------------------------------------------------------
const std::vector<int>& IndexMapNew::ghost_owners() const
{
  return _ghost_owners;
}
//----------------------------------------------------------------------------
MPI_Comm IndexMapNew::comm() const { return _comm.comm(); }
//----------------------------------------------------------------------------
