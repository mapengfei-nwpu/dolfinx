// Copyright (C) 2022 Igor A. Baratta
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <dolfinx/common/IndexMap.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <memory>

using namespace dolfinx;

namespace dolfinx::common
{
class VectorScatter
{
public:
  /// Mode for reverse scatter operation
  enum class Mode
  {
    insert,
    add
  };

  VectorScatter(const std::shared_ptr<const common::IndexMap> index_map, int bs)
      : _map(index_map), _bs(bs)
  {
    graph::AdjacencyList<std::int32_t> local_shared_ind
        = _map->scatter_fwd_indices();

    // Expand shared indices with block size
    const std::vector<int32_t>& inds = local_shared_ind.array();
    std::vector<std::int32_t> indices(inds.size() * _bs);
    for (std::size_t i = 0; i < inds.size(); i++)
      for (int j = 0; j < _bs; j++)
        indices[i * _bs + j] = inds[i] * _bs + j;

    std::vector<std::int32_t> offsets(local_shared_ind.offsets().size());
    std::transform(local_shared_ind.offsets().begin(),
                   local_shared_ind.offsets().end(), offsets.begin(),
                   [bs = _bs](auto& e) { return e * bs; });
    _sizes_send_fwd.reserve(offsets.size());
    std::adjacent_difference(std::next(offsets.begin()), offsets.end(),
                             std::back_inserter(_sizes_send_fwd));
    _shared_indices = std::make_unique<graph::AdjacencyList<std::int32_t>>(
        std::move(indices), std::move(offsets));

    // Get number of neighbors
    int indegree(-1), outdegree(-2), weighted(-1);
    MPI_Dist_graph_neighbors_count(_map->comm(IndexMap::Direction::forward),
                                   &indegree, &outdegree, &weighted);

    const std::vector<int32_t>& ghost_owners
        = _map->ghost_owner_neighbor_rank();

    // Create displacement vectors fwd scatter
    _sizes_recv_fwd.resize(indegree, 0);
    std::for_each(ghost_owners.cbegin(), ghost_owners.cend(),
                  [&recv = _sizes_recv_fwd, bs = _bs](auto owner)
                  { recv[owner] += bs; });

    _displs_recv_fwd.resize(indegree + 1, 0);
    std::partial_sum(_sizes_recv_fwd.begin(), _sizes_recv_fwd.end(),
                     _displs_recv_fwd.begin() + 1);

    // Build array that maps ghost indicies to a position in the recv
    // (forward scatter) and send (reverse scatter) buffers
    std::vector<std::int32_t> displs = _displs_recv_fwd;
    _ghost_pos_recv_fwd.resize(ghost_owners.size() * _bs);
    _ghost_pos_inv.resize(ghost_owners.size() * _bs);

    for (std::size_t i = 0; i < ghost_owners.size(); i++)
    {
      for (int j = 0; j < _bs; j++)
      {
        std::int32_t pos = displs[ghost_owners[i]]++;
        _ghost_pos_recv_fwd[i * _bs + j] = pos;
        _ghost_pos_inv[pos] = i * _bs + j;
      }
    }
  }

  /// Start a non-blocking send of owned data to ranks that ghost the
  /// data. The communication is completed by calling
  /// VectorScatter::fwd_end. The send and receive buffer should not
  /// be changed until after VectorScatter::fwd_end has been called.
  ///
  /// @param[in] send_buffer Local data associated with each owned local
  /// index to be sent to process where the data is ghosted. It must not
  /// be changed until after a call to VectorScatter::fwd_end. The
  /// order of data in the buffer is given by
  /// VectorScatter::scatter_fwd_indices.
  /// @param request The MPI request handle for tracking the status of
  /// the non-blocking communication
  /// @param recv_buffer A buffer used for the received data. The
  /// position of ghost entries in the buffer is given by
  /// VectorScatter::scatter_fwd_ghost_positions. The buffer must not be
  /// accessed or changed until after a call to
  /// VectorScatter::fwd_end.
  template <typename T>
  void scatter_fwd_begin(const xtl::span<const T>& send_buffer,
                         MPI_Request& request,
                         const xtl::span<T>& recv_buffer) const
  {

    // Send displacement
    const std::vector<int32_t>& displs_send_fwd = _shared_indices->offsets();
    assert(send_buffer.size() == std::size_t(displs_send_fwd.back()));

    MPI_Ineighbor_alltoallv(send_buffer.data(), _sizes_send_fwd.data(),
                            displs_send_fwd.data(), MPI::mpi_type<T>(),
                            recv_buffer.data(), _sizes_recv_fwd.data(),
                            _displs_recv_fwd.data(), MPI::mpi_type<T>(),
                            _map->comm(IndexMap::Direction::forward), &request);
  }

  /// Complete a non-blocking send from the local owner of to process
  /// ranks that have the index as a ghost. This function complete the
  /// communication started by VectorScatter::scatter_fwd_begin.
  ///
  /// @param[in] request The MPI request handle for tracking the status
  /// of the send
  void scatter_fwd_end(MPI_Request& request) const
  {
    // Wait for communication to complete
    MPI_Wait(&request, MPI_STATUS_IGNORE);
  }

  /// Send n values for each index that is owned to processes that have
  /// the index as a ghost. The size of the input array local_data must
  /// be the same as bs * size_local().
  ///
  /// @param[in] local_data Local data associated with each owned local
  /// index to be sent to process where the data is ghosted.
  /// @param[in,out] remote_data Ghost data on this process received
  /// from the owning process.
  template <typename T, typename Functor>
  void scatter_fwd(const xtl::span<const T>& local_data,
                   xtl::span<T> remote_data, Functor gather_fn) const
  {
    const std::vector<std::int32_t>& indices = _shared_indices->array();
    std::vector<T> send_buffer(indices.size());

    gather_fn(local_data, indices, send_buffer);

    MPI_Request request;
    std::vector<T> buffer_recv(_displs_recv_fwd.back());
    scatter_fwd_begin(xtl::span<const T>(send_buffer), request,
                      xtl::span<T>(buffer_recv));
    scatter_fwd_end(request);

    gather_fn(buffer_recv, _ghost_pos_recv_fwd, remote_data);
  }

  /// Send n values for each index that is owned to processes that have
  /// the index as a ghost. The size of the input array local_data must
  /// be the same as bs * size_local().
  ///
  /// @param[in] local_data Local data associated with each owned local
  /// index to be sent to process where the data is ghosted.
  /// @param[in,out] remote_data Ghost data on this process received
  /// from the owning process.
  template <typename T>
  void scatter_fwd(const xtl::span<const T>& local_data,
                   xtl::span<T> remote_data) const
  {
    scatter_fwd(local_data, remote_data, VectorScatter::gather());
  }

  /// Start a non-blocking send of ghost values to the owning rank. The
  /// non-blocking communication is completed by calling
  /// VectorScatter::scatter_rev_end. A reverse scatter is the transpose of
  /// VectorScatter::scatter_fwd_begin.
  ///
  /// @param[in] send_buffer Send buffer filled with ghost data on this
  /// process to be sent to the owning rank. The order of the data is
  /// given by VectorScatter::scatter_fwd_ghost_positions, with
  /// VectorScatter::scatter_fwd_ghost_positions()[i] being the index of the
  /// ghost data that should be placed in position `i` of the buffer.
  /// @param data_type The MPI data type. To send data with a block size
  /// use `MPI_Type_contiguous` with size `n`
  /// @param request The MPI request handle for tracking the status of
  /// the send
  /// @param recv_buffer A buffer used for the received data. It must
  /// not be changed until after a call to VectorScatter::scatter_rev_end.
  /// The ordering of the data is given by
  /// VectorScatter::scatter_fwd_indices, with
  /// VectorScatter::scatter_fwd_indices()[i] being the position in the owned
  /// data array that corresponds to position `i` in the buffer.
  template <typename T>
  void scatter_rev_begin(const xtl::span<const T>& send_buffer,
                         MPI_Request& request,
                         const xtl::span<T>& recv_buffer) const
  {
    // Get displacement vector
    const std::vector<int32_t>& displs_send_fwd = _shared_indices->offsets();

    // Return early if there are no incoming or outgoing edges
    if (_displs_recv_fwd.size() == 1 and displs_send_fwd.size() == 1)
      return;

    // Send and receive data
    MPI_Ineighbor_alltoallv(send_buffer.data(), _sizes_recv_fwd.data(),
                            _displs_recv_fwd.data(), MPI::mpi_type<T>(),
                            recv_buffer.data(), _sizes_send_fwd.data(),
                            displs_send_fwd.data(), MPI::mpi_type<T>(),
                            _map->comm(IndexMap::Direction::reverse), &request);
  }

  /// Complete a non-blocking send of ghost values to the owning rank.
  /// This function complete the communication started by
  /// VectorScatter::scatter_rev_begin.
  ///
  /// @param[in] request The MPI request handle for tracking the status
  /// of the send
  void scatter_rev_end(MPI_Request& request) const
  {
    // Return early if there are no incoming or outgoing edges
    const std::vector<int32_t>& displs_send_fwd = _shared_indices->offsets();
    if (_displs_recv_fwd.size() == 1 and displs_send_fwd.size() == 1)
      return;

    // Wait for communication to complete
    MPI_Wait(&request, MPI_STATUS_IGNORE);
  }

  /// Send n values for each ghost index to owning to the process
  ///
  /// @param[in,out] local_data Local data associated with each owned
  /// local index to be sent to process where the data is ghosted. Size
  /// must be n * size_local().
  /// @param[in] remote_data Ghost data on this process received from
  /// the owning process. Size will be n * num_ghosts().
  /// @param[in] n Number of data items per index
  /// @param[in] op Sum or set received values in local_data
  template <typename T, typename BinaryOp>
  void scatter_rev(xtl::span<T> local_data,
                   const xtl::span<const T>& remote_data, int n,
                   BinaryOp op) const
  {
    // Pack send buffer
    std::vector<T> buffer_send(_displs_recv_fwd.back());
    auto gather_fn = VectorScatter::gather();
    gather_fn(remote_data, _ghost_pos_inv, buffer_send);

    // Exchange data
    MPI_Request request;
    std::vector<T> buffer_recv(_shared_indices->array().size());
    scatter_rev_begin(xtl::span<const T>(buffer_send), request,
                      xtl::span<T>(buffer_recv));
    scatter_rev_end(request);

    // Copy or accumulate into "local_data"
    const std::vector<std::int32_t>& shared_indices = _shared_indices->array();

    auto scatter_fn = VectorScatter::scatter();
    scatter_fn(buffer_recv, shared_indices, local_data, op);
  }

  static auto gather()
  {
    return [](const auto& in, const auto& idx, auto& out)
    {
      for (std::size_t i = 0; i < idx.size(); ++i)
        out[i] = in[idx[i]];
    };
  }

  static auto scatter()
  {
    return [](const auto& in, const auto& idx, auto& out, auto& op)
    {
      for (std::size_t i = 0; i < idx.size(); ++i)
        out[idx[i]] = op(out[idx[i]], in[i]);
    };
  }

private:
  // Map describing the data layout
  std::shared_ptr<const dolfinx::common::IndexMap> _map;

  // Block size
  int _bs;

  // List of owned local indices that are in the ghost (halo) region on
  // other ranks, grouped by rank in the neighbor communicator
  // (destination ranks in forward communicator and source ranks in the
  // reverse communicator), i.e. `_shared_indices.num_nodes() ==
  // size(_comm_owner_to_ghost)`. The array _shared_indices.offsets() is
  // equivalent to 'displs_send_fwd'.
  std::unique_ptr<graph::AdjacencyList<std::int32_t>> _shared_indices;

  // MPI sizes and displacements for forward (owner -> ghost) scatter
  // Note: '_displs_send_fwd' can be got from _shared_indices->offsets()
  std::vector<std::int32_t> _sizes_send_fwd, _sizes_recv_fwd, _displs_recv_fwd;

  // Position in the recv buffer for a forward scatter for the ith ghost
  // index (_ghost[i]) entry
  // NOTE: Should be removed once ghost region is ordered
  std::vector<std::int32_t> _ghost_pos_recv_fwd;

  // Indices to map the ghost region to send buffer (ghost to owner)
  // NOTE: Should be removed once ghost region is ordered
  std::vector<std::int32_t> _ghost_pos_inv;
};

} // namespace dolfinx::common