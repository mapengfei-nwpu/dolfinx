// Copyright (C) 2014-2015 Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2014-06-10
// Last changed: 2015-11-11
//
// This demo program solves the Stokes equations on a domain defined
// by three overlapping and non-matching meshes.

#include <dolfin.h>
#include "MultiMeshStokes.h"

using namespace dolfin;

// Value for inflow boundary condition for velocity
class InflowValue : public Expression
{
public:

  InflowValue() : Expression(2) {}

  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = sin(x[1]*DOLFIN_PI);
    values[1] = 0.0;
  }

};

// Subdomain for no-slip boundary
class NoslipBoundary : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return on_boundary && (near(x[1], 0.0) || near(x[1], 1.0));
  }
};

// Subdomain for inflow boundary
class InflowBoundary : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return on_boundary && near(x[0], 0.0);
  }
};

// Subdomain for outflow boundary
class OutflowBoundary : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return on_boundary && near(x[0], 1.0);
  }
};

class AllDomain : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return true;
  }
};

int main(int argc, char* argv[])
{
  if (dolfin::MPI::size(MPI_COMM_WORLD) > 1)
  {
    info("Sorry, this demo does not (yet) run in parallel.");
    return 0;
  }

  // FIXME: Testing
  parameters["reorder_dofs_serial"] = false;

  // Create meshes
  int N = 32;
  UnitSquareMesh mesh_0(N, N);
  RectangleMesh  mesh_1(Point(0.2, 0.2), Point(0.6, 0.6), N, N);
  RectangleMesh  mesh_2(Point(0.4, 0.4), Point(0.8, 0.8), N, N);

  // Build multimesh
  MultiMesh multimesh;
  multimesh.add(mesh_0);
  multimesh.add(mesh_1);
  multimesh.add(mesh_2);
  multimesh.build();

  // Create function space
  MultiMeshStokes::MultiMeshFunctionSpace W(multimesh);
  W.parameters("multimesh")["quadrature_order"] = 2;

  // Create forms
  MultiMeshStokes::MultiMeshBilinearForm a(W, W);
  MultiMeshStokes::MultiMeshLinearForm L(W);

  // Attach coefficients
  Constant f(0, 0);
  L.f = f;

  // Assemble linear system
  Matrix A;
  Vector b;
  assemble_multimesh(A, a);
  assemble_multimesh(b, L);

  // Create boundary values
  Constant noslip_value(0, 0);
  InflowValue inflow_value;
  Constant outflow_value(0);

  // Create subdomains for boundary conditions
  NoslipBoundary noslip_boundary;
  InflowBoundary inflow_boundary;
  OutflowBoundary outflow_boundary;

  // Create subspaces for boundary conditions
  MultiMeshSubSpace V(W, 0);
  MultiMeshSubSpace Q(W, 1);

  // Create boundary conditions
  MultiMeshDirichletBC bc0(V, noslip_value,  noslip_boundary);
  MultiMeshDirichletBC bc1(V, inflow_value,  inflow_boundary);
  MultiMeshDirichletBC bc2(Q, outflow_value, outflow_boundary);

  // Apply boundary conditions
  bc0.apply(A, b);
  bc1.apply(A, b);
  bc2.apply(A, b);

  // Compute solution
  MultiMeshFunction w(W);
  solve(A, *w.vector(), b);

  /*

  // Extract solution parts and components
  Function u0 = (*w.part(0))[0];
  Function u1 = (*w.part(1))[0];
  Function u2 = (*w.part(2))[0];
  Function p0 = (*w.part(0))[1];
  Function p1 = (*w.part(1))[1];
  Function p2 = (*w.part(2))[1];

  // Save to file
  File u0_file("u0.pvd");
  File u1_file("u1.pvd");
  File u2_file("u2.pvd");
  File p0_file("p0.pvd");
  File p1_file("p1.pvd");
  File p2_file("p2.pvd");
  u0_file << u0;
  u1_file << u1;
  u2_file << u2;
  p0_file << p0;
  p1_file << p1;
  p2_file << p2;

  // Plot solution
  plot(W.multimesh());
  plot(u0, "u_0");
  plot(u1, "u_1");
  plot(u2, "u_2");
  plot(p0, "p_0");
  plot(p1, "p_1");
  plot(p2, "p_2");

  */

  /*

  // FIXME: Temporary fix until extraction of subdofmaps works
  AllDomain all_domain;
  Constant c(0);
  MultiMeshDirichletBC bc(Q, c, all_domain);
  bc.apply(*w.vector());
  plot(w.part(0), "u_0");
  plot(w.part(1), "u_1");
  plot(w.part(2), "u_2");
  plot(W.multimesh());
  File u0_file("u0.pvd");
  File u1_file("u1.pvd");
  File u2_file("u2.pvd");
  u0_file << *w.part(0);
  u1_file << *w.part(1);
  u2_file << *w.part(2);

  interactive();

  */

  return 0;
}
