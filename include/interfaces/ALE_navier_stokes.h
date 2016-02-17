/*! \addtogroup equations
 * @{
 */

/**
 * This interface solves ALE Navier Stokes Equation:
 *
 */

#ifndef _pidomus_ALE_navier_stokes_h_
#define _pidomus_ALE_navier_stokes_h_

#include "pde_system_interface.h"

#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/block_linear_operator.h>
#include <deal.II/lac/packaged_operation.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>

#include <deal.II/base/conditional_ostream.h>

template <int dim, int spacedim=dim, typename LAC=LATrilinos>
class ALENavierStokes
  :
  public PDESystemInterface<dim,spacedim,ALENavierStokes<dim,spacedim,LAC>, LAC>
{

public:
  ~ALENavierStokes () {};
  ALENavierStokes (bool dynamic, bool convection);

  void declare_parameters (ParameterHandler &prm);
  void parse_parameters_call_back ();

  template <typename EnergyType, typename ResidualType>
  void
  energies_and_residuals(
    const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
    FEValuesCache<dim,spacedim> &scratch,
    std::vector<EnergyType> &energies,
    std::vector<std::vector<ResidualType>> &residuals,
    bool compute_only_system_terms) const;

  void
  compute_system_operators(
    const std::vector<shared_ptr<LATrilinos::BlockMatrix>>,
    LinearOperator<LATrilinos::VectorType> &,
    LinearOperator<LATrilinos::VectorType> &) const;

  void
  set_matrix_couplings(std::vector<std::string> &couplings) const;

  virtual UpdateFlags get_face_update_flags() const
  {
    return (update_values |
            update_gradients | /* this is the new entry */
            update_quadrature_points |
            update_normal_vectors |
            update_JxW_values);
  }

private:
  mutable ParsedMappedFunctions<spacedim> nietsche;

  double nu;
  double rho;

// mutable shared_ptr<TrilinosWrappers::PreconditionAMG> amg_A;
  mutable shared_ptr<TrilinosWrappers::PreconditionAMG> P00_preconditioner, P11_preconditioner, P22_preconditioner;
  mutable shared_ptr<TrilinosWrappers::PreconditionJacobi> P33_preconditioner;

  ConditionalOStream pcout;
};


template <int dim, int spacedim, typename LAC>
ALENavierStokes<dim,spacedim, LAC>::
ALENavierStokes(bool dynamic, bool convection)
  :
  PDESystemInterface<dim,spacedim,ALENavierStokes<dim,spacedim,LAC>, LAC>(
    "ALE Navier Stokes Interface",
    dim+dim+dim+1,
    2,
    "FESystem[FE_Q(1)^d-FE_Q(1)^d-FE_Q(2)^d-FE_Q(1)]",
    "d,d,v,v,u,u,p",
    "1,1,1,0"),
  nietsche("Nietsche boundary conditions",
           this->n_components,
           this->get_component_names(),
           "" /* do nothing by default */
          ),
  pcout(std::cout,
        Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
{
  this->init();
}

template <int dim, int spacedim, typename LAC>
void ALENavierStokes<dim,spacedim,LAC>::
declare_parameters (ParameterHandler &prm)
{
  PDESystemInterface<dim,spacedim, ALENavierStokes<dim,spacedim,LAC>,LAC>::
  declare_parameters(prm);

  this->add_parameter(prm, &nu,
                      "nu [Pa s]", "1.0",
                      Patterns::Double(0.0),
                      "Viscosity");

  this->add_parameter(prm, &rho,
                      "rho [Kg m^-d]", "1.0",
                      Patterns::Double(0.0),
                      "Density");
}

template <int dim, int spacedim, typename LAC>
void ALENavierStokes<dim,spacedim,LAC>::
parse_parameters_call_back ()
{}

template <int dim, int spacedim, typename LAC>
void ALENavierStokes<dim,spacedim,LAC>::
set_matrix_couplings(std::vector<std::string> &couplings) const
{
  couplings[0] = "1,1,1,1; 1,1,1,1; 1,1,1,1; 1,1,1,1"; // TODO: Select only not null entries
  couplings[1] = "0,0,0,0; 0,0,0,0; 0,0,0,0; 0,0,0,1";
}

double get_double(Sdouble num)
{
  return num.val();
}

double get_double(double num)
{
  return num;
}

template <int dim, int spacedim, typename LAC>
template <typename EnergyType, typename ResidualType>
void
ALENavierStokes<dim,spacedim,LAC>::
energies_and_residuals(const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                       FEValuesCache<dim,spacedim> &fe_cache,
                       std::vector<EnergyType> &,
                       std::vector<std::vector<ResidualType> > &residual,
                       bool compute_only_system_terms) const
{
  const FEValuesExtractors::Vector displacement(0);
  const FEValuesExtractors::Vector displacement_velocity(dim);
  const FEValuesExtractors::Vector velocity(2*dim);
  const FEValuesExtractors::Scalar pressure(3*dim);

  ResidualType et = this->alpha;
  double dummy = 0.0;

  for (unsigned int face=0; face < GeometryInfo<dim>::faces_per_cell; ++face)
    {
      unsigned int face_id = cell->face(face)->boundary_id();
      if (cell->face(face)->at_boundary() && nietsche.acts_on_id(face_id))
        {
          this->reinit(et, cell, face, fe_cache);

// Displacement:
          auto &d_dot_ = fe_cache.get_values( "solution_dot", "d_dot", displacement, et);

// Displacement velocity:
          auto &v_ = fe_cache.get_values("solution", "grad_v", displacement_velocity, et);
          auto &v_dot_ = fe_cache.get_values( "solution_dot", "v_dot", displacement_velocity, et);

// Velocity:
          auto &grad_u_ = fe_cache.get_gradients("solution", "u", velocity, et);
          auto &u_ = fe_cache.get_values("solution", "grad_u", velocity, et);

// Pressure:
          auto &grad_p_ = fe_cache.get_gradients("solution", "p", pressure, et);

          auto &fev = fe_cache.get_current_fe_values();
          auto &q_points = fe_cache.get_quadrature_points();
          auto &JxW = fe_cache.get_JxW_values();

          for (unsigned int q=0; q<q_points.size(); ++q)
            {
              const Tensor<1, dim, double> n = fev.normal_vector(q);

// Displacement:
              const Tensor<1, dim, ResidualType> &d_dot = d_dot_[q];

// Displacement velocity:
              const Tensor<1, dim, ResidualType> &v = v_[q];
              const Tensor<1, dim, ResidualType> &v_dot = v_dot_[q];

// Velocity:
              const Tensor<1, dim, ResidualType> &u = u_[q];
              const Tensor<2, dim, ResidualType> &grad_u = grad_u_[q];

// Pressure:
              const Tensor<1, dim, ResidualType> &grad_p = grad_p_[q];

              for (unsigned int i=0; i<residual[0].size(); ++i)
                {
// Test functions:
                  auto d_test = fev[displacement].value(i,q);
                  auto v_test = fev[displacement_velocity].value(i,q);
                  auto u_test = fev[velocity].value(i,q);

// std::cout << "mignotta!"<<std::endl;
                  Tensor<1, dim, double> f_double;
                  Tensor<1, dim, ResidualType> force = grad_u * n + grad_p;
                  f_double[1]=1; //TODO: get_double(force[1]);
                  residual[0][i] += (
                                      (v_dot - f_double) * v_test
// + (u-v) * u_test
                                    )*JxW[q];
                }
            } // end loop over quadrature points
          break;
        } // endif face->at_boundary
    } // end loop over faces

  this->reinit (et, cell, fe_cache);

// displacement:
  auto &ds = fe_cache.get_values( "solution", "d", displacement, et);
  auto &grad_ds = fe_cache.get_gradients( "solution", "grad_d", displacement, et);
  auto &div_ds = fe_cache.get_divergences( "solution", "div_d", displacement, et);
  auto &Fs = fe_cache.get_deformation_gradients( "solution", "Fd", displacement, et);
  auto &ds_dot = fe_cache.get_values( "solution_dot", "d_dot", displacement, et);

// Displacement velocity:
  auto &v_ = fe_cache.get_values( "solution", "v", displacement_velocity, et);

// velocity:
  auto &us = fe_cache.get_values( "solution", "u", velocity, et);
  auto &grad_us = fe_cache.get_gradients( "solution", "grad_u", velocity, et);
  auto &div_us = fe_cache.get_divergences( "solution", "div_u", velocity, et);
  auto &sym_grad_us = fe_cache.get_symmetric_gradients( "solution", "u", velocity, et);
  auto &us_dot = fe_cache.get_values( "solution_dot", "u_dot", velocity, et);

// Previous time step solution:
  auto &u_olds = fe_cache.get_values("explicit_solution","u",velocity,dummy);
  auto &ds_dot_old = fe_cache.get_values("explicit_solution","d_dot",displacement,dummy);


// pressure:
  auto &ps = fe_cache.get_values( "solution", "p", pressure, et);

// Jacobian:
  auto &JxW = fe_cache.get_JxW_values();

  const unsigned int n_quad_points = us.size();

  auto &fev = fe_cache.get_current_fe_values();

  for (unsigned int quad=0; quad<n_quad_points; ++quad)
    {
// variables:
// velocity:
      const Tensor<1, dim, ResidualType> &u = us[quad];
      const ResidualType &div_u = div_us[quad];
      const Tensor<1, dim, ResidualType> &u_dot = us_dot[quad];
      const Tensor<2, dim, ResidualType> &grad_u = grad_us[quad];
      const Tensor <2, dim, ResidualType> &sym_grad_u = sym_grad_us[quad];
// displacement
      const Tensor<1, dim, ResidualType> &d = ds[quad];
      const Tensor<1, dim, ResidualType> &d_dot = ds_dot[quad];
      const Tensor<2, dim, ResidualType> &grad_d = grad_ds[quad];
      const ResidualType &div_d = div_ds[quad];
// Displacement velocity:
      const Tensor<1, dim, ResidualType> &v = v_[quad];

      const Tensor <2, dim, ResidualType> &F = Fs[quad];
      ResidualType J = determinant(F);
      const Tensor <2, dim, ResidualType> &F_inv = invert(F);
      const Tensor <2, dim, ResidualType> &Ft_inv = transpose(F_inv);

// Previous time step solution:
      const Tensor<1, dim, ResidualType> &u_old = u_olds[quad];
      const Tensor<1, dim, ResidualType> &d_dot_old = ds_dot_old[quad];

// pressure:
      const ResidualType &p = ps[quad];

// others:
      auto J_ale = J; // jacobian of ALE transformation
// auto div_u_ale = (J_ale * (F_inv * u) );
      Tensor <2, dim, ResidualType> Id;
      for (unsigned int i = 0; i<dim; ++i)
        Id[i][i] = p;

      ResidualType my_rho = rho;
      const Tensor <2, dim, ResidualType> sigma = 
      - Id + my_rho * ( nu* sym_grad_u * F_inv + ( Ft_inv * transpose(sym_grad_u) ) ) ;

      for (unsigned int i=0; i<residual[0].size(); ++i)
        {
// test functions:
// Displacement velocity:
          auto v_test = fev[displacement_velocity].value(i,quad);

// velocity:
          auto u_test = fev[velocity].value(i,quad);
          auto grad_v = fev[velocity].gradient(i,quad);
          auto div_v = fev[velocity].divergence(i,quad);
          auto sym_grad_v = fev[velocity].symmetric_gradient(i,quad);

// displacement:
          auto grad_e = fev[displacement].gradient(i,quad);
          auto e = fev[displacement].value(i,quad);

// pressure:
          auto m = fev[pressure].value(i,quad);
          auto q = fev[pressure].value(i,quad);
          auto grad_q = fev[pressure].gradient(i,quad);

          residual[1][i] +=
            (
              (1./nu)*p*m
            )*JxW[quad];
          residual[0][i] +=
            (
// time derivative term
              rho*scalar_product( u_dot * J_ale , u_test )
//
              + scalar_product( grad_u * ( F_inv * ( u_old - d_dot ) ) * J_ale , u_test )
//
              + scalar_product( J_ale * sigma * Ft_inv, grad_v )

// "stiffness" of the displacement
              + nu * scalar_product( grad_d , grad_e )

// divergence free constriant
              - div_u * m

// pressure term
              - p * div_v

              + (v-d_dot) * v_test
            )*JxW[quad];

        }
    }

  (void)compute_only_system_terms;
}


template <int dim, int spacedim, typename LAC>
void
ALENavierStokes<dim,spacedim,LAC>::compute_system_operators(
  const std::vector<shared_ptr<LATrilinos::BlockMatrix>> matrices,
  LinearOperator<LATrilinos::VectorType> &system_op,
  LinearOperator<LATrilinos::VectorType> &prec_op) const
{
  typedef LATrilinos::VectorType::BlockType BVEC;
  typedef LATrilinos::VectorType VEC;

// AMG data d
  TrilinosWrappers::PreconditionAMG::AdditionalData Amg_data_d;
  Amg_data_d.elliptic = true;
  Amg_data_d.higher_order_elements = true;
  Amg_data_d.smoother_sweeps = 2;
  Amg_data_d.aggregation_threshold = 0.02;

// AMG data v
  TrilinosWrappers::PreconditionAMG::AdditionalData Amg_data_v;
  std::vector<std::vector<bool>> constant_modes_v;
  FEValuesExtractors::Vector d_velocity_components(dim);
  DoFTools::extract_constant_modes (
    *this->dof_handler,
    this->dof_handler->get_fe().component_mask(d_velocity_components),
    constant_modes_v);
  Amg_data_v.constant_modes = constant_modes_v;
  Amg_data_v.elliptic = true;
  Amg_data_v.higher_order_elements = true;
  Amg_data_v.smoother_sweeps = 2;
  Amg_data_v.aggregation_threshold = 0.02;

// AMG data u
  TrilinosWrappers::PreconditionAMG::AdditionalData Amg_data_u;
  std::vector<std::vector<bool>> constant_modes_u;
  FEValuesExtractors::Vector velocity_components(2*dim);
  DoFTools::extract_constant_modes (
    *this->dof_handler,
    this->dof_handler->get_fe().component_mask(velocity_components),
    constant_modes_u);
  Amg_data_u.constant_modes = constant_modes_u;
  Amg_data_u.elliptic = true;
  Amg_data_u.higher_order_elements = true;
  Amg_data_u.smoother_sweeps = 2;
  Amg_data_u.aggregation_threshold = 0.02;

// Preconditioners:

  P00_preconditioner.reset (new TrilinosWrappers::PreconditionAMG());
  P00_preconditioner ->initialize (matrices[0]->block(0,0), Amg_data_d);
  P11_preconditioner.reset (new TrilinosWrappers::PreconditionAMG());
  P11_preconditioner ->initialize (matrices[0]->block(1,1), Amg_data_v);
  P22_preconditioner.reset (new TrilinosWrappers::PreconditionAMG());
  P22_preconditioner ->initialize (matrices[0]->block(2,2), Amg_data_u);
  P33_preconditioner.reset (new TrilinosWrappers::PreconditionJacobi());
  P33_preconditioner ->initialize (matrices[1]->block(3,3), 1.4);

////////////////////////////////////////////////////////////////////////////
// SYSTEM MATRIX:

  std::array<std::array<LinearOperator< BVEC >, 4>, 4> S;
  for (unsigned int i = 0; i<4; ++i)
    for (unsigned int j = 0; j<4; ++j)
      S[i][j] = linear_operator< BVEC >(matrices[0]->block(i,j) );
  system_op = BlockLinearOperator< VEC >(S);

////////////////////////////////////////////////////////////////////////////
// PRECONDITIONER MATRIX:

  std::array<std::array<LinearOperator<TrilinosWrappers::MPI::Vector>, 4>, 4> P;
  for (unsigned int i = 0; i<4; ++i)
    for (unsigned int j = 0; j<4; ++j)
      P[i][j] = linear_operator< TrilinosWrappers::MPI::Vector >( matrices[0]->block(i,j) );

  static ReductionControl solver_control_pre(5000, 1e-8);
  static SolverCG<TrilinosWrappers::MPI::Vector> solver_CG(solver_control_pre);
  static SolverGMRES<TrilinosWrappers::MPI::Vector> solver_GMRES(solver_control_pre);



  for (unsigned int i = 0; i<4; ++i)
    for (unsigned int j = 0; j<4; ++j)
      if (i!=j)
        P[i][j] = null_operator< TrilinosWrappers::MPI::Vector >(P[i][j]);

  auto A = linear_operator< BVEC>(matrices[0]->block(2,2) );
  auto B = linear_operator< BVEC>(matrices[0]->block(3,2) );
  auto Bt = transpose_operator< >(B);

  auto A_inv = inverse_operator( P[2][2],
                                 solver_GMRES,
                                 *P22_preconditioner);

  auto Mp = linear_operator< TrilinosWrappers::MPI::Vector >( matrices[1]->block(3,3) );
  auto Mp_inv = inverse_operator( Mp, solver_CG, *P33_preconditioner);

  auto Schur_inv = nu * Mp_inv;

  P[0][0] = inverse_operator< >( S[0][0],
                                 solver_CG,
                                 *P00_preconditioner);
  P[1][1] = inverse_operator< >( S[1][1],
                                 solver_CG,
                                 *P11_preconditioner);
  P[2][2] = A_inv;
  P[2][3] = A_inv * Bt * Schur_inv;
  P[3][2] = null_operator(B);
  P[3][3] = -1 * Schur_inv;


  prec_op = BlockLinearOperator< VEC >(P);
}

#endif

/*! @} */
