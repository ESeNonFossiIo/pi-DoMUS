#ifndef _N_FIELDS_LINEAR_PROBLEM_
#define _N_FIELDS_LINEAR_PROBLEM_


#include <deal.II/base/timer.h>
// #include <deal.II/base/parameter_handler.h>

#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/linear_operator.h>

// #include <deal.II/lac/precondition.h>

#include "assembly.h"
#include "interface.h"
#include "parsed_grid_generator.h"
#include "parsed_finite_element.h"
#include "error_handler.h"
#include "parsed_function.h"
#include "parsed_data_out.h"
#include "parameter_acceptor.h"
#include "ode_argument.h"
#include "dae_time_integrator.h"

#include "sak_data.h"

#include "mpi.h"

using namespace dealii;

typedef TrilinosWrappers::MPI::BlockVector VEC;

template <int dim, int spacedim=dim, int n_components=1>
class NFieldsProblem : public ParameterAcceptor, public OdeArgument<VEC>
{

  typedef typename Assembly::CopyData::NFieldsSystem<dim,spacedim> SystemCopyData;
  typedef typename Assembly::CopyData::NFieldsPreconditioner<dim,spacedim> PreconditionerCopyData;
  typedef typename Assembly::Scratch::NFields<dim,spacedim> Scratch;

  // This is a class required to make tests
  template<int fdim, int fspacedim>
  friend void test(NFieldsProblem<fdim,fspacedim> &);

public:

  enum RefinementMode
  {
    global_refinement=0,
    adaptive_refinement=1
  };

  NFieldsProblem (const Interface<dim,spacedim,n_components> &energy,
                  const MPI_Comm &comm=MPI_COMM_WORLD);

  virtual void declare_parameters(ParameterHandler &prm);

  void run ();


  /*********************************************************
   * Public interface from OdeArgument
   *********************************************************/
  virtual shared_ptr<VEC>
  create_new_vector() const;

  /** Returns the number of degrees of freedom. Pure virtual function. */
  virtual unsigned int n_dofs() const;

  /** This function is called at the end of each iteration step for
   * the ode solver. Once again, the conversion between pointers and
   * other forms of vectors need to be done inside the inheriting
   * class. */
  virtual void output_step(const double t,
                           const VEC &solution,
                           const VEC &solution_dot,                           const unsigned int step_number,
                           const double h);

  /** This function will check the behaviour of the solution. If it
   * is converged or if it is becoming unstable the time integrator
   * will be stopped. If the convergence is not achived the
   * calculation will be continued. If necessary, it can also reset
   * the time stepper. */
  virtual bool solution_check(const double t,
                              const VEC &solution,
                              const VEC &solution_dot,
                              const unsigned int step_number,
                              const double h) const;

  /** For dae problems, we need a
   residual function. */
  virtual int residual(const double t,
                       const VEC &src_yy,
                       const VEC &src_yp,
                       VEC &dst) const;

  /** Jacobian vector product. */
  virtual int jacobian(const double t,
                       const VEC &src_yy,
                       const VEC &src_yp,
                       const double alpha,
                       const VEC &src,
                       VEC &dst);

  /** Setup Jacobian preconditioner. */
  virtual int setup_jacobian_prec(const double t,
                                  const VEC &src_yy,
                                  const VEC &src_yp,
                                  const double alpha);

  /** Jacobian preconditioner
   vector product. */
  virtual int jacobian_prec(const double t,
                            const VEC &src_yy,
                            const VEC &src_yp,
                            const double alpha,
                            const VEC &src,
                            VEC &dst) const;

  /** And an identification of the
   differential components. This
   has to be 1 if the
   corresponding variable is a
   differential component, zero
   otherwise.  */
  virtual VEC &differential_components() const;

private:
  void make_grid_fe();
  void setup_dofs ();
  void assemble_preconditioner (const double t,
                                const VEC &y,
                                const VEC &y_dot,
                                const double alpha);

  void assemble_system (const double t,
                        const VEC &y,
                        const VEC &y_dot,
                        const double alpha);
  void solve ();
  //void refine_mesh (const unsigned int max_grid_level);
  void refine_mesh ();
  double compute_residual(const double alpha); // const;
  double determine_step_length () const;
  void process_solution ();

  const MPI_Comm &comm;
  const Interface<dim,spacedim,n_components>    &energy;




  unsigned int n_cycles;
  unsigned int initial_global_refinement;
  unsigned int max_time_iterations;
  double fixed_alpha;

  ConditionalOStream        pcout;
  std::ofstream         timer_outfile;
  ConditionalOStream        tcout;

  shared_ptr<Mapping<dim,spacedim> >             mapping;

  shared_ptr<parallel::distributed::Triangulation<dim,spacedim> > triangulation;
  shared_ptr<FiniteElement<dim,spacedim> >       fe;
  shared_ptr<DoFHandler<dim,spacedim> >          dof_handler;

  ConstraintMatrix                          constraints;

  TrilinosWrappers::BlockSparseMatrix       matrix;
  TrilinosWrappers::BlockSparseMatrix       preconditioner_matrix;

  LinearOperator<TrilinosWrappers::MPI::BlockVector> preconditioner_op;
  LinearOperator<TrilinosWrappers::MPI::BlockVector> system_op;

  TrilinosWrappers::MPI::BlockVector        solution;
  TrilinosWrappers::MPI::BlockVector        solution_dot;

  TimerOutput                               computing_timer;

  void setup_matrix ( const std::vector<IndexSet> &partitioning,
                      const std::vector<IndexSet> &relevant_partitioning);
  void setup_preconditioner ( const std::vector<IndexSet> &partitioning,
                              const std::vector<IndexSet> &relevant_partitioning);


  void
  local_assemble_preconditioner (const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                                 Assembly::Scratch::NFields<dim,spacedim> &scratch,
                                 Assembly::CopyData::NFieldsPreconditioner<dim,spacedim> &data);

  void
  copy_local_to_global_preconditioner (const Assembly::CopyData::NFieldsPreconditioner<dim,spacedim> &data);


  void
  local_assemble_system (const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                         Assembly::Scratch::NFields<dim,spacedim>  &scratch,
                         Assembly::CopyData::NFieldsSystem<dim,spacedim> &data);

  void
  copy_local_to_global_system (const Assembly::CopyData::NFieldsSystem<dim,spacedim> &data);

  ErrorHandler<1>       eh;
  ParsedGridGenerator<dim,spacedim>   pgg;

  ParsedFunction<spacedim, n_components>        exact_solution;

  ParsedDataOut<dim, spacedim>                  data_out;

  DAETimeIntegrator<VEC>  dae;
};

#endif
