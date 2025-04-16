from scipy.optimize import minimize
import matplotlib.pyplot as plt

def black_box_flow(domain, gamma, gamma_R, gamma_a, gamma_v, k_v):
    flow_solve = fem.SubCubes(
        domain = TEST_DOMAIN,
        gamma = 1.0e-6,
        gamma_a = 1.0e-6,
        gamma_R = 1.0e-6,
        mu = 1.0e-3,
        k_t = 1.0e-10,
        k_v = 1.0e-10,
        P_in = 100.0 * 133.322,
        P_cvp = 1.0 * 133.322,
        lower_cube_bounds = [[0.0, 0.0, 0.0], [0.010, 0.010, 0.010]],
        upper_cube_bounds = [[0.033, 0.030, 0.010],[0.043, 0.040, 0.020]]
    )
    flow_val = flow_solve.all_net_flow_dolfin()
    return flow_val

def cost_function_log(params_log, target_flow):
    gamma_linear   = 10.0**(params_log[0])
    gamma_R_linear = 10.0**(params_log[1])
    gamma_a_linear = 10.0**(params_log[2])
    gamma_v_linear = 10.0**(params_log[3])
    k_v_linear     = 10.0**(params_log[4])

    # Evaluate the flow from the PDE solver
    flow_val = black_box_flow(
        gamma_linear,
        gamma_R_linear,
        gamma_a_linear,
        gamma_v_linear,
        k_v=k_v_linear
    )

    # Least-squares difference from the target flow
    return (flow_val - target_flow)**2

def multi_param_fit(
    target_flow=500.6,
    initial_guess_log=None,
    max_iter=50
):
    if initial_guess_log is None:
        initial_guess_log = [0.0, 0.0, 0.0, 0.0, -8.0]

    n_params = len(initial_guess_log)
    initial_simplex = [np.array(initial_guess_log)]
    for i in range(n_params):
        vertex = np.array(initial_guess_log, copy=True)
        vertex[i] += 0.01
        initial_simplex.append(vertex)
    initial_simplex = np.array(initial_simplex)

    cost_history = []

    def callback(xk):
        cost = cost_function_log(xk, target_flow=target_flow)
        cost_history.append(cost)
        iteration_num = len(cost_history)
        print(f"Iteration {iteration_num}: parameters (log scale) = {xk}, cost = {cost}")

    result = minimize(
        fun=lambda p: cost_function_log(p, target_flow=target_flow),
        x0=np.array(initial_guess_log),
        method='Nelder-Mead',
        callback=callback,
        options={
            'maxiter': max_iter,
            'disp': True,
            'initial_simplex': initial_simplex
        }
    )

    best_log = result.x
    gamma_opt   = 10.0**(best_log[0])
    gammaR_opt  = 10.0**(best_log[1])
    gammaA_opt  = 10.0**(best_log[2])
    gammaV_opt  = 10.0**(best_log[3])
    k_v_opt     = 10.0**(best_log[4])

    flow_final = black_box_flow(gamma_opt, gammaR_opt, gammaA_opt, gammaV_opt, k_v_opt)

    # Plot the cost history
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(cost_history) + 1), cost_history, marker='o', linestyle='-')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Cost vs. Iterations')
    plt.grid(True)
    plt.show()

    return {
        'success': result.success,
        'status': result.status,
        'message': result.message,
        'n_iters': result.nit,
        'cost_final': result.fun,
        'gamma':   gamma_opt,
        'gamma_R': gammaR_opt,
        'gamma_a': gammaA_opt,
        'gamma_v': gammaV_opt,
        'k_v':     k_v_opt,
        'flow_final': flow_final,
        'cost_history': cost_history
    }

res = multi_param_fit(target_flow=5.0e-6, max_iter=50)

print("Optimization successful?", res['success'])
print("Status code:", res['status'])
print("Message:", res['message'])
print("Number of iterations:", res['n_iters'])
print("Final cost:", res['cost_final'])
print(f"gamma    = {res['gamma']:8.4g}")
print(f"gamma_R  = {res['gamma_R']:8.4g}")
print(f"gamma_a  = {res['gamma_a']:8.4g}")
print(f"gamma_v  = {res['gamma_v']:8.4g}")
print(f"k_v      = {res['k_v']:8.4g}")
print(f"flow     = {res['flow_final']:8.4g}")