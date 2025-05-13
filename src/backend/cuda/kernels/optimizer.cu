#include <math.h>

extern "C" {

// Adam optimizer kernel
__global__ void adam_step_kernel(
    float* param,         // Output: Updated parameters
    const float* grad,    // Input: Gradients
    float* m,             // Input/Output: 1st moment estimate
    float* v,             // Input/Output: 2nd raw moment estimate
    float lr,             // Input: Learning rate
    float beta1,          // Input: Beta1 hyperparameter
    float beta2,          // Input: Beta2 hyperparameter
    float epsilon,        // Input: Epsilon hyperparameter
    int t,                // Input: Timestep (use int)
    int n)                // Input: Number of elements
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float p_old = param[idx];
        float g_t = grad[idx];
        float m_old = m[idx];
        float v_old = v[idx];
        float m_t = beta1 * m_old + (1.0f - beta1) * g_t;
        float v_t = beta2 * v_old + (1.0f - beta2) * (g_t * g_t);
        float beta1_pow_t = powf(beta1, (float)t);
        float beta2_pow_t = powf(beta2, (float)t);
        float m_hat = m_t / (1.0f - beta1_pow_t);
        float v_hat = v_t / (1.0f - beta2_pow_t);
        float p_new = p_old - lr * m_hat / (sqrtf(v_hat) + epsilon);
        param[idx] = p_new;
        m[idx] = m_t;
        v[idx] = v_t;
    }
}

// Momentum SGD optimizer kernel
__global__ void momentum_sgd_step_kernel(
    float* param,         // Output: Updated parameters
    const float* grad,    // Input: Gradients
    float* velocity,      // Input/Output: Velocity state
    float lr,             // Input: Learning rate
    float momentum,       // Input: Momentum factor
    int n                 // Input: Number of elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v_old = velocity[idx];
        float g_t = grad[idx];
        float v_t = momentum * v_old + g_t;
        velocity[idx] = v_t; // Store updated velocity
        param[idx] -= lr * v_t; // Update parameter
    }
}

// AdaGrad optimizer kernel
__global__ void adagrad_step_kernel(
    float* param,               // Output: Updated parameters
    const float* grad,          // Input: Gradients
    float* accum_sq_grad,       // Input/Output: Accumulated squared gradient state
    float lr,                   // Input: Learning rate
    float epsilon,              // Input: Epsilon hyperparameter
    int n)                      // Input: Number of elements
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float g_t = grad[idx];
        float acc_old = accum_sq_grad[idx];

        // Update accumulated squared gradient state: G_t = G_{t-1} + g_t^2
        float acc_new = acc_old + g_t * g_t;
        accum_sq_grad[idx] = acc_new; // Update state in-place

        // Update parameter: theta_t = theta_{t-1} - lr * g_t / (sqrt(G_t) + epsilon)
        param[idx] = param[idx] - lr * g_t / (sqrtf(acc_new) + epsilon);
    }
}

// (Add other kernels here as needed)

} // extern "C"
