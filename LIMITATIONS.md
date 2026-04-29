# Limitations

This is a minimal autodiff engine built for learning purposes. The following are known limitations worth understanding before using or extending it.

---

## Gradient Representation

Gradients are stored as raw `np.ndarray` objects rather than `Tensor` objects. This means the gradient computation is a dead end — you cannot differentiate through the backward pass to get second-order derivatives automatically. Higher-order differentiation (e.g. exact Hessians via reverse-over-reverse) requires a separate implementation where gradients are themselves `Tensor` objects with their own computation graph. A prototype of this exists in the `second-order-autodiff` branch.

## Memory

Gradients accumulate via Python closures attached to each node. The entire computation graph is kept in memory until it goes out of scope. There is no mechanism to free intermediate activations during the backward pass, which makes this engine memory-inefficient for deep graphs.

## In-place Operations

In-place operations (e.g. `x += 1`) are not supported and will silently corrupt the computation graph. Always use out-of-place operations (`x = x + 1`).

## Broadcasting

Broadcasting is handled via `unbroadcast`, a manual utility that sums gradients along axes that were broadcast during the forward pass. This works correctly for common cases but is not a complete implementation — exotic broadcasting patterns may produce incorrect gradients.

## Gradient Scoping

The `req_grad` flag is coarse-grained — it applies to an entire Tensor. There is no gradient tape scoping (as in `torch.no_grad()`) to temporarily disable gradient tracking for a subgraph.

## Numerical Hessians

Hessians are computed via central finite differences rather than automatic differentiation. This introduces floating point error of order $O(\epsilon^2)$ and requires $2n$ backward passes for an $n \times n$ Hessian, making it expensive for large parameter counts. Exact Hessians via the forward-over-reverse trick would be more efficient but require the higher-order gradient representation mentioned above.

## No GPU Support

All operations run on CPU via NumPy. There is no device abstraction or GPU backend.

## Sequential Execution

There is no parallelism — operations execute sequentially. Vectorisation across batches is supported via NumPy broadcasting but the graph itself is single-threaded.

---

These limitations are intentional in the sense that removing them would each constitute a significant engineering project in their own right. The goal of this codebase is clarity and correctness on simple examples, not production readiness.