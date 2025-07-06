import torch
import numpy
import electrock_infer as ops
if __name__ == "__main__":

    d = 7168
    num_tokens = 1024
    DTYPE = torch.bfloat16
    input_shape = (num_tokens, 2 *d)
    output_shape = (num_tokens, d)
    input_tensor = torch.randn(input_shape, device='cuda', dtype=DTYPE)
    output_tensor = torch.empty(output_shape, device='cuda', dtype=DTYPE)
    ops.silu_and_mul(output_tensor, input_tensor)
    print("Custom kernel execution complete.")

    gate_part = input_tensor[:, :d]
    up_part = input_tensor[:, d:]
# Calculate SiLU manually using PyTorch
    silu_gate_native = torch.nn.functional.silu(gate_part)
# Perform multiplication
    expected_output = silu_gate_native * up_part
# Compare your kernel's output with the native PyTorch output
# Use torch.allclose for floating-point comparisons
    tolerance = 1e-3 

    if torch.allclose(output_tensor, expected_output, atol=tolerance, rtol=tolerance):
        print("Verification successful! Custom kernel output matches PyTorch's native implementation.")
    else:
        print("Verification FAILED! Custom kernel output differs from PyTorch's native implementation.")
        # Optional: Print max absolute difference
        abs_diff = torch.abs(output_tensor - expected_output)
        print(f"Max absolute difference: {abs_diff.max().item()}")
        # Optional: Print a small portion of the tensors for debugging
        # print("Custom output sample:\n", output_tensor[0, :10])
        # print("Expected output sample:\n", expected_output[0, :10])

    # --- Basic performance check (optional) ---
    num_runs = 200
    print(f"\nRunning basic performance check ({num_runs} runs)...")

    # Warm-up run
    ops.silu_and_mul(output_tensor, input_tensor)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(num_runs):
        ops.silu_and_mul(output_tensor, input_tensor)
    end_event.record()
    torch.cuda.synchronize() # Wait for the events to complete
    cuda_time_ms = start_event.elapsed_time(end_event) / num_runs

    print(f"Average custom kernel time: {cuda_time_ms:.3f} ms/run")

    # Compare with native PyTorch (optional)
    # Warm-up run
    _ = torch.nn.functional.silu(gate_part) * up_part
    torch.cuda.synchronize()

    start_event.record()
    for _ in range(num_runs):
        _ = torch.nn.functional.silu(gate_part) * up_part
    end_event.record()
    torch.cuda.synchronize()
    native_time_ms = start_event.elapsed_time(end_event) / num_runs

    print(f"Average native PyTorch time: {native_time_ms:.3f} ms/run")