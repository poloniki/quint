import pytest

# jax is an optional, deploy-time dependency (installed via Dockerfile.jax, not
# requirements.txt), so skip this hardware check anywhere it isn't installed.
jax = pytest.importorskip("jax")


def test_jax_device():
    devices = jax.devices()

    # Check if there's any device
    assert devices, "No JAX devices available!"

    # Check device type: this is a hardware/deploy check, so skip (rather than
    # fail) when only a CPU is available, e.g. on a laptop or CI runner.
    device_types = {device.device_kind for device in devices}
    accelerators = {"gpu", "tpu", "Metal"}
    if not (device_types & accelerators):
        pytest.skip(f"No GPU/TPU/Metal accelerator available (got {device_types})")
    assert device_types & accelerators, f"Expected GPU/TPU/Metal, but got: {device_types}"
