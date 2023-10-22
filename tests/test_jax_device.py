import pytest
import jax


def test_jax_device():
    devices = jax.devices()

    # Check if there's any device
    assert devices, "No JAX devices available!"

    # Check device type
    device_types = {device.device_kind for device in devices}
    assert (
        "gpu" in device_types or "tpu" or "Metal" in device_types
    ), f"Expected GPU or TPU, but got: {device_types}"
