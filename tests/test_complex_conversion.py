import numpy as np

from learn_motion.data.complex import channels_to_complex, complex_to_channels


def test_complex_round_trip() -> None:
    generator = np.random.default_rng(7)
    source = generator.standard_normal((16, 12, 4)) + 1j * generator.standard_normal((16, 12, 4))
    channels = complex_to_channels(source)
    assert channels.shape == (2, 4, 16, 12)
    recovered = channels_to_complex(channels).transpose(1, 2, 0)
    np.testing.assert_allclose(recovered, source, rtol=1e-6, atol=1e-6)

