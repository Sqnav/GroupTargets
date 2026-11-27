"""CPU-only FFT fixtures; these tests never open a sensor dataset or weights.

Fixtures are inverse-FFT impulses, not an approximate scattering simulator.
They exercise the transform's measured axis/sign/normalization conventions.
"""
import inspect

import numpy as np
import pytest
from scipy import fft

from rvloc.contracts import RFMetadata
from rvloc.radar import C, FFTDetector, RadarSettings


def rf_mapping():
    carrier = 4.9e9
    return dict(carrier_hz=carrier, bandwidth_hz=32e6, pulse_interval_s=.0025,
                shape=(40, 128, 16, 16), spacing_y_m=C / carrier / 2,
                spacing_z_m=C / carrier / 2, frame_interval_s=1.,
                frequencies_hz=((np.arange(128) - 64) * 32e6 / 128).tolist(),
                frequency_grid="centered_baseband_uniform")


def fft_impulse_iq(cells, pulse_gains=None):
    """Invert a range impulse carrying exactly native spatial FFT tones.

    Each tuple is (native range index, native y bin, native z bin, amplitude).
    Spatial bins use signed fftfreq ordering, so no physical forward model is
    needed to manufacture the calibration input.
    """
    q, k, ny, nz = 40, 128, 16, 16
    profiles = np.zeros((k, ny, nz), np.complex64)
    y, z = np.arange(ny)[:, None], np.arange(nz)[None, :]
    for r, by, bz, amplitude in cells:
        profiles[r] += (np.sqrt(k) * amplitude * np.exp(2j * np.pi * (by * y / ny + bz * z / nz))).astype(np.complex64)
    native = fft.fftshift(fft.fft(profiles, axis=0, norm="ortho"), axes=0)
    gains = np.ones(q) if pulse_gains is None else np.asarray(pulse_gains)
    return (gains[:, None, None, None] * native[None]).astype(np.complex64)


def test_explicit_frequencies_and_metadata_are_required():
    rf = rf_mapping()
    detector = FFTDetector(RFMetadata.from_mapping(rf))
    assert detector.shape == (40, 128, 16, 16)
    del rf["frequencies_hz"]
    with pytest.raises(ValueError, match="frequencies"):
        FFTDetector(rf)
    rf = rf_mapping()
    rf["frequencies_hz"] = np.linspace(-16e6, 16e6, 128)
    with pytest.raises(ValueError, match="grid"):
        FFTDetector(rf)


def test_native_range_spatial_sign_and_coherent_power():
    detector = FFTDetector(rf_mapping())
    iq = fft_impulse_iq([(21, -3, 2, 1.)])
    result = detector.detect(iq)
    assert len(result) == 1
    radius = 21 * C / (2 * 32e6)
    u, v = -3 / 8, 2 / 8
    expected = radius * np.array([np.sqrt(1 - u*u - v*v), u, v])
    np.testing.assert_allclose(result.points_base[0], expected, rtol=0, atol=1e-10)
    np.testing.assert_allclose(result.metadata["peak_power"], [128 * 16 * 16], rtol=2e-6)
    assert result.valid.tolist() == [True]
    assert result.metadata["uses_history"] is False
    assert result.metadata["uses_source_labels"] is False


def test_joint_cube_keeps_same_range_pairing_without_cross_products():
    detector = FFTDetector(rf_mapping())
    result = detector.detect(fft_impulse_iq([(30, -2, -2, 1.), (30, 2, 2, 1.)]))
    assert len(result) == 2
    assert set(zip(result.metadata["u"], result.metadata["v"])) == {(-.25, -.25), (.25, .25)}
    assert (-.25, .25) not in zip(result.metadata["u"], result.metadata["v"])


def test_observer_annotation_never_erases_a_peak_or_whole_cell():
    detector = FFTDetector(rf_mapping(), dict(observer_tag_radius_m=10.))
    iq = fft_impulse_iq([(20, 0, 0, 1.), (20, 2, 0, .8)])
    normal = detector.detect(iq)
    tagged = detector.detect(iq, observer=normal.points_base[0])
    np.testing.assert_array_equal(tagged.points_base, normal.points_base)
    np.testing.assert_array_equal(tagged.valid, normal.valid)
    assert len(tagged) == 2 and tagged.observer_near.sum() == 1
    assert tagged.metadata["observer_cells_cleared"] is False
    with pytest.raises(ValueError, match="independent"):
        detector.detect(iq, observer={"position": normal.points_base[0], "from_iq": True})


def test_thresholds_determine_count_and_zero_input_has_no_forced_peak():
    iq = fft_impulse_iq([(20, 0, 0, 1.), (40, 2, 0, .5)])
    both = FFTDetector(rf_mapping(), dict(min_peak_fraction=.1)).detect(iq)
    stronger_only = FFTDetector(rf_mapping(), dict(min_peak_fraction=.3)).detect(iq)
    assert len(both) == 2 and len(stronger_only) == 1
    empty = FFTDetector(rf_mapping()).detect(np.zeros((40, 128, 16, 16), np.complex64))
    assert empty.points_base.shape == (0, 3)
    assert empty.valid.dtype == np.bool_ and empty.observer_near.shape == (0,)
    assert set(inspect.signature(FFTDetector.detect).parameters) == {"self", "iq", "station", "observer"}


def test_all_pulses_contribute_by_power_and_detection_has_no_history():
    detector = FFTDetector(rf_mapping())
    baseline = fft_impulse_iq([(25, 1, -2, 1.)])
    # Same power with alternating pulse phases must survive averaging.
    alternating = fft_impulse_iq([(25, 1, -2, 1.)], (-1.) ** np.arange(40))
    np.testing.assert_allclose(detector.power_volume(alternating), detector.power_volume(baseline), rtol=0, atol=0)
    gains = np.r_[np.sqrt(40.), np.zeros(39)]
    concentrated = detector.detect(fft_impulse_iq([(25, 1, -2, 1.)], gains))
    original = detector.detect(baseline)
    np.testing.assert_allclose(concentrated.metadata["peak_power"], original.metadata["peak_power"], rtol=2e-6)
    detector.detect(fft_impulse_iq([(50, -2, 2, 3.)]))
    repeated = detector.detect(baseline)
    np.testing.assert_array_equal(repeated.points_base, original.points_base)
    np.testing.assert_array_equal(repeated.scores, original.scores)


def test_input_shape_station_and_settings_fail_explicitly():
    detector = FFTDetector(rf_mapping())
    with pytest.raises(ValueError, match="complex native IQ"):
        detector.detect(np.zeros((40, 128, 16, 16)))
    with pytest.raises(ValueError, match="station"):
        detector.detect(np.zeros((40, 128, 16, 16), np.complex64), station=[1., 0., 0.])
    with pytest.raises(ValueError, match="Unknown"):
        RadarSettings.from_mapping({"target_count": 4})
    with pytest.raises(ValueError):
        RadarSettings(spatial_oversampling=True)
