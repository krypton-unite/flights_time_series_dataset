import pytest
from flights_time_series_dataset import FlightSeriesDataset
from .fixtures import expected_shape

@pytest.mark.usefixtures('expected_shape')
@pytest.mark.parametrize('except_last', [1, 36])
@pytest.mark.parametrize('augmentation', [0, 1])
def test_series_dataset_shape(except_last, augmentation, expected_shape):
    exp = expected_shape(except_last, augmentation)
    flight_series_dataset = FlightSeriesDataset(36, 12, except_last, stride = exp['stride'], augmentation=augmentation)
    assert flight_series_dataset.get_y_shape() == exp['y_shape']
    assert flight_series_dataset.get_x_shape() == exp['x_shape']
    assert flight_series_dataset.y_test.shape == exp['y_test_shape']
    assert flight_series_dataset.X_test.shape == exp['x_test_shape']