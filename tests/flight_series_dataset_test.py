import pytest
from flights_time_series_dataset import FlightSeriesDataset
from .fixtures import expected_shape

@pytest.mark.usefixtures('expected_shape')
@pytest.mark.parametrize('augmentation', [0])
def test_series_dataset_shape(augmentation, expected_shape):
    exp = expected_shape(augmentation)
    flight_series_dataset = FlightSeriesDataset(36, 12, 36, stride = 1, augmentation=augmentation)
    assert flight_series_dataset.get_y_shape() == exp['y_shape']
    assert flight_series_dataset.get_x_shape() == exp['x_shape']
    assert flight_series_dataset.test.y.shape == exp['y_test_shape']
    assert flight_series_dataset.test.x.shape == exp['x_test_shape']