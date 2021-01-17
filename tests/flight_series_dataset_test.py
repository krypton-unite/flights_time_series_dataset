import pytest
from flights_time_series_dataset import FlightSeriesDataset, flights_dataset
from .fixtures import expected_results

@pytest.mark.usefixtures('expected_results')
@pytest.mark.parametrize('augmentation', [0])
def test_series_dataset_result(augmentation, expected_results):
    exp = expected_results(augmentation)
    flight_series_dataset = FlightSeriesDataset(36, 12, 36, stride = 1, augmentation=augmentation, generate_test_dataset=True)
    assert (flight_series_dataset.x == exp['x']).all()
    assert (flight_series_dataset.y == exp['y']).all()
    assert (flight_series_dataset.test.x == exp['test_x']).all()
    assert (flight_series_dataset.test.y == exp['test_y']).all()