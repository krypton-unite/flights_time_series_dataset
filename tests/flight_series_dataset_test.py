from flights_time_series_dataset import FlightSeriesDataset


def test_series_dataset_shape():
    flight_series_dataset = FlightSeriesDataset(36, 12, 36)
    assert flight_series_dataset.get_y_shape() == (49, 24, 1)
    assert flight_series_dataset.get_x_shape() == (49, 24, 2)
    assert flight_series_dataset.y_test.shape == (36, 24, 1)
    assert flight_series_dataset.X_test.shape == (36, 24, 2)

def test_series_dataset_shape_with_augmentation():
    flight_series_dataset = FlightSeriesDataset(36, 12, 36, data_augmentation=1)
    assert flight_series_dataset.get_y_shape() == (71, 24, 1)
    assert flight_series_dataset.get_x_shape() == (71, 24, 2)
    assert flight_series_dataset.y_test.shape == (36, 24, 1)
    assert flight_series_dataset.X_test.shape == (36, 24, 2)