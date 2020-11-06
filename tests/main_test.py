"""
main_test.py
"""

from flights_time_series_dataset import FlightsDataset, convert_year_month_array_to_datetime
from datetime import datetime
import numpy as np


def test_dataset_shape():
    flights_dataset = FlightsDataset()
    assert flights_dataset.get_y_shape() == (1, 144, 1)
    assert flights_dataset.get_x_shape() == (1, 144, 2)
    fd = flights_dataset.make_future_dataframe(12)
    assert fd.shape == (1, 156, 2)


def test_conversion_to_datetime():
    flights_dataset = FlightsDataset()
    first_date = flights_dataset.x[0, 0, :]
    dt = convert_year_month_array_to_datetime(first_date)
    assert dt == datetime(1949, 1, 15)


def test_vector_conversion_to_datetime():
    flights_dataset = FlightsDataset()
    dt_vector = convert_year_month_array_to_datetime(
        flights_dataset.x[0, 0:5, :])
    expected_output_vector = [
        datetime(1949, 1, 15),
        datetime(1949, 2, 15),
        datetime(1949, 3, 15),
        datetime(1949, 4, 15),
        datetime(1949, 5, 15),
    ]
    assert np.all(expected_output_vector == dt_vector)
