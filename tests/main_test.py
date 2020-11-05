"""
main_test.py
"""

from flights_time_series_dataset import FlightsDataset, convert_year_month_array_to_datetime
from datetime import datetime

def test_dataset_shape():
    flights_dataset = FlightsDataset()
    assert flights_dataset.get_y_shape() == (1, 144, 1)
    assert flights_dataset.get_x_shape() == (1, 144, 2)
    fd = flights_dataset.make_future_dataframe(12)
    assert fd.shape == (156, 2)

def test_conversion_to_datetime():
    flights_dataset = FlightsDataset()
    dt = convert_year_month_array_to_datetime(flights_dataset.x[0,0,:])
    assert dt == datetime(1949, 1, 15, 0, 0)