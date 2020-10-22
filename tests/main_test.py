"""
main_test.py
"""

from flights_time_series_dataset import FlightsDataset

def test_dataset():
    flights_dataset = FlightsDataset()
    assert flights_dataset.get_y_shape() == (1, 144, 1)
    assert flights_dataset.get_x_shape() == (1, 144, 2)
    fd = flights_dataset.make_future_dataframe(12)
    assert fd.shape == (156, 2)
