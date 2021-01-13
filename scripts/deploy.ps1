python setup.py sdist bdist_wheel
$version="1.1.4"
$files_to_handle_str="dist/flights_time_series_dataset-$version*" 
twine check $files_to_handle_str
twine upload $files_to_handle_str