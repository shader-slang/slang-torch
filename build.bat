python3 -m build
pip install -e .
cd tests/
python -m unittest discover
cd ../