.PHONY: install test benchmark clean

install:
	pip install -r requirements.txt
	pip install -e .

test:
	python -m pytest tests/ -v

benchmark:
	python benchmarks/run_benchmark.py

clean:
	rm -rf __pycache__ .pytest_cache build dist *.egg-info
