
install_package:
	@pip install -e .

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr quint-*.dist-info
	@rm -fr quint.egg-info
	@find . -name "__pycache__" -type d -exec rm -r {} +
	@find . -name "*.pyc" -exec rm -f {} +


run_api:
	uvicorn quint.api.fast:app --port 8083 --reload
