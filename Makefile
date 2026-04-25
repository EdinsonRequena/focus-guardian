install:
	pipenv install

install-dev:
	pipenv install --dev pylint autopep8

run:
	pipenv run python src/main.py

lint:
	pipenv run pylint src

format:
	pipenv run autopep8 --in-place --recursive src

check:
	pipenv run python -m compileall src
	pipenv run pylint src
