#!/bin/bash
set -e

PYTHON=""

function check_python() {
  if ! which python >/dev/null || ! which python3 >/dev/null; then
    return 1
  fi
  if python --version | grep -E '3.8|3.9|3.10|3.11'
  then
    PYTHON="python"
    return 0
  else
    if python3 --version | grep -E '3.8|3.9|3.10|3.11'
    then
      PYTHON="python3"
      return 0
    else
      return 1
    fi
  fi
}

echo -e "===\nThis script will automatically setup quint inside a pipenv\n===\n"
echo "[] Checking python..."
if ! check_python; then
  echo "[!] Suitable Python version not found"
  exit 1
else
  echo "[] Found suitable python: $(which $PYTHON) = $($PYTHON --version)"
fi

echo "[] Installing pipenv..."
$PYTHON -m pip install --user pipenv

echo "[] Cloning quint..."
git clone https://github.com/V3ntus/quint
cd quint

echo "[] Setting up venv with pipenv..."
$PYTHON -m pipenv install

echo "[] Starting the quint API..."
$PYTHON -m pipenv run python "$(pwd)/scripts/serve_api.py"
