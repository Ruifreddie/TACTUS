#!/usr/bin/env bash
# Stands in for the recipes' python: `pyshim.sh <script> <args>` = `$ABLATE_REAL_PY launch.py <script> <args>`.
# run.sh sets PY (UniTAP recipes, via pipeline/recipe.sh) or PYTHON (classification and interval recipes) to this
# file, so the recipe command lines stay unchanged; the switches are environment variables (see launch.py).
exec "${ABLATE_REAL_PY:-python}" "$(dirname "$(readlink -f "$0")")/launch.py" "$@"
