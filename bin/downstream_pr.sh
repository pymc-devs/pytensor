#!/bin/bash

if [ -z "$1" ]
  then
    echo "Usage: downstream_pr.sh <PR_number>"
    echo "Will create a new branch, apply changes, and create a new PR"
    exit 1
fi

set -e

git checkout main
git pull origin main
git checkout -b downstream_$1

echo "Downloading patch..."
wget -O $1.patch https://patch-diff.githubusercontent.com/raw/aesara-devs/aesara/pull/$1.patch

echo "Replacing aesara strings..."
declare -a replace_strings=(
    "s/aesara/pytensor/g"
    "s/Aesara/PyTensor/g"
#    "s/import pytensor.tensor as at/import pytensor.tensor as pt/g"
    "s/at\./pt./g"
#    "s/from pytensor import tensor as pt/from pytensor import tensor as pt/g"
)

for replace in "${replace_strings[@]}"; do
    find . -name "*$1.patch" -type f -exec sed -i -e "/png/!$replace" {} \;
done

echo "Applying patch..."
git am -3 --reject $1.patch

echo "Running pre-commit"
pre-commit run --all

git push origin downstream_$1

gh pr create --repo pymc-devs/pytensor --title "Downstreaming $1" --body "Downstreaming https://github.com/aesara-devs/aesara/pull/$1"
