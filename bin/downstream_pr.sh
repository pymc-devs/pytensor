#!/bin/bash

if [ -z "$1" ]
  then
    echo "Usage: downstream_pr.sh <PR_number>"
    echo "Port a specified PR from the Aesara repo to the PyTensor repo. Will create a new branch, adapt and apply the upstream PR, and create a new PR to PyTensor."
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
#    "s/at\./pt./g"
#    "s/from pytensor import tensor as pt/from pytensor import tensor as pt/g"
)

for replace in "${replace_strings[@]}"; do
    sed -i -e "$replace" *$1.patch
done

echo "Applying patch..."
if git am -3 --reject $1.patch ; then
    echo "Patch applied successfully..."
else
    echo "Patch failed. Find the .rej file and apply the changes manually. Then 'git add' all changed files, followed by 'git am --continue'. Then create a PR manually."
    exit 1
fi

echo "Running pre-commit"
pre-commit run --all

git push origin downstream_$1

gh pr create --repo pymc-devs/pytensor --title "Downstreaming Aesara PR $1" --body "Downstreaming https://github.com/aesara-devs/aesara/pull/$1. PR port done by downstream_pr.sh script."
