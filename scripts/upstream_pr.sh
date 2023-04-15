#!/bin/bash

if [ -z "$1" ]
  then
    echo "Usage: upstream_pr.sh <PR_number>"
    echo "Port a specified PR from the PyTensor repo to the Aesara repo. Will create a new branch, adapt and apply the upstream PR, and create a new PR to PyTensor."
    exit 1
fi

set -e

git checkout main
git pull origin main
git checkout -b downstream_$1

echo "Downloading patch..."
wget -O $1.patch https://patch-diff.githubusercontent.com/raw/pymc-devs/pytensor/pull/$1.patch

echo "Replacing aesara strings..."
declare -a replace_strings=(
    "s/pytensor/aesara/g"
    "s/PyTensor/Aesara/g"
)

for replace in "${replace_strings[@]}"; do
    sed -i -e "$replace" $1.patch
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

title=$(curl https://api.github.com/repos/pymc-devs/pytensor/pulls/$1 2>/dev/null | jq '.title')
gh pr create --repo aesara-devs/aesara --title "Upstreaming PyTensor PR $1: $title" --body "Upstreaming https://github.com/pymc-devs/pytensor/pull/$1. PR port done by upstream_pr.sh script."
