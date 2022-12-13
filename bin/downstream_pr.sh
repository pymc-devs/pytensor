#!/bin/bash

if [ -z "$1" ]
  then
    echo "Usage: downstream_pr.sh <PR_number>"
    echo "Port a specified PR from the Aesara repo to the PyTensor repo. Will create a new branch, adapt and apply the upstream PR, and create a new PR to PyTensor. Requires a remote aesara."
    exit 1
fi

git am --abort
git checkout main
git branch -D downstream_$1
git fetch aesara

set -e
git pull origin main
git checkout -b downstream_$1

echo "Downloading patch..."
wget -O $1.patch https://patch-diff.githubusercontent.com/raw/aesara-devs/aesara/pull/$1.patch

echo "Replacing aesara strings..."
declare -a replace_strings=(
    "s/aesara/pytensor/g"
    "s/Aesara/PyTensor/g"
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
# get the informative title
title=$(curl https://api.github.com/repos/aesara-devs/aesara/pulls/$1 2>/dev/null | jq '.title')
gh pr create --repo pymc-devs/pytensor --label "aesara downstream" --title "🔄 From Aesara: $1: $title" --body "Downstreaming https://github.com/aesara-devs/aesara/pull/$1. PR port done by downstream_pr.sh script."
