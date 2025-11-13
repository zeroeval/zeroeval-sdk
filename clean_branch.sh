#!/bin/bash

set -e  # Exit on any error

echo "=================================="
echo "Git Branch Cleanup Script"
echo "=================================="
echo ""
echo "This will:"
echo "  1. Create a backup of your current branch"
echo "  2. Create a fresh improved-ab-testing branch"
echo "  3. Copy all your changes without the problematic commit history"
echo "  4. Force push the clean branch"
echo ""
read -p "Continue? (y/n): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

echo ""
echo "Step 1: Creating backup branch..."
git branch improved-ab-testing-backup 2>/dev/null || echo "Backup already exists, continuing..."

echo ""
echo "Step 2: Switching to main branch..."
git checkout main
git pull origin main

echo ""
echo "Step 3: Deleting old improved-ab-testing branch..."
git branch -D improved-ab-testing

echo ""
echo "Step 4: Creating fresh improved-ab-testing branch..."
git checkout -b improved-ab-testing

echo ""
echo "Step 5: Checking what files were changed..."
echo "Files to be copied:"
git diff main..improved-ab-testing-backup --name-only

echo ""
echo "Step 6: Copying your changes from backup..."
# Copy the entire examples_v2 directory
if git show improved-ab-testing-backup:examples_v2/ &>/dev/null; then
    git checkout improved-ab-testing-backup -- examples_v2/ 2>/dev/null || echo "No examples_v2 changes"
fi

# Copy tests directory if it exists
if git show improved-ab-testing-backup:tests/ &>/dev/null; then
    git checkout improved-ab-testing-backup -- tests/ 2>/dev/null || echo "No tests changes"
fi

# Copy any other modified files from src
if git show improved-ab-testing-backup:src/ &>/dev/null; then
    git checkout improved-ab-testing-backup -- src/ 2>/dev/null || echo "No src changes"
fi

# Copy other common files
for file in README.md INTEGRATIONS.md pyproject.toml; do
    if git diff main..improved-ab-testing-backup --name-only | grep -q "^$file$"; then
        git checkout improved-ab-testing-backup -- "$file" 2>/dev/null || true
    fi
done

echo ""
echo "Step 7: Creating one clean commit..."
git add .
git commit -m "Add improved A/B testing examples and features"

echo ""
echo "Step 8: Force pushing to remote..."
echo "This will overwrite the remote branch completely."
read -p "Continue with force push? (y/n): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Push aborted. Your local branch is ready but not pushed."
    echo "You can manually push later with: git push --force origin improved-ab-testing"
    exit 0
fi

git push --force origin improved-ab-testing

echo ""
echo "=================================="
echo "✅ SUCCESS!"
echo "=================================="
echo ""
echo "Your branch has been cleaned and pushed."
echo "The backup branch 'improved-ab-testing-backup' is still available locally."
echo ""
echo "To delete the backup branch later:"
echo "  git branch -D improved-ab-testing-backup"
echo ""

