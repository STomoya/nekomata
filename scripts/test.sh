#!/bin/bash
set -e

# Use the first script argument as the versions, or default to 3.12 -> 3.14
PYTHON_VERSIONS=${1:-"3.12 3.13 3.14"}
IMAGE_NAME="uv-test-matrix"

# 1. Ensure pyproject.toml exists
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: pyproject.toml not found in the current directory."
    exit 1
fi

# 2. Extract the minimum required Python version
MIN_PYTHON_VERSION=$(grep -m 1 -E '^requires-python' pyproject.toml | grep -oE '[0-9]+\.[0-9]+' | head -1)

# Fallback in case the field is missing or unparsable
if [ -z "$MIN_PYTHON_VERSION" ]; then
    echo "⚠️  Warning: Could not parse 'requires-python' from pyproject.toml. Assuming minimum is 3.11."
    MIN_PYTHON_VERSION="3.11"
fi

# Extract the minimum minor version integer (e.g., '11' from '3.11')
MIN_MINOR=$(echo "$MIN_PYTHON_VERSION" | cut -d. -f2)

echo "🔍 Validating Python versions (Minimum required: >=$MIN_PYTHON_VERSION)..."

# 3. Check that no requested version is older than the minimum
for py in $PYTHON_VERSIONS; do
    # Extract the minor version number of the target version (e.g., '10' from '3.10')
    target_minor=$(echo "$py" | cut -d. -f2)

    if [ -z "$target_minor" ] || [ "$target_minor" -lt "$MIN_MINOR" ]; then
        echo "❌ Error: Unsupported Python version '$py' detected."
        echo "   Your pyproject.toml requires Python >=$MIN_PYTHON_VERSION."
        echo "   Please run the script with valid versions (e.g., ./build_and_test.sh \"$MIN_PYTHON_VERSION 3.12\")"
        exit 1
    fi
done

echo "✅ Versions validated: $PYTHON_VERSIONS"
echo "----------------------------------------"
echo "🔨 Building Docker image '$IMAGE_NAME'..."

# Build the image, passing the dynamic versions as a build arg
docker build \
    --file docker/test.Containerfile \
    --build-arg PYTHON_VERSIONS="$PYTHON_VERSIONS" \
    -t "$IMAGE_NAME" .

echo ""
echo "✅ Build complete. Running tests..."
echo "----------------------------------------"

# Run the container (which automatically executes your run_tests.sh script)
# --rm ensures the container is cleaned up after the tests finish
docker run --rm "$IMAGE_NAME"
