#!/bin/bash

# Get the latest release tag from GitHub API
LATEST_TAG=$(curl -s https://api.github.com/repos/shader-slang/slang/releases/latest | grep '"tag_name":' | sed -E 's/.*"([^"]+)".*/\1/')

echo "Latest Slang release: $LATEST_TAG"

# Remove 'v' prefix from tag to get version number
VERSION=${LATEST_TAG#v}

# Download Linux x86_64 build
echo "Downloading slang-${VERSION}-linux-x86_64-glibc-2.27.zip..."
curl -L -o slang-${VERSION}-linux-x86_64-glibc-2.27.zip \
  https://github.com/shader-slang/slang/releases/download/${LATEST_TAG}/slang-${VERSION}-linux-x86_64-glibc-2.27.zip

# Download Windows x86_64 build
echo "Downloading slang-${VERSION}-windows-x86_64.zip..."
curl -L -o slang-${VERSION}-windows-x86_64.zip \
  https://github.com/shader-slang/slang/releases/download/${LATEST_TAG}/slang-${VERSION}-windows-x86_64.zip

# Set environment variables
export LINUX64ZIP="slang-${VERSION}-linux-x86_64-glibc-2.27.zip"
export WIN64ZIP="slang-${VERSION}-windows-x86_64.zip"

echo "LINUX64ZIP: $LINUX64ZIP"
echo "WIN64ZIP: $WIN64ZIP"

# Run the build script
echo "Running build-package.sh..."
bash build-package.sh
