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

mkdir -p ./tmp
mkdir -p ./tmp/win64
mkdir -p ./tmp/linux64
echo "extracting $WIN64ZIP"
unzip -n $WIN64ZIP -d ./tmp/win64
echo "extracting $LINUX64ZIP"
unzip -n $LINUX64ZIP -d ./tmp/linux64

mkdir -p ./slangtorch/bin/
if [ -e "./tmp/win64/bin/slang-compiler.dll" ]; then
    mv ./tmp/win64/bin/slang-compiler.dll ./slangtorch/bin/
else
    mv ./tmp/win64/bin/slang.dll ./slangtorch/bin/
fi
mv ./tmp/win64/bin/slang-glsl-module.dll ./slangtorch/bin/
mv ./tmp/win64/bin/slangc.exe ./slangtorch/bin/
if [ -e "./tmp/linux64/lib/libslang-compiler.so" ]; then
    mv `realpath ./tmp/linux64/lib/libslang-compiler.so` ./slangtorch/bin/
else
    mv ./tmp/linux64/lib/libslang.so ./slangtorch/bin/
fi
mv ./tmp/linux64/lib/libslang-glsl-module*.so ./slangtorch/bin/
mv ./tmp/linux64/bin/slangc ./slangtorch/bin/slangc
chmod +x ./slangtorch/bin/slangc

echo "content of ./slangtorch/bin/:"
ls ./slangtorch/bin/

rm $WIN64ZIP
rm $LINUX64ZIP
rm -rf ./tmp/

python -m build