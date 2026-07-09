set -euo pipefail

case "${PACKAGE_PLATFORM:-}" in
    windows-x86_64)
        PACKAGE_ZIP="$WIN64ZIP"
        PACKAGE_TMP="./tmp/win64"
        WHEEL_TAG="py3-none-win_amd64"
        ;;
    linux-x86_64)
        PACKAGE_ZIP="$LINUX64ZIP"
        PACKAGE_TMP="./tmp/linux64"
        WHEEL_TAG="py3-none-manylinux_2_27_x86_64"
        ;;
    linux-aarch64)
        PACKAGE_ZIP="$LINUXAARCH64ZIP"
        PACKAGE_TMP="./tmp/linux-aarch64"
        WHEEL_TAG="py3-none-manylinux_2_28_aarch64"
        ;;
    *)
        echo "Error: PACKAGE_PLATFORM must be one of: windows-x86_64, linux-x86_64, linux-aarch64"
        exit 1
        ;;
esac

rm -rf ./tmp ./slangtorch/bin
mkdir -p "$PACKAGE_TMP" ./slangtorch/bin/
echo "extracting $PACKAGE_ZIP"
unzip -n "$PACKAGE_ZIP" -d "$PACKAGE_TMP"

if [ "$PACKAGE_PLATFORM" = "windows-x86_64" ]; then
    if [ -e "$PACKAGE_TMP/bin/slang-compiler.dll" ]; then
        cp "$PACKAGE_TMP/bin/slang-compiler.dll" ./slangtorch/bin/
    else
        cp "$PACKAGE_TMP/bin/slang.dll" ./slangtorch/bin/
    fi
    cp "$PACKAGE_TMP/bin/slang-glsl-module.dll" ./slangtorch/bin/
    cp "$PACKAGE_TMP/bin/slangc.exe" ./slangtorch/bin/
else
    if [ -e "$PACKAGE_TMP/lib/libslang-compiler.so" ]; then
        cp "$(realpath "$PACKAGE_TMP/lib/libslang-compiler.so")" ./slangtorch/bin/
    else
        cp "$PACKAGE_TMP/lib/libslang.so" ./slangtorch/bin/
    fi
    cp "$PACKAGE_TMP"/lib/libslang-glsl-module*.so ./slangtorch/bin/
    cp "$PACKAGE_TMP/bin/slangc" ./slangtorch/bin/slangc
    chmod +x ./slangtorch/bin/slangc
fi

echo "content of ./slangtorch/bin/:"
ls ./slangtorch/bin/

rm "$PACKAGE_ZIP"
rm -rf ./tmp/

# Detect Python command - check if python has pip module
PYTHON_CMD=""
for cmd in python3 python py; do
    if command -v $cmd &> /dev/null; then
        if $cmd -m pip --version &> /dev/null; then
            PYTHON_CMD=$cmd
            echo "Found Python with pip: $cmd"
            break
        else
            echo "Warning: $cmd found but no pip module available"
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "Error: Python with pip not found. Please install Python with pip or add it to PATH."
    exit 1
fi

echo "Using Python command: $PYTHON_CMD"
$PYTHON_CMD --version

$PYTHON_CMD -m pip install --upgrade pip
$PYTHON_CMD -m pip install build hatchling

SLANGTORCH_WHEEL_TAG="$WHEEL_TAG" $PYTHON_CMD -m build --wheel
