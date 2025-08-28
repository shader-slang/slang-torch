mkdir -p ./tmp
mkdir -p ./tmp/win64
mkdir -p ./tmp/linux64
echo "extracting $WIN64ZIP"
unzip -n $WIN64ZIP -d ./tmp/win64
echo "extracting $LINUX64ZIP"
unzip -n $LINUX64ZIP -d ./tmp/linux64

mkdir -p ./slangtorch/bin/
cp ./tmp/win64/bin/slang.dll ./slangtorch/bin/slang.dll
cp ./tmp/win64/bin/slang-glsl-module.dll ./slangtorch/bin/slang-glsl-module.dll
cp ./tmp/win64/bin/slangc.exe ./slangtorch/bin/slangc.exe
cp ./tmp/linux64/lib/libslang.so ./slangtorch/bin/libslang.so
cp ./tmp/linux64/lib/libslang-glsl-module.so ./slangtorch/bin/libslang-glsl-module.so
cp ./tmp/linux64/bin/slangc ./slangtorch/bin/slangc
chmod +x ./slangtorch/bin/slangc

echo "content of ./slangtorch/bin/:"
ls ./slangtorch/bin/

rm $WIN64ZIP
rm $LINUX64ZIP
rm -rf ./tmp/

python3 --version

python -m pip install --upgrade pip
pip install build hatchling

python -m build