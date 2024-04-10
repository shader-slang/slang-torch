mkdir -p ./tmp
echo "extracting $WIN64ZIP"
unzip -n $WIN64ZIP -d ./tmp
echo "extracting $LINUX64ZIP"
unzip -n $LINUX64ZIP -d ./tmp

mkdir -p ./slangtorch/bin/
cp ./tmp/bin/windows-x64/release/slang.dll ./slangtorch/bin/slang.dll
cp ./tmp/bin/windows-x64/release/slangc.exe ./slangtorch/bin/slangc.exe
cp ./tmp/bin/linux-x64/release/libslang.so ./slangtorch/bin/libslang.so
cp ./tmp/bin/linux-x64/release/slangc ./slangtorch/bin/slangc
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