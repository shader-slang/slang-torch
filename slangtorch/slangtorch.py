import os
import sys
import pkg_resources
import subprocess
import glob
import hashlib
import json
import re
import time
from filelock import FileLock

from .util import jit_compile, run_ninja, NinjaResult
from .util import wrapModule

packageDir = pkg_resources.resource_filename(__name__, '')
versionCode = my_version = pkg_resources.get_distribution('slangtorch').version

DEFAULT_CUDA_CFLAGS = ["-U__CUDA_NO_HALF_OPERATORS__",
                       "-U__CUDA_NO_HALF_CONVERSIONS__",
                       "-U__CUDA_NO_HALF2_OPERATORS__",
                       "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                       "-DSLANG_CUDA_ENABLE_HALF=1"]

if sys.platform == "win32":
    # Windows
    executable_extension = ".exe"

elif sys.platform == "darwin":
    # macOS
    executable_extension = ""
else:
    # Linux and other Unix-like systems
    executable_extension = ""

slangcPath = os.path.join(
    packageDir, 'bin', 'slangc'+executable_extension)

# If we have SLANGC_PATH set, use that instead
if 'SLANGC_PATH' in os.environ:
    slangcPath = os.environ['SLANGC_PATH']

# Ensure that slangcPath is a proper path
slangcPath = os.path.realpath(slangcPath)

if not os.path.exists(slangcPath):
    raise RuntimeError(f"Could not find slangc executable at {slangcPath}")

# Mapping from module key to latest version number. Used to create unique build directories.
MODULE_VERSIONS = {}

# Mapping from module key to set of loaded build directories. Used to avoid re-using build directories
# whose binaries are in use (compilation will fail)
#
LOADED_BUILD_DIRS = {}

def getUniqueSessionVersion(moduleKey):
    if moduleKey not in MODULE_VERSIONS:
        MODULE_VERSIONS[moduleKey] = 0
    else:
        MODULE_VERSIONS[moduleKey] += 1
    
    return MODULE_VERSIONS[moduleKey]

def getCurrentSessionVersion(moduleKey):
    if moduleKey not in MODULE_VERSIONS:
        MODULE_VERSIONS[moduleKey] = 0
    
    return MODULE_VERSIONS[moduleKey]

def addLoadedDirectoryEntry(moduleKey, version):
    if moduleKey not in LOADED_BUILD_DIRS:
        LOADED_BUILD_DIRS[moduleKey] = set()
    
    LOADED_BUILD_DIRS[moduleKey].add(version)

def isDirectoryInUse(moduleKey, version):
    if moduleKey not in LOADED_BUILD_DIRS:
        return False
    
    return version in LOADED_BUILD_DIRS[moduleKey]

def _replaceFileExt(fileName, newExt, suffix=None):
    baseName, old_extension = os.path.splitext(fileName)
    if suffix:
        new_filename = baseName + suffix + newExt
    else:
        new_filename = baseName + newExt
    return new_filename


def find_cl():
    # Look for cl.exe in the default installation path for Visual Studio
    vswhere_path = os.environ.get('ProgramFiles(x86)', '') + '\\Microsoft Visual Studio\\Installer\\vswhere.exe'

    # Get the installation path of the latest version of Visual Studio
    result = subprocess.run([vswhere_path, '-latest', '-products', '*', '-property', 'installationPath'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    vs_install_path = result.stdout.decode('utf-8').rstrip()

    # Find the path to cl.exe
    cl_path = glob.glob(os.path.join(vs_install_path, "**", "VC", "Tools", "MSVC", "**", "bin", "HostX64", "X64"), recursive=True)

    if not cl_path:
        raise ValueError("cl.exe not found in default Visual Studio installation path")

    # Get the latest version of cl.exe
    cl_path.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return cl_path[0]


def _add_msvc_to_env_var():
    if sys.platform == 'win32':
        path_to_add = find_cl()
        if path_to_add not in os.environ["PATH"].split(os.pathsep):
            os.environ["PATH"] += os.pathsep + path_to_add


def tryGetSlangDynamicLibraryPath():
    # Search in the slangcPath directory for the slang dynamic library.
    slangcDir = os.path.dirname(slangcPath)
    slangcDir = os.path.realpath(slangcDir)

    if sys.platform == "win32":
        # Windows
        slangLibPath = os.path.join(slangcDir, "slang.dll")
    elif sys.platform == "darwin":
        # macOS
        slangLibPath = os.path.join(slangcDir, "libslang.dylib")
    elif (sys.platform == "" or sys.platform == "linux"):
        # Linux
        slangLibPath = os.path.join(slangcDir, "libslang.so")
    else:
        return None

    if os.path.exists(slangLibPath):
        return slangLibPath
    else:
        return None


def getHash(obj, truncate_at=16):
    # Convert obj to JSON string
    jsonString = json.dumps(obj, sort_keys=True)

    # Compute SHA-256 hash of the JSON string
    hashObject = hashlib.sha256(jsonString.encode())
    hashCode = hashObject.hexdigest()

    # Truncate the hash code
    return hashCode[:truncate_at]


def convertNonAlphaNumericToUnderscore(name):
    converted_name = re.sub(r'\W+', '_', name)
    return converted_name


def makeOptionsList(defines):
    if defines is None:
        return []
    
    defines = dict(defines)
    return list([f"-D{key}={value}" for (key, value) in defines.items()])


def makeBuildDirPath(baseDir, buildID):
    return os.path.join(baseDir, f"{buildID}")


def getLatestDir(moduleKey, baseDir):
    latestFile = os.path.join(baseDir, "latest.txt")
    if os.path.exists(latestFile):
        with open(latestFile, 'r') as f:
            latestBuildID = int(f.read())
    else:
        return None
    
    return makeBuildDirPath(baseDir, latestBuildID)


def getOrCreateUniqueDir(moduleKey, baseDir):
    # Check if buildDir has a latest.txt file. If so, read the contents.
    # If not, create a latest.txt with '0' as the contents.
    #
    latestFile = os.path.join(baseDir, "latest.txt")
    if os.path.exists(latestFile):
        with open(latestFile, 'r') as f:
            latestBuildID = int(f.read())
    else:
        latestBuildID = None

    targetBuildID = getUniqueSessionVersion(moduleKey)

    while (isDirectoryInUse(moduleKey, makeBuildDirPath(baseDir, targetBuildID))):
        # If the build directory is in use, we need to create a new build directory.
        targetBuildID = getUniqueSessionVersion(moduleKey)
    
    targetDir = None
    if (latestBuildID is None) or targetBuildID == latestBuildID:
        # If latestBuildID is None, then we are building the first version of this module.
        # If targetBuildID is the same as latestBuildID, the contents will simply
        # be reused by the ninja build system.
        # 
        dirname = makeBuildDirPath(baseDir, targetBuildID)

        # Create the directory if it doesn't exist (this happens if latestBuildID is None)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        targetDir = dirname
    else:
        # Copy all contents of the latestBuildID folder to the targetBuildID folder.
        # including metadata!
        #
        latestDir = makeBuildDirPath(baseDir, latestBuildID)
        targetDir = makeBuildDirPath(baseDir, targetBuildID)

        if os.path.exists(targetDir):
            import shutil
            shutil.rmtree(targetDir)

        import distutils.dir_util
        distutils.dir_util.copy_tree(latestDir, targetDir)

        # Modify the build-dir-sensitive metadata in metadata.json
        metadataFile = os.path.join(targetDir, "metadata.json")
        if (os.path.exists(metadataFile) and os.path.isfile(metadataFile)):
            with open(metadataFile, 'r') as f:
                metadata = json.load(f)
                metadata['moduleBinary'] = os.path.join(targetDir, f"{metadata['moduleName']}.pyd")

            with open(metadataFile, 'w') as f:
                json.dump(metadata, f, indent=4)
    
    # Update latest.txt
    with open(latestFile, 'w') as f:
        f.write(str(targetBuildID))
    
    return targetDir


def compileSlang(metadata, fileName, targetMode, options, outputFile, verbose=False, includePaths=[], dryRun=False, extraSlangFlags=[]):
    needsRecompile = False

    # If version either doesn't exist or is different, we need to recompile.
    if metadata and metadata.get("version", None):
        oldVersion = metadata["version"]
        if verbose:
            print("Checking slangtorch version... ", file=sys.stderr)
        if oldVersion != versionCode:
            if verbose:
                print(f"Version is different \"{oldVersion}\" => \"{versionCode}\". Needs recompile.", file=sys.stderr)
            needsRecompile = True
    else:
        if verbose:
            print("Version number missing. Needs recompile ", file=sys.stderr)
        needsRecompile = True

    # If any of the depfiles are newer than outputFile, we need to recompile.
    if metadata and metadata.get("deps"):
        depFiles = metadata["deps"]
        if not os.path.exists(outputFile):
            if verbose:
                print(f"Output file {outputFile} does not exist. Needs recompile.", file=sys.stderr)
            needsRecompile = True
        else:
            for depFile, timestamp in depFiles:
                if verbose:
                    print(f"Checking dependency: {depFile}", file=sys.stderr)
                
                if not os.path.exists(depFile):
                    if verbose:
                        print(f"\tDependency does not exist. Needs recompile.", file=sys.stderr)
                    needsRecompile = True
                    break
                
                if os.path.getmtime(depFile) > timestamp:
                    if verbose:
                        print(f"\tDependency is newer. Needs recompile.", file=sys.stderr)
                    needsRecompile = True
                    break
    else:
        needsRecompile = True

    # If any of the options are different, we need to recompile.
    if metadata and (metadata.get("options", None) != None):
        oldOptions = metadata["options"]
        if verbose:
            print("Checking options... ", file=sys.stderr)
        if oldOptions != options:
            if verbose:
                print(f"Options are different \"{oldOptions}\" => \"{options}\". Needs recompile.", file=sys.stderr)
            needsRecompile = True
    else:
        needsRecompile = True

    # If any of the include paths are different, we need to recompile (include paths can 
    # potentially change which version of a header is imported)
    #
    if metadata and (metadata.get("includePaths", None) != None):
        oldIncludePaths = metadata["includePaths"]
        if verbose:
            print("Checking include paths... ", file=sys.stderr)
        
        # Check that the two lists have the same elements.
        if set(oldIncludePaths) != set(includePaths):
            if verbose:
                print(f"Include paths are different \"{oldIncludePaths}\" => \"{includePaths}\". Needs recompile.", file=sys.stderr)
            needsRecompile = True
    else:
        needsRecompile = True
    
    if needsRecompile:
        return True, (_compileSlang(metadata, fileName, targetMode, options, outputFile, includePaths, verbose, extraSlangFlags) if not dryRun else None)
    else:
        return False, (metadata if not dryRun else None)


def _compileSlang(metadata, fileName, targetMode, options, outputFile, includePaths=[], verbose=False, extraSlangFlags=[]):
    # Create a temporary depfile path.
    depFile = f"{outputFile}.d.out"

    compileCommand = [slangcPath, fileName, *options, 
                      '-target', targetMode,
                      '-o', outputFile,
                      '-depfile', depFile,
                      '-ignore-capabilities']
    compileCommand.extend(extraSlangFlags)

    if includePaths is not None:
        for includePath in includePaths:
            compileCommand.extend(["-I", includePath])

    if verbose:
        print(f"Building {os.path.basename(fileName)} -> {os.path.basename(outputFile)}: ", 
              " ".join(compileCommand), file=sys.stderr)

    result = subprocess.run(compileCommand, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    slangcErr = result.stderr.decode('utf-8')
    if slangcErr.strip():
        print(slangcErr, file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Compilation failed with error {result.returncode}")
    
    deps = parseDepfile(depFile)

    # Add slangc executable & dynamic library location & mtime to the dependency list.
    deps.append((slangcPath, os.path.getmtime(slangcPath)))

    slangLibPath = tryGetSlangDynamicLibraryPath()
    if slangLibPath is not None:
        deps.append((slangLibPath, os.path.getmtime(slangLibPath)))

    # Erase depfile.
    os.remove(depFile)

    # Update metadata.
    return {"options": options, "deps": deps, "version": versionCode, "includePaths": includePaths}


def compileAndLoadModule(metadata, sources, moduleName, buildDir, slangSourceDir=None, verbose=False, dryRun=False, skipNinjaCheck=False, extraCudaFlags=[], extraSyclFlags=[]):
    needsRebuild = False
    needsReload = False

    newMetadata = metadata.copy()

    # Check if any of the sources are newer than the module binary.
    if metadata and metadata.get("moduleBinary", None):
        moduleBinary = os.path.realpath(metadata["moduleBinary"])
        if os.path.exists(moduleBinary):
            for source in sources:
                if verbose:
                    print("Checking dependency: ", source, file=sys.stderr)
                    
                if not os.path.exists(source):
                    raise RuntimeError(f"Dependency {source} does not exist")

                if os.path.getmtime(source) > os.path.getmtime(moduleBinary):
                    if verbose:
                        print("Dependency is newer than module binary. Rebuilding.", file=sys.stderr)
                    needsRebuild = True
                    break
        else:
            needsRebuild = True
    else:
        needsRebuild = True

    if not needsRebuild and not skipNinjaCheck:
        # One more check: we will run ninja on the build directory to see if there is anything to do.
        # This check catches the case where the Slang products are up-to-date, but any downstream 
        # dependencies such as prelude header files, or user-defined header files have changed.
        #
        if verbose:
            print(f"Running ninja in existing build-dir: {buildDir}", file=sys.stderr)

        # verbose set to False on purpose to avoid clogging up the console with the trial run
        # which is expected to show failure messages even on certain successful states.
        #
        ninja_result = run_ninja(buildDir, verbose=False)

        if ninja_result == NinjaResult.BUILD_SUCCESS:
            if verbose:
                print(f"\tBuild non-trivial success. Need to reload module", file=sys.stderr)
            needsRebuild = False
            needsReload = True
        elif ninja_result == NinjaResult.NO_WORK_TO_DO:
            if verbose:
                print(f"\tNo work to do.", file=sys.stderr)
            needsRebuild = False
            needsReload = False
        elif ninja_result == NinjaResult.BUILD_FAIL:
            if verbose:
                print(f"\tBuild failed (either the build files are improper or target is in use."
                      f"Fresh build required)", file=sys.stderr)
            needsRebuild = True
            needsReload = False
        else:
            raise RuntimeError(f"Unknown ninja result: {ninja_result}")
    else:
        if verbose:
            print(f"Skipping additional ninja check (WARNING: this may ignore changes to non-slang files)", file=sys.stderr)

    cacheLookupKey = moduleName
    if not needsRebuild:
        if not needsReload:
            # Try the session cache. If we find a hit, the module is already loaded.
            if compileAndLoadModule._moduleCache is not None:
                if cacheLookupKey in compileAndLoadModule._moduleCache:
                    if verbose:
                        print(f"Build & load skipped. Using cached module ({cacheLookupKey})", file=sys.stderr)
                    if dryRun:
                        return False, None
                    return compileAndLoadModule._moduleCache[cacheLookupKey], newMetadata
        
        # If not, try the persistent cache (load shared object). It's a lot quicker to import the binary
        # than going through torch's build-file generation + ninja dependency detection
        # even if ultimately have nothing to do.
        #
        moduleBinary = os.path.realpath(metadata["moduleBinary"])
        if os.path.exists(moduleBinary):
            if dryRun:
                return False, None
            
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    metadata["moduleName"], moduleBinary)
                slangLib = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(slangLib)
            except Exception as e:
                if verbose:
                    print(f"Failed to load existing module binary {moduleBinary}: {e}", file=sys.stderr)
                    print(f"Needs rebuild", file=sys.stderr)

                needsRebuild = True
        else:
            needsRebuild = True

    if needsRebuild:
        if dryRun:
            return True, None
        
        # Compile the module.
        slangLib = _compileAndLoadModule(metadata, sources, moduleName, buildDir, slangSourceDir, extraCudaFlags, extraSyclFlags, verbose)

        newMetadata = metadata.copy()
        newMetadata["moduleName"] = moduleName
        newMetadata["moduleBinary"] = os.path.join(buildDir, f"{moduleName}.pyd")
    
    if dryRun:
        return False, None
    
    # Cache the module for later.
    compileAndLoadModule._moduleCache[cacheLookupKey] = slangLib

    return slangLib, newMetadata


compileAndLoadModule._moduleCache = {}


def _compileAndLoadModule(metadata, sources, moduleName, buildDir, slangSourceDir, extraCudaFlags=[], extraSyclFlags=[], verbose=False):
    # make sure to add cl.exe to PATH on windows so ninja can find it.
    _add_msvc_to_env_var()

    extra_cflags = []
    extra_cuda_cflags = []
    # If windows, add /std:c++17 to extra_cflags
    if sys.platform == "win32":
        extra_cflags = ["/std:c++17"]
        extra_cuda_cflags = ["--std=c++17"]

    # If linux/darwin, add -std=c++17 to extra_cflags
    if sys.platform == "linux" or sys.platform == "darwin":
        extra_cflags = ["-std=c++17"]
        extra_cuda_cflags = ["-std=c++17"]

    if extraCudaFlags:
        extra_cuda_cflags.extend(extraCudaFlags)

    if slangSourceDir:
        extra_include_paths = [slangSourceDir]
    else:
        extra_include_paths = None

    extra_cuda_cflags = extra_cuda_cflags + DEFAULT_CUDA_CFLAGS
    extra_sycl_cflags = extraSyclFlags if extraSyclFlags else []
    
    return jit_compile(
        moduleName,
        sources,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags if extra_cuda_cflags else None,
        extra_sycl_cflags=extra_sycl_cflags if extra_sycl_cflags else None,
        extra_ldflags=None,
        extra_include_paths=extra_include_paths,
        build_directory=os.path.realpath(buildDir),
        verbose=verbose,
        is_python_module=True,
        is_standalone=False,
        keep_intermediates=True,
        with_cuda=None,
        with_sycl=None)


def parseDepfile(depFile):
    with open(depFile, 'r') as f:
        depFileContents = f.readlines()

    allDepFiles = []
    for targetEntry in depFileContents:
        _ = targetEntry.split(": ")[0]
        depFiles = [os.path.realpath(token.replace(r"\:", ":")) for token in targetEntry.split(": ")[1].strip().split(" ")]

        # Convert all depfiles to real paths.
        depFilesWithTimestamps = [
            (depFile, os.path.getmtime(depFile)) for depFile in depFiles]
        
        allDepFiles.extend(depFilesWithTimestamps)

    return allDepFiles


def _loadModule(fileName, moduleName, outputFolder, options, sourceDir=None, verbose=False, includePaths=[], dryRun=False, skipNinjaCheck=False, extraCudaFlags=[], extraSlangFlags=[]):

    # Try to find a metadata file "metadata.json" in outputFolder.
    metadataFile = os.path.join(outputFolder, "metadata.json")
    metadata = {}
    if os.path.exists(metadataFile):
        with open(metadataFile, 'r') as f:
            metadata = json.load(f)

    baseName = os.path.basename(fileName)
    realFilePath = os.path.realpath(fileName)
    slangSourceDir = os.path.dirname(realFilePath) if realFilePath else None

    if sourceDir is None:
        cppOutName = os.path.join(outputFolder, _replaceFileExt(baseName, ".cpp"))
        cudaOutName = os.path.join(outputFolder, _replaceFileExt(baseName, "_cuda.cu"))
    else:
        cppOutName = os.path.join(sourceDir, _replaceFileExt(baseName, ".cpp"))
        cudaOutName = os.path.join(sourceDir, _replaceFileExt(baseName, "_cuda.cu"))

    # Compile slang files to intermediate host and kernel modules.
    compileStartTime = time.perf_counter()

    resultCpp, metadataCpp = compileSlang(metadata.get("cpp", None), fileName, "torch-binding", options, cppOutName, verbose, includePaths=includePaths, dryRun=dryRun, extraSlangFlags=extraSlangFlags)
    metadata["cpp"] = metadataCpp

    resultCuda, metadataCuda = compileSlang(metadata.get("cuda", None), fileName, "cuda", options, cudaOutName, verbose, includePaths=includePaths, dryRun=dryRun, extraSlangFlags=extraSlangFlags)
    metadata["cuda"] = metadataCuda

    if dryRun and (resultCuda or resultCpp):
        return True

    compileEndTime = time.perf_counter()

    # Compile host and kernel modules to torch module.
    downstreamStartTime = time.perf_counter()
    
    slangLib, metadata = compileAndLoadModule(
        metadata, [cppOutName, cudaOutName], 
        moduleName, outputFolder, slangSourceDir,
        verbose, dryRun=dryRun, 
        skipNinjaCheck=skipNinjaCheck,
        extraCudaFlags=extraCudaFlags)

    if dryRun:
        if slangLib:
            return True
        else:
            return False
    
    downstreamEndTime = time.perf_counter()

    # Save metadata.
    with open(metadataFile, 'w') as f:
        json.dump(metadata, f, indent=4)

    if verbose:
        print(f"Slang compile time: {compileEndTime-compileStartTime:.3f}s", file=sys.stderr)
        print(f'Downstream compile time: {downstreamEndTime-downstreamStartTime:.3f}s', file=sys.stderr)
    
    return slangLib


def loadModule(fileName, skipSlang=None, verbose=False, defines={}, includePaths=[], skipNinjaCheck=False, slangGenLineInfo=True, cudaFastMath=True, cudaGenLineInfo=True, extraSlangFlags=[], extraCudaFlags=[]):
    # Print warning
    if skipSlang is not None:
        print("Warning: skipSlang is deprecated in favor of a dependency-based cache.", file=sys.stderr)

    if verbose:
        print(f"Loading slang module: {fileName}", file=sys.stderr)
        print(f"Using slangc location: {slangcPath}", file=sys.stderr)

    if not defines:
        defines = {}
    
    if not extraCudaFlags:
        extraCudaFlags = []
    
    if not extraSlangFlags:
        extraSlangFlags = []

    assert(isinstance(cudaFastMath, bool))    
    if cudaFastMath:
        if verbose:
            print("Using fast math (--use_fast_math)", file=sys.stderr)
        extraCudaFlags.append("--use_fast_math")

    assert(isinstance(cudaGenLineInfo, bool))
    if cudaGenLineInfo:
        if verbose:
            print("Using line info (--generate-line-info)", file=sys.stderr)
        extraCudaFlags.append("--generate-line-info")
    
    assert(isinstance(slangGenLineInfo, bool))
    if not slangGenLineInfo:
        if verbose:
            print("Disabling slang line info (-line-directive-mode none)", file=sys.stderr)
        extraSlangFlags.append("-line-directive-mode")
        extraSlangFlags.append("none")

    optionsHash = getHash([defines, extraCudaFlags, extraSlangFlags], truncate_at=16)
    
    parentFolder = os.path.dirname(fileName)
    baseNameWoExt = os.path.splitext(os.path.basename(fileName))[0]
    baseOutputFolder = os.path.join(parentFolder, ".slangtorch_cache", baseNameWoExt)
    outputFolder = os.path.join(baseOutputFolder, optionsHash)

    lockFile = os.path.join(parentFolder, os.path.basename(fileName) + optionsHash + ".lock")
    with FileLock(lockFile):
        # Specialize output folder with hash of the specialization parameters

        if not os.path.exists(outputFolder):
            os.makedirs(outputFolder)
        
        # Common options
        options = makeOptionsList(defines)

        # Module name
        moduleName = f"_slangtorch_{convertNonAlphaNumericToUnderscore(baseNameWoExt)}_{optionsHash}"
        
        # Dry run with latest build dir
        buildDir = getLatestDir(outputFolder, outputFolder)

        if buildDir is not None:
            if verbose:
                print(f"Dry-run using latest build directory: {buildDir}", file=sys.stderr)

            needsRecompile = _loadModule(fileName, moduleName, buildDir, options, sourceDir=outputFolder, verbose=verbose, includePaths=includePaths, dryRun=True, skipNinjaCheck=skipNinjaCheck, extraCudaFlags=extraCudaFlags, extraSlangFlags=extraSlangFlags)
        else:
            if verbose:
                print(f"No latest build directory.", file=sys.stderr)

            needsRecompile = True

        if needsRecompile:
            if verbose:
                print("Build required. Creating unique build directory", file=sys.stderr)
            # Handle versioning
            buildDir = getOrCreateUniqueDir(outputFolder, outputFolder)
        else:
            buildDir = buildDir
        
        if verbose:
            print(f"Working folder: {buildDir}", file=sys.stderr)

        rawModule = _loadModule(fileName, moduleName, buildDir, options, sourceDir=outputFolder, verbose=verbose, includePaths=includePaths, dryRun=False, skipNinjaCheck=skipNinjaCheck, extraCudaFlags=extraCudaFlags, extraSlangFlags=extraSlangFlags)
        addLoadedDirectoryEntry(outputFolder, buildDir)

    return wrapModule(rawModule)


def clearSessionShaderCache():
    compileAndLoadModule._moduleCache = {}


def clearPersistentShaderCache():
    baseOutputFolder = os.path.join(packageDir, '.slangtorch_cache')
    if os.path.exists(baseOutputFolder):
        import shutil
        shutil.rmtree(baseOutputFolder)


def clearShaderCaches():
    clearSessionShaderCache()
    clearPersistentShaderCache()
