#
# Modified version of _jit_compile from torch.utils.cpp_extension that does not 
# implement versioning.
#

from torch.utils.cpp_extension import (
    _write_ninja_file_and_build_library,
    _import_module_from_library,
    _get_exec_path,
    _join_rocm_home,
    _is_cuda_file,
    _get_num_workers,
    PLAT_TO_VCVARS,
    _TORCH_PATH,
    JIT_EXTENSION_VERSIONER,
    IS_HIP_EXTENSION,
    IS_WINDOWS
)

from torch.utils.file_baton import FileBaton
from torch.utils.hipify import hipify_python

from typing import Optional
import sys
import os
import subprocess

def jit_compile(name,
                 sources,
                 extra_cflags,
                 extra_cuda_cflags,
                 extra_ldflags,
                 extra_include_paths,
                 build_directory: str,
                 verbose: bool,
                 with_cuda: Optional[bool],
                 is_python_module,
                 is_standalone,
                 keep_intermediates=True) -> None:
    if is_python_module and is_standalone:
        raise ValueError("`is_python_module` and `is_standalone` are mutually exclusive.")

    if with_cuda is None:
        with_cuda = any(map(_is_cuda_file, sources))
    with_cudnn = any(['cudnn' in f for f in extra_ldflags or []])

    baton = FileBaton(os.path.join(build_directory, 'lock'))
    if baton.try_acquire():
        try:
            with hipify_python.GeneratedFileCleaner(keep_intermediates=keep_intermediates) as clean_ctx:
                if IS_HIP_EXTENSION and (with_cuda or with_cudnn):
                    hipify_result = hipify_python.hipify(
                        project_directory=build_directory,
                        output_directory=build_directory,
                        header_include_dirs=(extra_include_paths if extra_include_paths is not None else []),
                        extra_files=[os.path.abspath(s) for s in sources],
                        ignores=[_join_rocm_home('*'), os.path.join(_TORCH_PATH, '*')],  # no need to hipify ROCm or PyTorch headers
                        show_detailed=verbose,
                        show_progress=verbose,
                        is_pytorch_extension=True,
                        clean_ctx=clean_ctx
                    )

                    hipified_sources = set()
                    for source in sources:
                        s_abs = os.path.abspath(source)
                        hipified_sources.add(hipify_result[s_abs]["hipified_path"] if s_abs in hipify_result else s_abs)

                    sources = list(hipified_sources)

                _write_ninja_file_and_build_library(
                    name=name,
                    sources=sources,
                    extra_cflags=extra_cflags or [],
                    extra_cuda_cflags=extra_cuda_cflags or [],
                    extra_ldflags=extra_ldflags or [],
                    extra_include_paths=extra_include_paths or [],
                    build_directory=build_directory,
                    verbose=verbose,
                    with_cuda=with_cuda,
                    is_standalone=is_standalone)
        finally:
            baton.release()
    else:
        baton.wait()

    if verbose:
        print(f'Loading extension module {name}...', file=sys.stderr)

    if is_standalone:
        return _get_exec_path(name, build_directory)

    return _import_module_from_library(name, build_directory, is_python_module)


class NinjaResult:
    BUILD_FAIL = 0
    BUILD_SUCCESS = 1
    NO_WORK_TO_DO = 2

def run_ninja(
        build_directory: str,
        verbose: bool) -> int:
    r'''Modified version of torch.utils.cpp_extension._run_ninja that explicitly 
        detects a couple of cases: when there's no work to do & when ninja fails.
    '''
    command = ['ninja', '-v']
    num_workers = _get_num_workers(verbose)
    if num_workers is not None:
        command.extend(['-j', str(num_workers)])
    env = os.environ.copy()
    # Try to activate the vc env for the users
    if IS_WINDOWS and 'VSCMD_ARG_TGT_ARCH' not in env:
        from setuptools import distutils

        plat_name = distutils.util.get_platform()
        plat_spec = PLAT_TO_VCVARS[plat_name]

        vc_env = distutils._msvccompiler._get_vc_env(plat_spec)
        vc_env = {k.upper(): v for k, v in vc_env.items()}
        for k, v in env.items():
            uk = k.upper()
            if uk not in vc_env:
                vc_env[uk] = v
        env = vc_env
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        stdout_fileno = 1
        proc = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=build_directory,
            check=True,
            env=env)

        # Read stdout and check for the "no work to do" message
        stdout = proc.stdout.decode()
        if verbose:
            print(stdout)
        
        if "ninja: no work to do." in stdout:
            return NinjaResult.NO_WORK_TO_DO

        # Otherwise, we assume ninja did something & succeeded.
        #  
        # This is a fairly flimsy way to check for success, but as
        # long as ninja doesn't change its output format, it should work..
        #
        return NinjaResult.BUILD_SUCCESS
    except subprocess.CalledProcessError as e:
        if verbose:
            print(e.stdout.decode())
            print(e.stderr.decode())
        return NinjaResult.BUILD_FAIL