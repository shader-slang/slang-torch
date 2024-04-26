import unittest
import os
import sys
from contextlib import contextmanager

import torch
import slangtorch


class TestSlangTorchSmoke(unittest.TestCase):
    def test_smoke(self):
        test_dir = os.path.dirname(os.path.abspath(__file__))
        slangModuleFile = os.path.join(test_dir, 'smoke.slang')
        module = slangtorch.loadModule(slangModuleFile)
        
        X = torch.zeros(2, 2).cuda()
        Y = module.runCompute([X, 1.0])[1].cpu()

        expected = torch.tensor([[1, 1],[1, 1]])

        assert(torch.all(torch.eq(Y, expected)))

@unittest.skip("implicit cast fails currently")
class TestImplicitTypeCast(unittest.TestCase):
    def setUp(self) -> None:
        test_dir = os.path.dirname(os.path.abspath(__file__))
        slangModuleFile = os.path.join(test_dir, 'addarrays.slang')
        module = slangtorch.loadModule(slangModuleFile)
        self.module = module
    
    def test_no_cast(self):
        X = torch.tensor([[1., 2.], [3., 4.]]).cuda()
        Y = torch.tensor([[10., 20.], [30., 40.]]).cuda()
        Z = self.module.add_fwd(X, Y).cpu()

        expected = torch.tensor([[11., 22.],[33., 44.]]).cpu()

        assert(torch.all(torch.eq(Z, expected)))
    
    def test_implicit_cast(self):
        X = torch.tensor([[1, 2], [3, 4]]).cuda()
        Y = torch.tensor([[10, 20], [30, 40]]).cuda()
        Z = self.module.add_fwd(X, Y).cpu()

        expected = torch.tensor([[11, 22],[33, 44]]).cpu()

        assert(torch.all(torch.eq(Z, expected)))

@unittest.skip("implicit device transfer fails currently")
class TestDeviceCast(unittest.TestCase):
    def setUp(self) -> None:
        test_dir = os.path.dirname(os.path.abspath(__file__))
        slangModuleFile = os.path.join(test_dir, 'addarrays.slang')
        module = slangtorch.loadModule(slangModuleFile)
        self.module = module
    
    def test_baseline(self):
        X = torch.tensor([[1., 2.], [3., 4.]]).cuda()
        Y = torch.tensor([[10., 20.], [30., 40.]]).cuda()
        Z = self.module.add_fwd(X, Y).cpu()

        expected = torch.tensor([[11., 22.],[33., 44.]]).cpu()

        assert(torch.all(torch.eq(Z, expected)))
    
    def test_cpu_to_cuda_implicit_transfer(self):
        X = torch.tensor([[1, 2], [3, 4]]).cpu()
        Y = torch.tensor([[10, 20], [30, 40]]).cpu()
        Z = self.module.add_fwd(X, Y).cpu()

        expected = torch.tensor([[11, 22],[33, 44]]).cpu()

        assert(torch.all(torch.eq(Z, expected)))

class TestOptions(unittest.TestCase):
    def test_load_different_options(self):
        test_dir = os.path.dirname(os.path.abspath(__file__))
        slangModuleFile = os.path.join(test_dir, 'multiply.slang')

        module1 = slangtorch.loadModule(slangModuleFile, defines={'FACTOR': '2.0'})
        module2 = slangtorch.loadModule(slangModuleFile, defines={'FACTOR': '1.0'})

        X = torch.tensor([[1., 2.], [3., 4.]]).cuda()
        Y1 = module1.multiply(X).cpu()
        Y2 = module2.multiply(X).cpu()

        expected1 = torch.tensor([[2., 4.],[6., 8.]]).cpu()
        expected2 = torch.tensor([[1., 2.],[3., 4.]]).cpu()

        assert(torch.all(torch.eq(Y1, expected1)))
        assert(torch.all(torch.eq(Y2, expected2)))

    def test_invalid_load(self):
        test_dir = os.path.dirname(os.path.abspath(__file__))
        slangModuleFile = os.path.join(test_dir, 'multiply.slang')

        with self.assertRaises(RuntimeError):
            with suppressOutput():
                module = slangtorch.loadModule(slangModuleFile, defines={})

class TestHotReload(unittest.TestCase):
    def test_hot_reload(self):
        test_dir = os.path.dirname(os.path.abspath(__file__))
        slangModuleTemplateFile = os.path.join(test_dir, 'multiply_template.slang')

        # Get a temporary directory.
        import tempfile
        tmpdir = tempfile.mkdtemp()

        slangModuleFile = os.path.join(tmpdir, 'multiply.slang')

        # Read template file, replace %FACTOR% with 2.0, and write to temporary directory.
        with open(slangModuleTemplateFile, 'r') as f:
            template = f.read()
            template = template.replace(r'%FACTOR%', '2.0')
            with open(slangModuleFile, 'w') as f2:
                f2.write(template)

        module1 = slangtorch.loadModule(slangModuleFile)
        X = torch.tensor([[1., 2.], [3., 4.]]).cuda()
        Y1 = module1.multiply(X).cpu()
        expected1 = torch.tensor([[2., 4.],[6., 8.]]).cpu()
        assert(torch.all(torch.eq(Y1, expected1)))

        # Read template file, replace %FACTOR% with 1.0, and write to temporary directory.
        with open(slangModuleTemplateFile, 'r') as f:
            template = f.read()
            template = template.replace(r'%FACTOR%', '1.0')
            with open(slangModuleFile, 'w') as f2:
                f2.write(template)

        module2 = slangtorch.loadModule(slangModuleFile)
        Y2 = module2.multiply(X).cpu()
        expected2 = torch.tensor([[1., 2.],[3., 4.]]).cpu()
        assert(torch.all(torch.eq(Y2, expected2)))


class TestMultiFileModule(unittest.TestCase):
    def test_multi_file_reload(self):
        test_dir = os.path.dirname(os.path.abspath(__file__))

        importModuleTemplateFile = os.path.join(test_dir, 'import-module.slang')
        importedModuleTemplateFile = os.path.join(test_dir, 'imported-module.slang')

        # Get a temporary directory.
        import tempfile
        import shutil
        tmpdir = tempfile.mkdtemp()

        importedModuleFile = os.path.join(tmpdir, 'imported-module.slang')
        importModuleFile = os.path.join(tmpdir, 'import-module.slang')

        # Read template file, replace %FACTOR% with 2.0, and write to temporary directory.
        with open(importedModuleTemplateFile, 'r') as f:
            template = f.read()
            template = template.replace(r'%FACTOR%', '2.0')
            with open(importedModuleFile, 'w') as f2:
                f2.write(template)
        # Simply copy the other template file to the temporary directory.
        shutil.copy(importModuleTemplateFile, tmpdir)

        module1 = slangtorch.loadModule(importModuleFile)
        X = torch.tensor([[1., 2.], [3., 4.]]).cuda()
        Y1 = module1.multiply(X).cpu()
        expected1 = torch.tensor([[2., 4.],[6., 8.]]).cpu()
        assert(torch.all(torch.eq(Y1, expected1)))

        # Read template file, replace %FACTOR% with 1.0, and write to temporary directory.
        with open(importedModuleTemplateFile, 'r') as f:
            template = f.read()
            template = template.replace(r'%FACTOR%', '1.0')
            with open(importedModuleFile, 'w') as f2:
                f2.write(template)
        
        # Should trigger a recompile. If not, we'll see invalid results.

        module2 = slangtorch.loadModule(importModuleFile)
        Y2 = module2.multiply(X).cpu()
        expected2 = torch.tensor([[1., 2.],[3., 4.]]).cpu()
        assert(torch.all(torch.eq(Y2, expected2)))

class TestCacheState(unittest.TestCase):
    def test_cache_state_on_build_failure(self):
        test_dir = os.path.dirname(os.path.abspath(__file__))
        slangModuleSourceFile = os.path.join(test_dir, 'multiply.slang')

        # Get a temporary directory.
        import tempfile
        tmpdir = tempfile.mkdtemp()

        slangModuleFile = os.path.join(tmpdir, 'multiply.slang')

        # Copy the source file to the temporary directory.
        import shutil
        shutil.copy(slangModuleSourceFile, slangModuleFile)

        # Expect a RuntimeError (define is not set)
        with self.assertRaises(RuntimeError):
            with suppressOutput():
                slangtorch.loadModule(slangModuleFile)
        
        # Run again, should succeed assuming the cache is in proper state.
        module = slangtorch.loadModule(slangModuleFile, defines={'FACTOR': '2.0'})

        X = torch.tensor([[1., 2.], [3., 4.]]).cuda()
        Y1 = module.multiply(X).cpu()
        expected1 = torch.tensor([[2., 4.],[6., 8.]]).cpu()
        assert(torch.all(torch.eq(Y1, expected1)))


class TestCudaPreludeCache(unittest.TestCase):
    def test_cache_state_on_cuda_prelude_modification(self):
        test_dir = os.path.dirname(os.path.abspath(__file__))

        cuPreludeTemplateTestFile = os.path.join(test_dir, 'cuda-intrinsic-test.cuh')
        cuIntrinsicTestFile = os.path.join(test_dir, 'cuda-intrinsic-binding.slang')

        # Get a temporary directory.
        import tempfile
        import shutil
        tmpdir = tempfile.mkdtemp()

        cuPreludeFile = os.path.join(tmpdir, 'cuda-intrinsic-test.cuh')
        cuIntrinsicFile = os.path.join(tmpdir, 'cuda-intrinsic-binding.slang')

        # Read template file, replace %CONST_VAL% with 2.0, and write to temporary directory.
        with open(cuPreludeTemplateTestFile, 'r') as f:
            template = f.read()
            template = template.replace(r'%CONST_VAL%', '2.0')
            with open(cuPreludeFile, 'w') as f2:
                f2.write(template)
        # Simply copy the base slang file to the temporary directory.
        shutil.copy(cuIntrinsicTestFile, tmpdir)

        module1 = slangtorch.loadModule(cuIntrinsicFile)
        X = torch.zeros((1,), dtype=torch.float, device='cuda:0')
        module1.getConst(output=X).launchRaw(blockSize=(1,1,1), gridSize=(1,1,1))
        expected1 = torch.tensor([2.0]).cpu()
        assert(torch.all(torch.eq(X.cpu(), expected1)))

        # Read template file, replace %CONST_VAL% with 1.0, and write to temporary directory.
        with open(cuPreludeTemplateTestFile, 'r') as f:
            template = f.read()
            template = template.replace(r'%CONST_VAL%', '1.0')
            with open(cuPreludeFile, 'w') as f2:
                f2.write(template)
        
        # Should trigger a recompile. If not, we'll see invalid results.

        module2 = slangtorch.loadModule(cuIntrinsicFile)
        X = torch.zeros((1,), dtype=torch.float, device='cuda:0')
        module2.getConst(output=X).launchRaw(blockSize=(1,1,1), gridSize=(1,1,1))
        expected2 = torch.tensor([1.0]).cpu()
        assert(torch.all(torch.eq(X.cpu(), expected2)))

@contextmanager
def suppressOutput():
    # Suppress stdout and stderr for the duration of the context manager.
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

class TestIncludePaths(unittest.TestCase):
    def test_include_paths(self):
        # Create a temp directory structure with two folders
        # each with a separate .slang file. 
        # Call .loadModule() on one of them and attempt to include
        # the other. It should fail.
        # Then add the folder to the include path and try again.
        # It should succeed.
        #

        test_dir = os.path.dirname(os.path.abspath(__file__))
        slangModuleFile = os.path.join(test_dir, 'import-module.slang')
        slangDependencyFile = os.path.join(test_dir, 'imported-module.slang')

        # Get a temporary directory.
        import tempfile
        import shutil
        tmpdir = tempfile.mkdtemp()

        # Create three subdirectories
        subdir1 = os.path.join(tmpdir, 'subdir1')
        subdir2 = os.path.join(tmpdir, 'subdir2')
        subdir3 = os.path.join(tmpdir, 'subdir3')
        os.mkdir(subdir1)
        os.mkdir(subdir2)
        os.mkdir(subdir3)

        # Copy the source file to the temporary directory.
        shutil.copy(slangModuleFile, subdir1)

        # Write the dependency file by replacing %FACTOR% with 2.0
        with open(slangDependencyFile, 'r') as f:
            template = f.read()
            template = template.replace(r'%FACTOR%', '2.0')
            with open(os.path.join(subdir2, 'imported-module.slang'), 'w') as f2:
                f2.write(template)
        
        # Write the dependency file by replacing %FACTOR% with 1.0
        with open(slangDependencyFile, 'r') as f:
            template = f.read()
            template = template.replace(r'%FACTOR%', '1.0')
            with open(os.path.join(subdir3, 'imported-module.slang'), 'w') as f2:
                f2.write(template)

        # Expect a RuntimeError (includePaths is not set)
        # Suppress stderr output for this test.
        with self.assertRaises(RuntimeError):
            with suppressOutput():
                slangtorch.loadModule(os.path.join(subdir1, 'import-module.slang'))

        # Run again, should succeed.
        module = slangtorch.loadModule(os.path.join(subdir1, 'import-module.slang'), includePaths=[subdir2])

        # Check that the output is multiplied by 2.0
        X = torch.tensor([[1., 2.], [3., 4.]]).cuda()
        Y1 = module.multiply(X).cpu()
        expected1 = torch.tensor([[2., 4.],[6., 8.]]).cpu()

        assert(torch.all(torch.eq(Y1, expected1)))

        # Run again with a different include path, should recompile & succeed.
        module = slangtorch.loadModule(os.path.join(subdir1, 'import-module.slang'), includePaths=[subdir3])

        # Check that the output is multiplied by 1.0
        X = torch.tensor([[1., 2.], [3., 4.]]).cuda()
        Y2 = module.multiply(X).cpu()
        expected2 = torch.tensor([[1., 2.],[3., 4.]]).cpu()

        assert(torch.all(torch.eq(Y2, expected2)))


class TestAutoPyBind(unittest.TestCase):
    def test_autopybind(self):
        test_dir = os.path.dirname(os.path.abspath(__file__))
        slangModuleSourceFile = os.path.join(test_dir, 'autobind-square.slang')
        
        module = slangtorch.loadModule(slangModuleSourceFile)

        X = torch.tensor([1., 2., 3., 4.]).cuda()
        Y = torch.zeros_like(X).cuda()

        module.square(input=X, output=Y).launchRaw(blockSize=(32, 32, 1), gridSize=(1, 1, 1))
        expected1 = torch.tensor([1., 4., 9., 16.]).cpu()

        assert(torch.all(torch.eq(Y.cpu(), expected1)))

class TestAutoPyBindDiff(unittest.TestCase):
    def setUp(self) -> None:
        test_dir = os.path.dirname(os.path.abspath(__file__))
        slangModuleSourceFile = os.path.join(test_dir, 'autobind-square-diff.slang')
        
        module = slangtorch.loadModule(slangModuleSourceFile)
        self.module = module
        
    def test_primal_call(self):
        X = torch.tensor([1., 2., 3., 4.]).cuda()
        expected = torch.tensor([1., 4., 9., 16.]).cpu()
        
        # Test call by direct argument
        Y = torch.zeros_like(X).cuda()
        self.module.square(input=X, output=Y).launchRaw(blockSize=(32, 1, 1), gridSize=(1, 1, 1))
        assert(torch.all(torch.eq(Y.cpu(), expected)))

        # Test call by singleton tuple
        Y = torch.zeros_like(X).cuda()
        self.module.square(input=(X,), output=(Y,)).launchRaw(blockSize=(32, 1, 1), gridSize=(1, 1, 1))
        assert(torch.all(torch.eq(Y.cpu(), expected)))

    def test_fwd_diff(self):
        X = torch.tensor([1., 2., 3., 4.]).cuda()
        X_d = torch.tensor([1., 0., 0., 1.]).cuda()
        expected = torch.tensor([1., 4., 9., 16.]).cpu()
        expected_d = torch.tensor([2., 0., 0., 8.]).cpu()
        
        # Test call by direct argument
        Y = torch.zeros_like(X).cuda()
        Y_d = torch.zeros_like(X_d).cuda()
        self.module.square.fwd(input=(X, X_d), output=(Y, Y_d)).launchRaw(blockSize=(32, 1, 1), gridSize=(1, 1, 1))
        assert(torch.all(torch.eq(Y.cpu(), expected)))
        assert(torch.all(torch.eq(Y_d.cpu(), expected_d)))

        # Test call by named tuple
        Y = torch.zeros_like(X).cuda()
        Y_d = torch.zeros_like(X_d).cuda()
        self.module.square.fwd(
            input=self.module.DiffTensorView(value=X, grad=X_d),
            output=self.module.DiffTensorView(value=Y, grad=Y_d)).launchRaw(blockSize=(32, 1, 1), gridSize=(1, 1, 1))
        assert(torch.all(torch.eq(Y.cpu(), expected)))
        assert(torch.all(torch.eq(Y_d.cpu(), expected_d)))
    
    def test_bwd_diff(self):
        X = torch.tensor([1., 2., 3., 4.]).cuda()
        Y = torch.zeros_like(X).cuda()
        Y_d = torch.tensor([1., 0., 1., 0.]).cuda()
        
        expected_d = torch.tensor([2., 0., 6., 0.]).cpu()

        # Test call by direct argument
        X_d = torch.zeros_like(X).cuda()
        self.module.square.bwd(input=(X, X_d), output=(Y, Y_d)).launchRaw(blockSize=(32, 1, 1), gridSize=(1, 1, 1))
        assert(torch.all(torch.eq(X_d.cpu(), expected_d)))

        # Test call by named tuple
        X_d = torch.zeros_like(X).cuda()
        Y_d = torch.tensor([1., 0., 1., 0.]).cuda()
        self.module.square.bwd(
            input=self.module.DiffTensorView(value=X, grad=X_d),
            output=self.module.DiffTensorView(value=Y, grad=Y_d)).launchRaw(blockSize=(32, 1, 1), gridSize=(1, 1, 1))
        assert(torch.all(torch.eq(X_d.cpu(), expected_d)))

class TestAutoPyBindStruct(unittest.TestCase):
    def setUp(self) -> None:
        test_dir = os.path.dirname(os.path.abspath(__file__))
        slangModuleSourceFile = os.path.join(test_dir, 'autobind-multiply-struct.slang')
        
        module = slangtorch.loadModule(slangModuleSourceFile)
        self.module = module

    def test_struct_input(self):
        # Test call struct by tuple
        A = torch.tensor([[1., 2.], [3., 4.]]).cuda()
        B = torch.tensor([[10., 20.], [30., 40.]]).cuda()
        Y = torch.zeros_like(A).cuda()

        self.module.multiply(foo=(A, B), result=Y).launchRaw(blockSize=(32, 32, 1), gridSize=(1, 1, 1))
        expected1 = torch.tensor([[10., 40.],[90., 160.]]).cpu()

        assert(torch.all(torch.eq(Y.cpu(), expected1)))

        # Reset Y
        Y = torch.zeros_like(A).cuda()

        # Test call struct by dict
        self.module.multiply(foo={'A': A, 'B': B}, result=Y).launchRaw(blockSize=(32, 32, 1), gridSize=(1, 1, 1))

        assert(torch.all(torch.eq(Y.cpu(), expected1)))

        # Reset Y
        Y = torch.zeros_like(A).cuda()

        # Test call struct by named tuple
        footype = self.module.Foo
        self.module.multiply(foo=footype(A=A, B=B), result=Y).launchRaw(blockSize=(32, 32, 1), gridSize=(1, 1, 1))
        
        assert(torch.all(torch.eq(Y.cpu(), expected1)))

    def test_struct_failed_input(self):
        A = torch.tensor([[1., 2.], [3., 4.]]).cuda()
        B = torch.tensor([[10., 20.], [30., 40.]]).cuda()
        Y = torch.zeros_like(A).cuda()

        with self.assertRaises(TypeError):
            self.module.multiply(foo=(A,), result=Y).launchRaw(blockSize=(32, 32, 1), gridSize=(1, 1, 1))
        
        with self.assertRaises(TypeError):
            self.module.multiply(foo=(A, B, A), result=Y).launchRaw(blockSize=(32, 32, 1), gridSize=(1, 1, 1))
        
        with self.assertRaises(TypeError):
            self.module.multiply(foo={'A': A}, result=Y).launchRaw(blockSize=(32, 32, 1), gridSize=(1, 1, 1))
        
        with self.assertRaises(TypeError):
            self.module.multiply(foo={'A': A, 'Ba': B}, result=Y).launchRaw(blockSize=(32, 32, 1), gridSize=(1, 1, 1))

class TestBuiltinTypeInputs(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        test_dir = os.path.dirname(os.path.abspath(__file__))
        slangModuleSourceFile = os.path.join(test_dir, 'builtin-type-input.slang')

        module = slangtorch.loadModule(slangModuleSourceFile)
        self.module = module

    def test_plain_vector_input(self):
        Y = torch.tensor([0., 0., 0.]).cuda()

        self.module.plain_copy_float3(input=(1.0, 2.0, 3.0), output=Y).launchRaw(blockSize=(32, 1, 1), gridSize=(1, 1, 1))
        expected1 = torch.tensor([1., 2., 3.]).cpu()

        assert(torch.all(torch.eq(Y.cpu(), expected1)))
    
    def test_plain_matrix_input(self):
        Y = torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 0.]).cuda()

        self.module.plain_copy_float3x3(input=((1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)), output=Y).launchRaw(blockSize=(32, 1, 1), gridSize=(1, 1, 1))
        expected1 = torch.tensor([1., 2., 3., 4., 5., 6., 7., 8., 9.]).cpu()

        assert(torch.all(torch.eq(Y.cpu(), expected1)))

class TestEmptyTensor(unittest.TestCase):
    def setUp(self) -> None:
        test_dir = os.path.dirname(os.path.abspath(__file__))
        slangModuleSourceFile = os.path.join(test_dir, 'copy.slang')
        
        module = slangtorch.loadModule(slangModuleSourceFile)
        self.module = module

    def test_empty_tensor(self):
        # Create empty torch tensor.
        X = torch.tensor([]).cuda()
        Y = torch.zeros_like(X).cuda()

        # Call the module with empty tensor.
        self.module.copy(input=X, output=Y).launchRaw(blockSize=(32, 32, 1), gridSize=(1, 1, 1))

        # Should not crash.
