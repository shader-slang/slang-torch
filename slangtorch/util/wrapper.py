
from typing import Any, Tuple
from collections import namedtuple
import re
import torch

from .builtin_wrappers import wrappers as builtin_wrappers

class LaunchableObject(object):
    def __init__(self, dispatch_fn, name, no_warnings=False) -> None:
        self.dispatch_fn = dispatch_fn
        self.name = name
        self.no_warnings = no_warnings
        self.has_launched = False

    def launchRaw(self, blockSize: Tuple[int, int, int], gridSize: Tuple[int, int, int]):
        # validate inputs blockSize and gridSize should
        # be an iterable of 3 integers
        if not isinstance(blockSize, tuple) or len(blockSize) != 3 or (not all(isinstance(x, int) for x in blockSize)):
            raise ValueError(f"blockSize should be a tuple of 3 integers. Got: {blockSize}")
        
        if not isinstance(gridSize, tuple) or len(gridSize) != 3 or (not all(isinstance(x, int) for x in gridSize)):
            raise ValueError(f"gridSize should be a tuple of 3 integers. Got: {gridSize}")

        # TODO: Do other validation here (blockSize should be <= device max threads per block, etc.)

        # Set state to "launched". This is used to warn the user if they forget to launch the kernel
        self.has_launched = True

        # Dispatch
        return self.dispatch_fn(blockSize, gridSize)

    def launchTotal(self, blockSize: Tuple[int, int, int], totalSize: Tuple[int, int, int]):
        # Calculates gridSize from totalSize & blockSize
        raise NotImplementedError("launchTotal not implemented yet. Use launchRaw(blockSize, gridSize) instead.")

    def autoLaunch(self, totalSize: Tuple[int, int, int]):
        # Launches kernel with the largest possible block size, by querying device shared memory size
        # and using kernel metadata
        #
        raise NotImplementedError("autoLaunch not implemented yet. Use launchRaw(blockSize, gridSize) instead.")

    def __del__(self):
        if not self.has_launched and not self.no_warnings:
            print("\033[93m", end="")
            print(f"[slangtorch] [Warning] LaunchableObject('{self.name}') was never launched. "
                  f"Invoke launchRaw()/launchTotal()/autoLaunch() to run the kernel.")
            print("\033[0m", end="")
        
class WrappedFunction(object):
    def __init__(self, fn_name, fn_handle, argnames, argwrappers, fwd_wrapped_fn = None, bwd_wrapped_fn = None) -> None:
        self.fn_handle = fn_handle
        self.fn_name = fn_name
        self.argnames = argnames
        self.argwrappers = argwrappers

        self.fwd_wrapped_fn = fwd_wrapped_fn
        self.bwd_wrapped_fn = bwd_wrapped_fn

    def kwargs_to_arglist(self, **kwargs):
        arglist = []
        missing_from_input = []
        missing_from_output = []

        for argname in self.argnames:
            if argname in kwargs:
                arglist.append(kwargs[argname])
            else:
                missing_from_input.append(argname)
        
        for argname in kwargs.keys():
            if not argname in self.argnames:
                missing_from_output.append(argname)
        
        if len(missing_from_output) > 0:
            raise ValueError(
                f"{self.fn_name} got unexpected arguments: {missing_from_output}. Available arguments: {self.argnames}")
        
        if len(missing_from_input) > 0:
            raise ValueError(
                f"{self.fn_name} missing required arguments: {missing_from_input}. Available arguments: {self.argnames}")

        return arglist

    def process_arglist(self, arglist):
        return tuple([conv_fn(arg) for (argtype, conv_fn), arg in zip(self.argwrappers, arglist)])
    
    def __call__(self, *args, **kwargs):
        if len(args) > 0:
            raise ValueError(
                f"{self.fn_name} does not support positional arguments, use keyword arguments instead\n"
                "Available arguments: " + str(self.argnames))
        
        arglist = tuple(self.kwargs_to_arglist(**kwargs))
        arglist = self.process_arglist(arglist)
        return LaunchableObject(
            lambda blockSize, gridSize: self.fn_handle(*((blockSize, gridSize) + arglist)),
            name=self.fn_name)

    def fwd(self, *args, **kwargs):
        if self.fwd_wrapped_fn is None:
            raise ValueError(f"{self.fn_name} does not have a fwd-mode derivative attached")
        return self.fwd_wrapped_fn(*args, **kwargs)

    def bwd(self, **kwargs):
        if self.bwd_wrapped_fn is None:
            raise ValueError(f"{self.fn_name} does not have a bwd-mode derivative attached")
        return self.bwd_wrapped_fn(**kwargs)

customWrapperMap = {**builtin_wrappers}

def makeTypeWrapper(module, typename, wrappedTypeMap):
    # Returns a tuple of (public_class, conversion_fn)
    # For most types, this is just a namedtuple for the public class,
    # and a lambda that converts the namedtuple to a regular tuple
    #

    for (regex, wrapperFn) in customWrapperMap.items():
        if re.match(regex, typename):
            return wrapperFn(module, typename, wrappedTypeMap, makeTypeWrapper)
    
    if typename in wrappedTypeMap:
        return wrappedTypeMap[typename]

    # Look for module.__typeinfo__typename
    typeInfoFnName = f"__typeinfo__{typename}"
    if hasattr(module, typeInfoFnName):
        typeInfoFn = getattr(module, typeInfoFnName)
        (fieldnames, fieldtypenames) = typeInfoFn()

        fieldWrappers = []
        for fieldname, fieldtypename in zip(fieldnames, fieldtypenames):
            fieldWrappers.append(makeTypeWrapper(module, fieldtypename, wrappedTypeMap))
        
        publicType = namedtuple(typename, fieldnames)
        def convert(inp):
            # If inp is a namedtuple, convert it to a tuple
            if isinstance(inp, tuple):
                try:
                    inp = publicType(*inp)
                except TypeError as e:
                    raise TypeError(
                        f"Failed to convert {typename} from tuple: {e}")

            elif isinstance(inp, dict):
                inp = publicType(**inp)
        
            if isinstance(inp, publicType):
                # Convert each field of inp
                return tuple([
                    fieldConvertFn(getattr(inp, fieldname)) 
                    for (_, fieldConvertFn), fieldname in zip(fieldWrappers, fieldnames)])
            else:
                raise ValueError(
                    f"Expected {typename}, dict or tuple, got {type(inp)}")

        wrappedTypeMap[typename] = (publicType, convert)
        return publicType, convert
    else:
        # Most likely this is a tensor type.
        # TODO: Make this more robust (we may have basic types too)
        return torch.Tensor, lambda x: x

def wrapModule(module):
    attributes = dict()
    processed = set()
    wrapperTypeMap = dict()

    # Look for all members of module that start with "__funcinfo__"
    for name in dir(module):
        if name.startswith("__funcinfo__"):
            # Call the func_info function to get the function info
            (argnames, argtypes, fwdDiffFnName, bwdDiffFnName) = getattr(module, name)()
            
            for argtypename in argtypes:
                makeTypeWrapper(module, argtypename, wrapperTypeMap)

            # Check that argnames contains a "__blockSize" and "__gridSize" argument
            if not "__blockSize" in argnames:
                raise ValueError("Something went wrong, __blockSize not found in argnames")

            if not "__gridSize" in argnames:
                raise ValueError("Something went wrong, __gridSize not found in argnames")
            
            argwrappers = [wrapperTypeMap.get(argtypename, (None, lambda x: x)) for argtypename in argtypes]
            
            # Pop the "__blockSize" and "__gridSize" arguments from argnames
            argnames = argnames[2:]

            if not fwdDiffFnName == "":
                fwdDiffFn = WrappedFunction(fwdDiffFnName, getattr(module, fwdDiffFnName), argnames, argwrappers)
            else:
                fwdDiffFn = None
            
            if not bwdDiffFnName == "":
                bwdDiffFn = WrappedFunction(bwdDiffFnName, getattr(module, bwdDiffFnName), argnames, argwrappers)
            else:
                bwdDiffFn = None
            
            primalFnName = name[len("__funcinfo__"):]
            # Create a wrapped function
            wrappedFn = WrappedFunction(
                primalFnName,
                getattr(module, primalFnName),
                argnames, argwrappers, fwdDiffFn, bwdDiffFn)
            
            attributes[primalFnName] = wrappedFn
            processed.add(primalFnName)
            processed.add(fwdDiffFnName)
            processed.add(bwdDiffFnName)
    
    # Add all the other members of module to wrappedModule (also ignore __* members)
    for name in dir(module):
        if not name in processed and not name.startswith("__"):
            attributes[name] = getattr(module, name)
    
    # Add all the wrapped types as attributes
    attributes = {**attributes, **{key: publicType for key, (publicType, _) in wrapperTypeMap.items()}}
    
    return type(module.__name__, (object,), attributes)
