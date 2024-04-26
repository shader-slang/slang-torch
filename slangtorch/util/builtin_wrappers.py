from collections import namedtuple
import re
import torch

DiffTensorView = namedtuple('DiffTensorView', ['value', 'grad'], defaults=[None])


def make_diff_tensor_view_wrapper(module, typename, wrappedTypeMap, makeTypeWrapper):
    # Look for module.__typeinfo__DiffTensorView
    typeInfoFnName = f"__typeinfo__{typename}"
    if hasattr(module, typeInfoFnName):
        typeInfoFn = getattr(module, typeInfoFnName)
        (fieldnames, fieldtypenames) = typeInfoFn()

        grad_tensor_typename = fieldtypenames[1]

        # Marshal the user provided input to a tuple(torch.Tensor, tuple(torch.Tensor,))
        def accept_diff_tensor_view(inp):
            if grad_tensor_typename.startswith("AtomicAdd"):
                # Handle a few different cases:
                if isinstance(inp, DiffTensorView):
                    return (inp.value, (inp.grad,))
                elif isinstance(inp, tuple):
                    if len(inp) == 1:
                        return (inp[0], (torch.empty(1, device='cuda'),))
                    elif len(inp) == 2:
                        return (inp[0], (inp[1],))
                    else:
                        raise ValueError(f"Failed to convert to DiffTensorView: Expected tuple of length 1 or 2, got {inp}")
                elif isinstance(inp, torch.Tensor):
                    return (inp, (torch.empty(1, device='cuda'),))
                else:
                    raise ValueError(f"Failed to convert to DiffTensorView: Expected DiffTensorView, tuple or torch.Tensor, got {type(inp)}")
                
            return inp
        wrappedTypeMap[typename] = (DiffTensorView, accept_diff_tensor_view)
        return DiffTensorView, accept_diff_tensor_view
    else:
        raise ValueError(f"Could not find typeinfo for {typename}")


def make_array_wrapper(module, typename, wrappedTypeMap, makeTypeWrapper):
    typeInfoFnName = f"__typeinfo__{typename}"
    if hasattr(module, typeInfoFnName):
        typeInfoFn = getattr(module, typeInfoFnName)
        (fieldnames, fieldtypenames) = typeInfoFn()

        # For an array we should see "type" & "size" fields
        assert len(fieldnames) == 2
        assert "type" in fieldnames
        assert "size" in fieldnames

        elementType = fieldtypenames[fieldnames.index("type")]
        arraySize = int(fieldtypenames[fieldnames.index("size")])

        # Get the wrapper for the element type
        (_, elementTypeConvertFn) = makeTypeWrapper(module, elementType, wrappedTypeMap)

        # convert various input formats to a tuple & perform type checking
        def accept_array(inp):
            if isinstance(inp, tuple):
                if len(inp) != arraySize:
                    raise ValueError(f"Expected tuple of length {arraySize}, got {len(inp)}, when marshalling type {typename}")
                return tuple([elementTypeConvertFn(x) for x in inp])
            elif isinstance(inp, list):
                if len(inp) != arraySize:
                    raise ValueError(f"Expected list of length {arraySize}, got {len(inp)}, when marshalling type {typename}")
                return tuple([elementTypeConvertFn(x) for x in inp])
            else:
                raise ValueError(f"Expected torch.Tensor, tuple or list, got {type(inp)}")
        
        wrappedTypeMap[typename] = (tuple, accept_array)
        
        return tuple, accept_array


def make_vector_wrapper(module, typename, wrappedTypeMap, makeTypeWrapper):
    typeInfoFnName = f"__typeinfo__{typename}"
    if hasattr(module, typeInfoFnName):
        typeInfoFn = getattr(module, typeInfoFnName)
        (fieldnames, fieldtypenames) = typeInfoFn()

        # Vector types get converted into 'VectorStorage' types with an embedded array called "data".
        # Our strategy here is to use the array wrapper to parse the data and pack it into a singleton tuple.
        # 

        assert len(fieldnames) == 1
        assert "data" in fieldnames

        elementType = fieldtypenames[0]

        # Get the wrapper for the element type
        (_, innerArrayConvertFn) = makeTypeWrapper(module, elementType, wrappedTypeMap)

        def accept_vector(inp):
            return tuple([innerArrayConvertFn(inp)])
        
        wrappedTypeMap[typename] = (tuple, accept_vector)
        
        return tuple, accept_vector


def make_matrix_wrapper(module, typename, wrappedTypeMap, makeTypeWrapper):
    typeInfoFnName = f"__typeinfo__{typename}"
    if hasattr(module, typeInfoFnName):
        typeInfoFn = getattr(module, typeInfoFnName)
        (fieldnames, fieldtypenames) = typeInfoFn()

        # Matrix types get converted into 'MatrixStorage' types with an embedded array called "data".
        # Our strategy here is to use the array wrapper to parse the data and pack it into a singleton tuple.
        # 

        assert len(fieldnames) == 1
        assert "data" in fieldnames

        # Parse matrix type name to get the element type and dimensions
        # Find the first two numbers in the typename of the form 'NxM'
        m = re.search(r'\d+x\d+', typename)
        if m is None:
            raise ValueError(f"Could not parse matrix typename {typename}")
        dimensions = m.group(0).split('x')
        assert len(dimensions) == 2

        def accept_matrix(inp):
            # Check that the input is a tuple of tuples
            if not isinstance(inp, tuple):
                raise ValueError(f"Expected tuple, got {type(inp)}")
            if not all(isinstance(x, tuple) for x in inp):
                raise ValueError(f"Expected tuple of tuples, got {inp}")
            if not all(len(x) == int(dimensions[1]) for x in inp):
                raise ValueError(f"Expected tuple of tuples of length {dimensions[1]}, got {inp}")

            # Flatten the input into a single tuple and nest it in another tuple
            return (sum(inp, ()),)
                
        wrappedTypeMap[typename] = (tuple, accept_matrix)
        
        return tuple, accept_matrix

wrappers = {
    'DiffTensorView': make_diff_tensor_view_wrapper,
    'Array_*': make_array_wrapper,
    '_VectorStorage_*': make_vector_wrapper,
    '_MatrixStorage_*': make_matrix_wrapper,
}