// Prelude for PyTorch cpp binding.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>
#include <vector>
#include <stdexcept>
#include <string>

#ifndef SLANG_NO_THROW
#   define SLANG_NO_THROW
#endif

#ifndef SLANG_STDCALL
#   define SLANG_STDCALL
#endif
#ifndef SLANG_MCALL
#   define SLANG_MCALL SLANG_STDCALL
#endif
#ifndef SLANG_FORCE_INLINE
#    define SLANG_FORCE_INLINE inline
#endif

#ifdef SLANG_LLVM
#ifndef SLANG_LLVM_H
#define SLANG_LLVM_H

// TODO(JS): 
// Disable exception declspecs, as not supported on LLVM without some extra options.
// We could enable with `-fms-extensions`
#define SLANG_DISABLE_EXCEPTIONS 1

#ifndef SLANG_PRELUDE_ASSERT
#   ifdef SLANG_PRELUDE_ENABLE_ASSERT
extern "C" void assertFailure(const char* msg);
#       define SLANG_PRELUDE_EXPECT(VALUE, MSG) if(VALUE) {} else assertFailure("assertion failed: '" MSG "'")
#       define SLANG_PRELUDE_ASSERT(VALUE) SLANG_PRELUDE_EXPECT(VALUE, #VALUE)
#   else // SLANG_PRELUDE_ENABLE_ASSERT
#       define SLANG_PRELUDE_EXPECT(VALUE, MSG)
#       define SLANG_PRELUDE_ASSERT(x) 
#   endif // SLANG_PRELUDE_ENABLE_ASSERT
#endif

/*
Taken from stddef.h 
*/

typedef __PTRDIFF_TYPE__ ptrdiff_t;
typedef __SIZE_TYPE__ size_t;
typedef __SIZE_TYPE__ rsize_t;

//typedef __WCHAR_TYPE__ wchar_t;

#if defined(__need_NULL)
#undef NULL
#ifdef __cplusplus
#  if !defined(__MINGW32__) && !defined(_MSC_VER)
#    define NULL __null
#  else
#    define NULL 0
#  endif
#else
#  define NULL ((void*)0)
#endif
#ifdef __cplusplus
#if defined(_MSC_EXTENSIONS) && defined(_NATIVE_NULLPTR_SUPPORTED)
namespace std { typedef decltype(nullptr) nullptr_t; }
using ::std::nullptr_t;
#endif
#endif
#undef __need_NULL
#endif /* defined(__need_NULL) */


/*
The following are taken verbatim from stdint.h from Clang in LLVM. Only 8/16/32/64 types are needed. 
*/

// LLVM/Clang types such that we can use LLVM/Clang without headers for C++ output from Slang

#ifdef __INT64_TYPE__
# ifndef __int8_t_defined /* glibc sys/types.h also defines int64_t*/
typedef __INT64_TYPE__ int64_t;
# endif /* __int8_t_defined */
typedef __UINT64_TYPE__ uint64_t;
# define __int_least64_t int64_t
# define __uint_least64_t uint64_t
#endif /* __INT64_TYPE__ */

#ifdef __int_least64_t
typedef __int_least64_t int_least64_t;
typedef __uint_least64_t uint_least64_t;
typedef __int_least64_t int_fast64_t;
typedef __uint_least64_t uint_fast64_t;
#endif /* __int_least64_t */

#ifdef __INT32_TYPE__

# ifndef __int8_t_defined /* glibc sys/types.h also defines int32_t*/
typedef __INT32_TYPE__ int32_t;
# endif /* __int8_t_defined */

# ifndef __uint32_t_defined  /* more glibc compatibility */
# define __uint32_t_defined
typedef __UINT32_TYPE__ uint32_t;
# endif /* __uint32_t_defined */

# define __int_least32_t int32_t
# define __uint_least32_t uint32_t
#endif /* __INT32_TYPE__ */

#ifdef __int_least32_t
typedef __int_least32_t int_least32_t;
typedef __uint_least32_t uint_least32_t;
typedef __int_least32_t int_fast32_t;
typedef __uint_least32_t uint_fast32_t;
#endif /* __int_least32_t */

#ifdef __INT16_TYPE__
#ifndef __int8_t_defined /* glibc sys/types.h also defines int16_t*/
typedef __INT16_TYPE__ int16_t;
#endif /* __int8_t_defined */
typedef __UINT16_TYPE__ uint16_t;
# define __int_least16_t int16_t
# define __uint_least16_t uint16_t
#endif /* __INT16_TYPE__ */

#ifdef __int_least16_t
typedef __int_least16_t int_least16_t;
typedef __uint_least16_t uint_least16_t;
typedef __int_least16_t int_fast16_t;
typedef __uint_least16_t uint_fast16_t;
#endif /* __int_least16_t */

#ifdef __INT8_TYPE__
#ifndef __int8_t_defined  /* glibc sys/types.h also defines int8_t*/
typedef __INT8_TYPE__ int8_t;
#endif /* __int8_t_defined */
typedef __UINT8_TYPE__ uint8_t;
# define __int_least8_t int8_t
# define __uint_least8_t uint8_t
#endif /* __INT8_TYPE__ */

#ifdef __int_least8_t
typedef __int_least8_t int_least8_t;
typedef __uint_least8_t uint_least8_t;
typedef __int_least8_t int_fast8_t;
typedef __uint_least8_t uint_fast8_t;
#endif /* __int_least8_t */

/* prevent glibc sys/types.h from defining conflicting types */
#ifndef __int8_t_defined
# define __int8_t_defined
#endif /* __int8_t_defined */

/* C99 7.18.1.4 Integer types capable of holding object pointers.
 */
#define __stdint_join3(a,b,c) a ## b ## c

#ifndef _INTPTR_T
#ifndef __intptr_t_defined
typedef __INTPTR_TYPE__ intptr_t;
#define __intptr_t_defined
#define _INTPTR_T
#endif
#endif

#ifndef _UINTPTR_T
typedef __UINTPTR_TYPE__ uintptr_t;
#define _UINTPTR_T
#endif

/* C99 7.18.1.5 Greatest-width integer types.
 */
typedef __INTMAX_TYPE__  intmax_t;
typedef __UINTMAX_TYPE__ uintmax_t;

/* C99 7.18.4 Macros for minimum-width integer constants.
 *
 * The standard requires that integer constant macros be defined for all the
 * minimum-width types defined above. As 8-, 16-, 32-, and 64-bit minimum-width
 * types are required, the corresponding integer constant macros are defined
 * here. This implementation also defines minimum-width types for every other
 * integer width that the target implements, so corresponding macros are
 * defined below, too.
 *
 * These macros are defined using the same successive-shrinking approach as
 * the type definitions above. It is likewise important that macros are defined
 * in order of decending width.
 *
 * Note that C++ should not check __STDC_CONSTANT_MACROS here, contrary to the
 * claims of the C standard (see C++ 18.3.1p2, [cstdint.syn]).
 */

#define __int_c_join(a, b) a ## b
#define __int_c(v, suffix) __int_c_join(v, suffix)
#define __uint_c(v, suffix) __int_c_join(v##U, suffix)

#ifdef __INT64_TYPE__
# ifdef __INT64_C_SUFFIX__
#  define __int64_c_suffix __INT64_C_SUFFIX__
# else
#  undef __int64_c_suffix
# endif /* __INT64_C_SUFFIX__ */
#endif /* __INT64_TYPE__ */

#ifdef __int_least64_t
# ifdef __int64_c_suffix
#  define INT64_C(v) __int_c(v, __int64_c_suffix)
#  define UINT64_C(v) __uint_c(v, __int64_c_suffix)
# else
#  define INT64_C(v) v
#  define UINT64_C(v) v ## U
# endif /* __int64_c_suffix */
#endif /* __int_least64_t */


#ifdef __INT32_TYPE__
# ifdef __INT32_C_SUFFIX__
#  define __int32_c_suffix __INT32_C_SUFFIX__
#else
#  undef __int32_c_suffix
# endif /* __INT32_C_SUFFIX__ */
#endif /* __INT32_TYPE__ */

#ifdef __int_least32_t
# ifdef __int32_c_suffix
#  define INT32_C(v) __int_c(v, __int32_c_suffix)
#  define UINT32_C(v) __uint_c(v, __int32_c_suffix)
# else
#  define INT32_C(v) v
#  define UINT32_C(v) v ## U
# endif /* __int32_c_suffix */
#endif /* __int_least32_t */

#ifdef __INT16_TYPE__
# ifdef __INT16_C_SUFFIX__
#  define __int16_c_suffix __INT16_C_SUFFIX__
#else
#  undef __int16_c_suffix
# endif /* __INT16_C_SUFFIX__ */
#endif /* __INT16_TYPE__ */

#ifdef __int_least16_t
# ifdef __int16_c_suffix
#  define INT16_C(v) __int_c(v, __int16_c_suffix)
#  define UINT16_C(v) __uint_c(v, __int16_c_suffix)
# else
#  define INT16_C(v) v
#  define UINT16_C(v) v ## U
# endif /* __int16_c_suffix */
#endif /* __int_least16_t */


#ifdef __INT8_TYPE__
# ifdef __INT8_C_SUFFIX__
#  define __int8_c_suffix __INT8_C_SUFFIX__
#else
#  undef  __int8_c_suffix
# endif /* __INT8_C_SUFFIX__ */
#endif /* __INT8_TYPE__ */

#ifdef __int_least8_t
# ifdef __int8_c_suffix
#  define INT8_C(v) __int_c(v, __int8_c_suffix)
#  define UINT8_C(v) __uint_c(v, __int8_c_suffix)
# else
#  define INT8_C(v) v
#  define UINT8_C(v) v ## U
# endif /* __int8_c_suffix */
#endif /* __int_least8_t */

/* C99 7.18.2.1 Limits of exact-width integer types.
 * C99 7.18.2.2 Limits of minimum-width integer types.
 * C99 7.18.2.3 Limits of fastest minimum-width integer types.
 *
 * The presence of limit macros are completely optional in C99.  This
 * implementation defines limits for all of the types (exact- and
 * minimum-width) that it defines above, using the limits of the minimum-width
 * type for any types that do not have exact-width representations.
 *
 * As in the type definitions, this section takes an approach of
 * successive-shrinking to determine which limits to use for the standard (8,
 * 16, 32, 64) bit widths when they don't have exact representations. It is
 * therefore important that the definitions be kept in order of decending
 * widths.
 *
 * Note that C++ should not check __STDC_LIMIT_MACROS here, contrary to the
 * claims of the C standard (see C++ 18.3.1p2, [cstdint.syn]).
 */

#ifdef __INT64_TYPE__
# define INT64_MAX           INT64_C( 9223372036854775807)
# define INT64_MIN         (-INT64_C( 9223372036854775807)-1)
# define UINT64_MAX         UINT64_C(18446744073709551615)
# define __INT_LEAST64_MIN   INT64_MIN
# define __INT_LEAST64_MAX   INT64_MAX
# define __UINT_LEAST64_MAX UINT64_MAX
#endif /* __INT64_TYPE__ */

#ifdef __INT_LEAST64_MIN
# define INT_LEAST64_MIN   __INT_LEAST64_MIN
# define INT_LEAST64_MAX   __INT_LEAST64_MAX
# define UINT_LEAST64_MAX __UINT_LEAST64_MAX
# define INT_FAST64_MIN    __INT_LEAST64_MIN
# define INT_FAST64_MAX    __INT_LEAST64_MAX
# define UINT_FAST64_MAX  __UINT_LEAST64_MAX
#endif /* __INT_LEAST64_MIN */

#ifdef __INT32_TYPE__
# define INT32_MAX           INT32_C(2147483647)
# define INT32_MIN         (-INT32_C(2147483647)-1)
# define UINT32_MAX         UINT32_C(4294967295)
# define __INT_LEAST32_MIN   INT32_MIN
# define __INT_LEAST32_MAX   INT32_MAX
# define __UINT_LEAST32_MAX UINT32_MAX
#endif /* __INT32_TYPE__ */

#ifdef __INT_LEAST32_MIN
# define INT_LEAST32_MIN   __INT_LEAST32_MIN
# define INT_LEAST32_MAX   __INT_LEAST32_MAX
# define UINT_LEAST32_MAX __UINT_LEAST32_MAX
# define INT_FAST32_MIN    __INT_LEAST32_MIN
# define INT_FAST32_MAX    __INT_LEAST32_MAX
# define UINT_FAST32_MAX  __UINT_LEAST32_MAX
#endif /* __INT_LEAST32_MIN */

#ifdef __INT16_TYPE__
#define INT16_MAX            INT16_C(32767)
#define INT16_MIN          (-INT16_C(32767)-1)
#define UINT16_MAX          UINT16_C(65535)
# define __INT_LEAST16_MIN   INT16_MIN
# define __INT_LEAST16_MAX   INT16_MAX
# define __UINT_LEAST16_MAX UINT16_MAX
#endif /* __INT16_TYPE__ */

#ifdef __INT_LEAST16_MIN
# define INT_LEAST16_MIN   __INT_LEAST16_MIN
# define INT_LEAST16_MAX   __INT_LEAST16_MAX
# define UINT_LEAST16_MAX __UINT_LEAST16_MAX
# define INT_FAST16_MIN    __INT_LEAST16_MIN
# define INT_FAST16_MAX    __INT_LEAST16_MAX
# define UINT_FAST16_MAX  __UINT_LEAST16_MAX
#endif /* __INT_LEAST16_MIN */


#ifdef __INT8_TYPE__
# define INT8_MAX            INT8_C(127)
# define INT8_MIN          (-INT8_C(127)-1)
# define UINT8_MAX          UINT8_C(255)
# define __INT_LEAST8_MIN    INT8_MIN
# define __INT_LEAST8_MAX    INT8_MAX
# define __UINT_LEAST8_MAX  UINT8_MAX
#endif /* __INT8_TYPE__ */

#ifdef __INT_LEAST8_MIN
# define INT_LEAST8_MIN   __INT_LEAST8_MIN
# define INT_LEAST8_MAX   __INT_LEAST8_MAX
# define UINT_LEAST8_MAX __UINT_LEAST8_MAX
# define INT_FAST8_MIN    __INT_LEAST8_MIN
# define INT_FAST8_MAX    __INT_LEAST8_MAX
# define UINT_FAST8_MAX  __UINT_LEAST8_MAX
#endif /* __INT_LEAST8_MIN */

/* Some utility macros */
#define  __INTN_MIN(n)  __stdint_join3( INT, n, _MIN)
#define  __INTN_MAX(n)  __stdint_join3( INT, n, _MAX)
#define __UINTN_MAX(n)  __stdint_join3(UINT, n, _MAX)
#define  __INTN_C(n, v) __stdint_join3( INT, n, _C(v))
#define __UINTN_C(n, v) __stdint_join3(UINT, n, _C(v))

/* C99 7.18.2.4 Limits of integer types capable of holding object pointers. */
/* C99 7.18.3 Limits of other integer types. */

#define  INTPTR_MIN  (-__INTPTR_MAX__-1)
#define  INTPTR_MAX    __INTPTR_MAX__
#define UINTPTR_MAX   __UINTPTR_MAX__
#define PTRDIFF_MIN (-__PTRDIFF_MAX__-1)
#define PTRDIFF_MAX   __PTRDIFF_MAX__
#define    SIZE_MAX      __SIZE_MAX__

/* ISO9899:2011 7.20 (C11 Annex K): Define RSIZE_MAX if __STDC_WANT_LIB_EXT1__
 * is enabled. */
#if defined(__STDC_WANT_LIB_EXT1__) && __STDC_WANT_LIB_EXT1__ >= 1
#define   RSIZE_MAX            (SIZE_MAX >> 1)
#endif

/* C99 7.18.2.5 Limits of greatest-width integer types. */
#define  INTMAX_MIN (-__INTMAX_MAX__-1)
#define  INTMAX_MAX   __INTMAX_MAX__
#define UINTMAX_MAX  __UINTMAX_MAX__

/* C99 7.18.3 Limits of other integer types. */
#define SIG_ATOMIC_MIN __INTN_MIN(__SIG_ATOMIC_WIDTH__)
#define SIG_ATOMIC_MAX __INTN_MAX(__SIG_ATOMIC_WIDTH__)
#ifdef __WINT_UNSIGNED__
# define WINT_MIN       __UINTN_C(__WINT_WIDTH__, 0)
# define WINT_MAX       __UINTN_MAX(__WINT_WIDTH__)
#else
# define WINT_MIN       __INTN_MIN(__WINT_WIDTH__)
# define WINT_MAX       __INTN_MAX(__WINT_WIDTH__)
#endif

#ifndef WCHAR_MAX
# define WCHAR_MAX __WCHAR_MAX__
#endif
#ifndef WCHAR_MIN
# if __WCHAR_MAX__ == __INTN_MAX(__WCHAR_WIDTH__)
#  define WCHAR_MIN __INTN_MIN(__WCHAR_WIDTH__)
# else
#  define WCHAR_MIN __UINTN_C(__WCHAR_WIDTH__, 0)
# endif
#endif

/* 7.18.4.2 Macros for greatest-width integer constants. */
#define  INTMAX_C(v) __int_c(v,  __INTMAX_C_SUFFIX__)
#define UINTMAX_C(v) __int_c(v, __UINTMAX_C_SUFFIX__)


#endif // SLANG_LLVM_H



#else // SLANG_LLVM
#   if SLANG_GCC_FAMILY && __GNUC__ < 6
#       include <cmath>
#       define SLANG_PRELUDE_STD std::
#   else
#       include <math.h>
#       define SLANG_PRELUDE_STD
#   endif

#   include <assert.h>
#   include <stdlib.h>
#   include <string.h>
#   include <stdint.h>
#endif // SLANG_LLVM

#ifndef SLANG_PRELUDE_CPP_TYPES_CORE_H
#define SLANG_PRELUDE_CPP_TYPES_CORE_H

#ifndef SLANG_PRELUDE_ASSERT
#   ifdef SLANG_PRELUDE_ENABLE_ASSERT
#       define SLANG_PRELUDE_ASSERT(VALUE) assert(VALUE)
#   else
#       define SLANG_PRELUDE_ASSERT(VALUE) 
#   endif
#endif

// Since we are using unsigned arithmatic care is need in this comparison.
// It is *assumed* that sizeInBytes >= elemSize. Which means (sizeInBytes >= elemSize) >= 0
// Which means only a single test is needed

// Asserts for bounds checking.
// It is assumed index/count are unsigned types.
#define SLANG_BOUND_ASSERT(index, count)  SLANG_PRELUDE_ASSERT(index < count); 
#define SLANG_BOUND_ASSERT_BYTE_ADDRESS(index, elemSize, sizeInBytes) SLANG_PRELUDE_ASSERT(index <= (sizeInBytes - elemSize) && (index & 3) == 0);

// Macros to zero index if an access is out of range
#define SLANG_BOUND_ZERO_INDEX(index, count) index = (index < count) ? index : 0; 
#define SLANG_BOUND_ZERO_INDEX_BYTE_ADDRESS(index, elemSize, sizeInBytes) index = (index <= (sizeInBytes - elemSize)) ? index : 0; 

// The 'FIX' macro define how the index is fixed. The default is to do nothing. If SLANG_ENABLE_BOUND_ZERO_INDEX
// the fix macro will zero the index, if out of range
#ifdef  SLANG_ENABLE_BOUND_ZERO_INDEX
#   define SLANG_BOUND_FIX(index, count) SLANG_BOUND_ZERO_INDEX(index, count)
#   define SLANG_BOUND_FIX_BYTE_ADDRESS(index, elemSize, sizeInBytes) SLANG_BOUND_ZERO_INDEX_BYTE_ADDRESS(index, elemSize, sizeInBytes)
#   define SLANG_BOUND_FIX_FIXED_ARRAY(index, count) SLANG_BOUND_ZERO_INDEX(index, count)
#else
#   define SLANG_BOUND_FIX(index, count) 
#   define SLANG_BOUND_FIX_BYTE_ADDRESS(index, elemSize, sizeInBytes) 
#   define SLANG_BOUND_FIX_FIXED_ARRAY(index, count) 
#endif

#ifndef SLANG_BOUND_CHECK
#   define SLANG_BOUND_CHECK(index, count) SLANG_BOUND_ASSERT(index, count) SLANG_BOUND_FIX(index, count)
#endif

#ifndef SLANG_BOUND_CHECK_BYTE_ADDRESS
#   define SLANG_BOUND_CHECK_BYTE_ADDRESS(index, elemSize, sizeInBytes) SLANG_BOUND_ASSERT_BYTE_ADDRESS(index, elemSize, sizeInBytes) SLANG_BOUND_FIX_BYTE_ADDRESS(index, elemSize, sizeInBytes)
#endif

#ifndef SLANG_BOUND_CHECK_FIXED_ARRAY
#   define SLANG_BOUND_CHECK_FIXED_ARRAY(index, count) SLANG_BOUND_ASSERT(index, count) SLANG_BOUND_FIX_FIXED_ARRAY(index, count)
#endif

struct TypeInfo
{
    size_t typeSize;
};

template <typename T, size_t SIZE>
struct FixedArray
{
    const T& operator[](size_t index) const { SLANG_BOUND_CHECK_FIXED_ARRAY(index, SIZE); return m_data[index]; }
    T& operator[](size_t index) { SLANG_BOUND_CHECK_FIXED_ARRAY(index, SIZE); return m_data[index]; }

    T m_data[SIZE];
};

// An array that has no specified size, becomes a 'Array'. This stores the size so it can potentially 
// do bounds checking.  
template <typename T>
struct Array
{
    const T& operator[](size_t index) const { SLANG_BOUND_CHECK(index, count); return data[index]; }
    T& operator[](size_t index) { SLANG_BOUND_CHECK(index, count); return data[index]; }

    T* data;
    size_t count;
};

/* Constant buffers become a pointer to the contained type, so ConstantBuffer<T> becomes T* in C++ code.
*/

template <typename T, int COUNT>
struct Vector;

template <typename T>
struct Vector<T, 1>
{
    T x;
    const T& operator[](size_t /*index*/) const { return x; }
    T& operator[](size_t /*index*/) { return x; }
    operator T() const { return x; }
    Vector() = default;
    Vector(T scalar)
    {
        x = scalar;
    }
    template <typename U>
    Vector(Vector<U, 1> other)
    {
        x = (T)other.x;
    }
    template <typename U, int otherSize>
    Vector(Vector<U, otherSize> other)
    {
        int minSize = 1;
        if (otherSize < minSize) minSize = otherSize;
        for (int i = 0; i < minSize; i++)
            (*this)[i] = (T)other[i];
    }
};

template <typename T>
struct Vector<T, 2>
{
    T x, y;
    const T& operator[](size_t index) const { return index == 0 ? x : y; }
    T& operator[](size_t index) { return index == 0 ? x : y; }
    Vector() = default;
    Vector(T scalar)
    {
        x = y = scalar;
    }
    Vector(T _x, T _y)
    {
        x = _x;
        y = _y;
    }
    template <typename U>
    Vector(Vector<U, 2> other)
    {
        x = (T)other.x;
        y = (T)other.y;
    }
    template <typename U, int otherSize>
    Vector(Vector<U, otherSize> other)
    {
        int minSize = 2;
        if (otherSize < minSize) minSize = otherSize;
        for (int i = 0; i < minSize; i++)
            (*this)[i] = (T)other[i];
    }
};

template <typename T>
struct Vector<T, 3>
{
    T x, y, z;
    const T& operator[](size_t index) const { return *((T*)(this) + index); }
    T& operator[](size_t index) { return *((T*)(this) + index); }

    Vector() = default;
    Vector(T scalar)
    {
        x = y = z = scalar;
    }
    Vector(T _x, T _y, T _z)
    {
        x = _x;
        y = _y;
        z = _z;
    }
    template <typename U>
    Vector(Vector<U, 3> other)
    {
        x = (T)other.x;
        y = (T)other.y;
        z = (T)other.z;
    }
    template <typename U, int otherSize>
    Vector(Vector<U, otherSize> other)
    {
        int minSize = 3;
        if (otherSize < minSize) minSize = otherSize;
        for (int i = 0; i < minSize; i++)
            (*this)[i] = (T)other[i];
    }
};

template <typename T>
struct Vector<T, 4>
{
    T x, y, z, w;

    const T& operator[](size_t index) const { return *((T*)(this) + index); }
    T& operator[](size_t index) { return *((T*)(this) + index); }
    Vector() = default;
    Vector(T scalar)
    {
        x = y = z = w = scalar;
    }
    Vector(T _x, T _y, T _z, T _w)
    {
        x = _x;
        y = _y;
        z = _z;
        w = _w;
    }
    template <typename U, int otherSize>
    Vector(Vector<U, otherSize> other)
    {
        int minSize = 4;
        if (otherSize < minSize) minSize = otherSize;
        for (int i = 0; i < minSize; i++)
            (*this)[i] = (T)other[i];
    }
 
};

template<typename T, int N>
SLANG_FORCE_INLINE T _slang_vector_get_element(Vector<T, N> x, int index)
{
    return x[index];
}

template<typename T, int N>
SLANG_FORCE_INLINE const T* _slang_vector_get_element_ptr(const Vector<T, N>* x, int index)
{
    return &((*const_cast<Vector<T,N>*>(x))[index]);
}

template<typename T, int N>
SLANG_FORCE_INLINE T* _slang_vector_get_element_ptr(Vector<T, N>* x, int index)
{
    return &((*x)[index]);
}

template<typename T, int n, typename OtherT, int m>
SLANG_FORCE_INLINE Vector<T, n> _slang_vector_reshape(const Vector<OtherT, m> other)
{
    Vector<T, n> result;
    for (int i = 0; i < n; i++)
    {
        OtherT otherElement = T(0);
        if (i < m)
            otherElement = _slang_vector_get_element(other, i);
        *_slang_vector_get_element_ptr(&result, i) = (T)otherElement;
    }
    return result;
}

typedef uint32_t uint;

#define SLANG_VECTOR_BINARY_OP(T, op) \
    template<int n> \
    SLANG_FORCE_INLINE Vector<T, n> operator op(const Vector<T, n>& thisVal, const Vector<T, n>& other) \
    { \
        Vector<T, n> result;\
        for (int i = 0; i < n; i++) \
            result[i] = thisVal[i] op other[i]; \
        return result;\
    }
#define SLANG_VECTOR_BINARY_COMPARE_OP(T, op) \
    template<int n> \
    SLANG_FORCE_INLINE Vector<bool, n> operator op(const Vector<T, n>& thisVal, const Vector<T, n>& other) \
    { \
        Vector<bool, n> result;\
        for (int i = 0; i < n; i++) \
            result[i] = thisVal[i] op other[i]; \
        return result;\
    }

#define SLANG_VECTOR_UNARY_OP(T, op) \
    template<int n> \
    SLANG_FORCE_INLINE Vector<T, n> operator op(const Vector<T, n>& thisVal) \
    { \
        Vector<T, n> result;\
        for (int i = 0; i < n; i++) \
            result[i] = op thisVal[i]; \
        return result;\
    }
#define SLANG_INT_VECTOR_OPS(T) \
    SLANG_VECTOR_BINARY_OP(T, +)\
    SLANG_VECTOR_BINARY_OP(T, -)\
    SLANG_VECTOR_BINARY_OP(T, *)\
    SLANG_VECTOR_BINARY_OP(T, / )\
    SLANG_VECTOR_BINARY_OP(T, &)\
    SLANG_VECTOR_BINARY_OP(T, |)\
    SLANG_VECTOR_BINARY_OP(T, &&)\
    SLANG_VECTOR_BINARY_OP(T, ||)\
    SLANG_VECTOR_BINARY_OP(T, ^)\
    SLANG_VECTOR_BINARY_OP(T, %)\
    SLANG_VECTOR_BINARY_OP(T, >>)\
    SLANG_VECTOR_BINARY_OP(T, <<)\
    SLANG_VECTOR_BINARY_COMPARE_OP(T, >)\
    SLANG_VECTOR_BINARY_COMPARE_OP(T, <)\
    SLANG_VECTOR_BINARY_COMPARE_OP(T, >=)\
    SLANG_VECTOR_BINARY_COMPARE_OP(T, <=)\
    SLANG_VECTOR_BINARY_COMPARE_OP(T, ==)\
    SLANG_VECTOR_BINARY_COMPARE_OP(T, !=)\
    SLANG_VECTOR_UNARY_OP(T, !)\
    SLANG_VECTOR_UNARY_OP(T, ~)
#define SLANG_FLOAT_VECTOR_OPS(T) \
    SLANG_VECTOR_BINARY_OP(T, +)\
    SLANG_VECTOR_BINARY_OP(T, -)\
    SLANG_VECTOR_BINARY_OP(T, *)\
    SLANG_VECTOR_BINARY_OP(T, /)\
    SLANG_VECTOR_UNARY_OP(T, -)\
    SLANG_VECTOR_BINARY_COMPARE_OP(T, >)\
    SLANG_VECTOR_BINARY_COMPARE_OP(T, <)\
    SLANG_VECTOR_BINARY_COMPARE_OP(T, >=)\
    SLANG_VECTOR_BINARY_COMPARE_OP(T, <=)\
    SLANG_VECTOR_BINARY_COMPARE_OP(T, ==)\
    SLANG_VECTOR_BINARY_COMPARE_OP(T, !=)

SLANG_INT_VECTOR_OPS(bool)
SLANG_INT_VECTOR_OPS(int)
SLANG_INT_VECTOR_OPS(int8_t)
SLANG_INT_VECTOR_OPS(int16_t)
SLANG_INT_VECTOR_OPS(int64_t)
SLANG_INT_VECTOR_OPS(uint)
SLANG_INT_VECTOR_OPS(uint8_t)
SLANG_INT_VECTOR_OPS(uint16_t)
SLANG_INT_VECTOR_OPS(uint64_t)

SLANG_FLOAT_VECTOR_OPS(float)
SLANG_FLOAT_VECTOR_OPS(double)

#define SLANG_VECTOR_INT_NEG_OP(T) \
    template<int N>\
    Vector<T, N> operator-(const Vector<T, N>& thisVal) \
    { \
        Vector<T, N> result;\
        for (int i = 0; i < N; i++) \
            result[i] = 0 - thisVal[i]; \
        return result;\
    }
SLANG_VECTOR_INT_NEG_OP(int)
SLANG_VECTOR_INT_NEG_OP(int8_t)
SLANG_VECTOR_INT_NEG_OP(int16_t)
SLANG_VECTOR_INT_NEG_OP(int64_t)
SLANG_VECTOR_INT_NEG_OP(uint)
SLANG_VECTOR_INT_NEG_OP(uint8_t)
SLANG_VECTOR_INT_NEG_OP(uint16_t)
SLANG_VECTOR_INT_NEG_OP(uint64_t)

#define SLANG_FLOAT_VECTOR_MOD(T)\
    template<int N> \
    Vector<T, N> operator%(const Vector<T, N>& left, const Vector<T, N>& right) \
    {\
        Vector<T, N> result;\
        for (int i = 0; i < N; i++) \
            result[i] = _slang_fmod(left[i], right[i]); \
        return result;\
    }

SLANG_FLOAT_VECTOR_MOD(float)
SLANG_FLOAT_VECTOR_MOD(double)
#undef SLANG_FLOAT_VECTOR_MOD
#undef SLANG_VECTOR_BINARY_OP
#undef SLANG_VECTOR_UNARY_OP
#undef SLANG_INT_VECTOR_OPS
#undef SLANG_FLOAT_VECTOR_OPS
#undef SLANG_VECTOR_INT_NEG_OP
#undef SLANG_FLOAT_VECTOR_MOD

template <typename T, int ROWS, int COLS>
struct Matrix
{
    Vector<T, COLS> rows[ROWS];
    Vector<T, COLS>& operator[](size_t index) { return rows[index]; }
    Matrix() = default;
    Matrix(T scalar)
    {
        for (int i = 0; i < ROWS; i++)
            rows[i] = Vector<T, COLS>(scalar);
    }
    Matrix(const Vector<T, COLS>& row0)
    {
        rows[0] = row0;
    }
    Matrix(const Vector<T, COLS>& row0, const Vector<T, COLS>& row1)
    {
        rows[0] = row0;
        rows[1] = row1;
    }
    Matrix(const Vector<T, COLS>& row0, const Vector<T, COLS>& row1, const Vector<T, COLS>& row2)
    {
        rows[0] = row0;
        rows[1] = row1;
        rows[2] = row2;
    }
    Matrix(const Vector<T, COLS>& row0, const Vector<T, COLS>& row1, const Vector<T, COLS>& row2, const Vector<T, COLS>& row3)
    {
        rows[0] = row0;
        rows[1] = row1;
        rows[2] = row2;
        rows[3] = row3;
    }
    template<typename U, int otherRow, int otherCol>
    Matrix(const Matrix<U, otherRow, otherCol>& other)
    {
        int minRow = ROWS;
        int minCol = COLS;
        if (minRow > otherRow) minRow = otherRow;
        if (minCol > otherCol) minCol = otherCol;
        for (int i = 0; i < minRow; i++)
            for (int j = 0; j < minCol; j++)
                rows[i][j] = (T)other.rows[i][j];
    }
    Matrix(T v0, T v1, T v2, T v3)
    {
        rows[0][0] = v0;  rows[0][1] = v1;
        rows[1][0] = v2;  rows[1][1] = v3;
    }
    Matrix(T v0, T v1, T v2, T v3, T v4, T v5)
    {
        if (COLS == 3)
        {
            rows[0][0] = v0;  rows[0][1] = v1; rows[0][2] = v2;
            rows[1][0] = v3;  rows[1][1] = v4; rows[1][2] = v5;
        }
        else
        {
            rows[0][0] = v0;  rows[0][1] = v1;
            rows[1][0] = v2;  rows[1][1] = v3;
            rows[2][0] = v4;  rows[2][1] = v5;
        }
    }
    Matrix(T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7)
    {
        if (COLS == 4)
        {
            rows[0][0] = v0;  rows[0][1] = v1; rows[0][2] = v2; rows[0][3] = v3;
            rows[1][0] = v4;  rows[1][1] = v5; rows[1][2] = v6; rows[1][3] = v7;
        }
        else
        {
            rows[0][0] = v0;  rows[0][1] = v1;
            rows[1][0] = v2;  rows[1][1] = v3;
            rows[2][0] = v4;  rows[2][1] = v5;
            rows[3][0] = v6;  rows[3][1] = v7;
        }
    }
    Matrix(T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7, T v8)
    {
        rows[0][0] = v0;  rows[0][1] = v1;  rows[0][2] = v2;
        rows[1][0] = v3;  rows[1][1] = v4;  rows[1][2] = v5;
        rows[2][0] = v6;  rows[2][1] = v7;  rows[2][2] = v8;
    }
    Matrix(T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7, T v8, T v9, T v10, T v11)
    {
        if (COLS == 4)
        {
            rows[0][0] = v0;  rows[0][1] = v1;  rows[0][2] = v2;  rows[0][3] = v3;
            rows[1][0] = v4;  rows[1][1] = v5;  rows[1][2] = v6;  rows[1][3] = v7;
            rows[2][0] = v8;  rows[2][1] = v9;  rows[2][2] = v10; rows[2][3] = v11;
        }
        else
        {
            rows[0][0] = v0;  rows[0][1] = v1;  rows[0][2] = v2;
            rows[1][0] = v3;  rows[1][1] = v4;  rows[1][2] = v5;
            rows[2][0] = v6;  rows[2][1] = v7;  rows[2][2] = v8;
            rows[3][0] = v9;  rows[3][1] = v10; rows[3][2] = v11;
        }
    }
    Matrix(T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7, T v8, T v9, T v10, T v11, T v12, T v13, T v14, T v15)
    {
        rows[0][0] = v0;  rows[0][1] = v1;  rows[0][2] = v2;  rows[0][3] = v3;
        rows[1][0] = v4;  rows[1][1] = v5;  rows[1][2] = v6;  rows[1][3] = v7;
        rows[2][0] = v8;  rows[2][1] = v9;  rows[2][2] = v10; rows[2][3] = v11;
        rows[3][0] = v12; rows[3][1] = v13; rows[3][2] = v14; rows[3][3] = v15;
    }
};

#define SLANG_MATRIX_BINARY_OP(T, op) \
    template<int R, int C> \
    Matrix<T, R, C> operator op(const Matrix<T, R, C>& thisVal, const Matrix<T, R, C>& other) \
    { \
        Matrix<T, R, C> result;\
        for (int i = 0; i < R; i++) \
            for (int j = 0; j < C; j++) \
                result.rows[i][j] = thisVal.rows[i][j] op other.rows[i][j]; \
        return result;\
    }

#define SLANG_MATRIX_UNARY_OP(T, op) \
    template<int R, int C> \
    Matrix<T, R, C> operator op(const Matrix<T, R, C>& thisVal) \
    { \
        Matrix<T, R, C> result;\
        for (int i = 0; i < R; i++) \
            for (int j = 0; j < C; j++) \
                result[i].rows[i][j] = op thisVal.rows[i][j]; \
        return result;\
    }
#define SLANG_INT_MATRIX_OPS(T) \
    SLANG_MATRIX_BINARY_OP(T, +)\
    SLANG_MATRIX_BINARY_OP(T, -)\
    SLANG_MATRIX_BINARY_OP(T, *)\
    SLANG_MATRIX_BINARY_OP(T, / )\
    SLANG_MATRIX_BINARY_OP(T, &)\
    SLANG_MATRIX_BINARY_OP(T, |)\
    SLANG_MATRIX_BINARY_OP(T, &&)\
    SLANG_MATRIX_BINARY_OP(T, ||)\
    SLANG_MATRIX_BINARY_OP(T, ^)\
    SLANG_MATRIX_BINARY_OP(T, %)\
    SLANG_MATRIX_UNARY_OP(T, !)\
    SLANG_MATRIX_UNARY_OP(T, ~)
#define SLANG_FLOAT_MATRIX_OPS(T) \
    SLANG_MATRIX_BINARY_OP(T, +)\
    SLANG_MATRIX_BINARY_OP(T, -)\
    SLANG_MATRIX_BINARY_OP(T, *)\
    SLANG_MATRIX_BINARY_OP(T, /)\
    SLANG_MATRIX_UNARY_OP(T, -)
SLANG_INT_MATRIX_OPS(int)
SLANG_INT_MATRIX_OPS(int8_t)
SLANG_INT_MATRIX_OPS(int16_t)
SLANG_INT_MATRIX_OPS(int64_t)
SLANG_INT_MATRIX_OPS(uint)
SLANG_INT_MATRIX_OPS(uint8_t)
SLANG_INT_MATRIX_OPS(uint16_t)
SLANG_INT_MATRIX_OPS(uint64_t)

SLANG_FLOAT_MATRIX_OPS(float)
SLANG_FLOAT_MATRIX_OPS(double)

#define SLANG_MATRIX_INT_NEG_OP(T) \
    template<int R, int C>\
    SLANG_FORCE_INLINE Matrix<T, R, C> operator-(Matrix<T, R, C> thisVal) \
    { \
        Matrix<T, R, C> result;\
        for (int i = 0; i < R; i++) \
            for (int j = 0; j < C; j++) \
            result.rows[i][j] = 0 - thisVal.rows[i][j]; \
        return result;\
    }
    SLANG_MATRIX_INT_NEG_OP(int)
    SLANG_MATRIX_INT_NEG_OP(int8_t)
    SLANG_MATRIX_INT_NEG_OP(int16_t)
    SLANG_MATRIX_INT_NEG_OP(int64_t)
    SLANG_MATRIX_INT_NEG_OP(uint)
    SLANG_MATRIX_INT_NEG_OP(uint8_t)
    SLANG_MATRIX_INT_NEG_OP(uint16_t)
    SLANG_MATRIX_INT_NEG_OP(uint64_t)

#define SLANG_FLOAT_MATRIX_MOD(T)\
    template<int R, int C> \
    SLANG_FORCE_INLINE Matrix<T, R, C> operator%(Matrix<T, R, C> left, Matrix<T, R, C> right) \
    {\
        Matrix<T, R, C> result;\
        for (int i = 0; i < R; i++) \
            for (int j = 0; j < C; j++) \
                result.rows[i][j] = _slang_fmod(left.rows[i][j], right.rows[i][j]); \
        return result;\
    }

    SLANG_FLOAT_MATRIX_MOD(float)
    SLANG_FLOAT_MATRIX_MOD(double)
#undef SLANG_FLOAT_MATRIX_MOD
#undef SLANG_MATRIX_BINARY_OP
#undef SLANG_MATRIX_UNARY_OP
#undef SLANG_INT_MATRIX_OPS
#undef SLANG_FLOAT_MATRIX_OPS
#undef SLANG_MATRIX_INT_NEG_OP
#undef SLANG_FLOAT_MATRIX_MOD

template<typename TResult, typename TInput>
TResult slang_bit_cast(TInput val)
{
    return *(TResult*)(&val);
}

#endif



#ifndef SLANG_PRELUDE_SCALAR_INTRINSICS_H
#define SLANG_PRELUDE_SCALAR_INTRINSICS_H

#if !defined(SLANG_LLVM) && SLANG_PROCESSOR_X86_64 && SLANG_VC
//  If we have visual studio and 64 bit processor, we can assume we have popcnt, and can include x86 intrinsics
#   include <intrin.h>
#endif

#ifndef SLANG_FORCE_INLINE
#    define SLANG_FORCE_INLINE inline
#endif

#ifdef SLANG_PRELUDE_NAMESPACE
namespace SLANG_PRELUDE_NAMESPACE {
#endif

#ifndef SLANG_PRELUDE_PI
#   define SLANG_PRELUDE_PI           3.14159265358979323846
#endif


union Union32 
{
    uint32_t u;
    int32_t i;
    float f;
};

union Union64
{
    uint64_t u;
    int64_t i;
    double d;
};

// 32 bit cast conversions
SLANG_FORCE_INLINE int32_t _bitCastFloatToInt(float f) { Union32 u; u.f = f; return u.i; }
SLANG_FORCE_INLINE float _bitCastIntToFloat(int32_t i) { Union32 u; u.i = i; return u.f; }
SLANG_FORCE_INLINE uint32_t _bitCastFloatToUInt(float f) { Union32 u; u.f = f; return u.u; }
SLANG_FORCE_INLINE float _bitCastUIntToFloat(uint32_t ui) { Union32 u; u.u = ui; return u.f; }

// ----------------------------- F16 -----------------------------------------


// This impl is based on FloatToHalf that is in Slang codebase
uint32_t f32tof16(const float value)
{
    const uint32_t inBits = _bitCastFloatToUInt(value);

    // bits initially set to just the sign bit
    uint32_t bits = (inBits >> 16) & 0x8000;
    // Mantissa can't be used as is, as it holds last bit, for rounding.
    uint32_t m = (inBits >> 12) & 0x07ff;
    uint32_t e = (inBits >> 23) & 0xff;

    if (e < 103)
    {
        // It's zero
        return bits;
    }
    if (e == 0xff)
    {
        // Could be a NAN or INF. Is INF if *input* mantissa is 0.
        
        // Remove last bit for rounding to make output mantissa.
        m >>= 1;
       
        // We *assume* float16/float32 signaling bit and remaining bits
        // semantics are the same. (The signalling bit convention is target specific!).
        // Non signal bit's usage within mantissa for a NAN are also target specific.
      
        // If the m is 0, it could be because the result is INF, but it could also be because all the 
        // bits that made NAN were dropped as we have less mantissa bits in f16. 
           
        // To fix for this we make non zero if m is 0 and the input mantissa was not.
        // This will (typically) produce a signalling NAN.
        m += uint32_t(m == 0 && (inBits & 0x007fffffu));
       
        // Combine for output
        return (bits | 0x7c00u | m);
    }
    if (e > 142)
    {
        // INF. 
        return bits | 0x7c00u;
    }
    if (e < 113)
    {
        m |= 0x0800u;
        bits |= (m >> (114 - e)) + ((m >> (113 - e)) & 1);
        return bits;
    }
    bits |= ((e - 112) << 10) | (m >> 1);
    bits += m & 1;
    return bits;
}

static const float g_f16tof32Magic = _bitCastIntToFloat((127 + (127 - 15)) << 23);

float f16tof32(const uint32_t value)
{
    const uint32_t sign = (value & 0x8000) << 16;
    uint32_t exponent = (value & 0x7c00) >> 10;
    uint32_t mantissa = (value & 0x03ff);

    if (exponent == 0)
    {
        // If mantissa is 0 we are done, as output is 0. 
        // If it's not zero we must have a denormal.
        if (mantissa)
        {
            // We have a denormal so use the magic to do exponent adjust
            return _bitCastIntToFloat(sign | ((value & 0x7fff) << 13)) * g_f16tof32Magic;
        }
    }
    else 
    {
        // If the exponent is NAN or INF exponent is 0x1f on input. 
        // If that's the case, we just need to set the exponent to 0xff on output
        // and the mantissa can just stay the same. If its 0 it's INF, else it is NAN and we just copy the bits
        //
        // Else we need to correct the exponent in the normalized case.
        exponent = (exponent == 0x1F) ? 0xff : (exponent + (-15 + 127));
    }
    
    return _bitCastUIntToFloat(sign | (exponent << 23) | (mantissa << 13));
}

// ----------------------------- F32 -----------------------------------------

// Helpers
SLANG_FORCE_INLINE float F32_calcSafeRadians(float radians);

#ifdef SLANG_LLVM

SLANG_PRELUDE_EXTERN_C_START

// Unary 
float F32_ceil(float f);
float F32_floor(float f);
float F32_round(float f);
float F32_sin(float f);
float F32_cos(float f);
float F32_tan(float f);
float F32_asin(float f);
float F32_acos(float f);
float F32_atan(float f);
float F32_sinh(float f);
float F32_cosh(float f);
float F32_tanh(float f);
float F32_log2(float f);
float F32_log(float f);
float F32_log10(float f);
float F32_exp2(float f);
float F32_exp(float f);
float F32_abs(float f);
float F32_trunc(float f);
float F32_sqrt(float f);

bool F32_isnan(float f);
bool F32_isfinite(float f); 
bool F32_isinf(float f);

// Binary
SLANG_FORCE_INLINE float F32_min(float a, float b) { return a < b ? a : b; }
SLANG_FORCE_INLINE float F32_max(float a, float b) { return a > b ? a : b; }
float F32_pow(float a, float b);
float F32_fmod(float a, float b);
float F32_remainder(float a, float b);
float F32_atan2(float a, float b);

float F32_frexp(float x, float* e);
float F32_modf(float x, float* ip);

// Ternary
SLANG_FORCE_INLINE float F32_fma(float a, float b, float c) { return a * b + c; }

SLANG_PRELUDE_EXTERN_C_END

#else

// Unary 
SLANG_FORCE_INLINE float F32_ceil(float f) { return ::ceilf(f); }
SLANG_FORCE_INLINE float F32_floor(float f) { return ::floorf(f); }
SLANG_FORCE_INLINE float F32_round(float f) { return ::roundf(f); }
SLANG_FORCE_INLINE float F32_sin(float f) { return ::sinf(f); }
SLANG_FORCE_INLINE float F32_cos(float f) { return ::cosf(f); }
SLANG_FORCE_INLINE float F32_tan(float f) { return ::tanf(f); }
SLANG_FORCE_INLINE float F32_asin(float f) { return ::asinf(f); }
SLANG_FORCE_INLINE float F32_acos(float f) { return ::acosf(f); }
SLANG_FORCE_INLINE float F32_atan(float f) { return ::atanf(f); }
SLANG_FORCE_INLINE float F32_sinh(float f) { return ::sinhf(f); }
SLANG_FORCE_INLINE float F32_cosh(float f) { return ::coshf(f); }
SLANG_FORCE_INLINE float F32_tanh(float f) { return ::tanhf(f); }
SLANG_FORCE_INLINE float F32_log2(float f) { return ::log2f(f); }
SLANG_FORCE_INLINE float F32_log(float f) { return ::logf(f); }
SLANG_FORCE_INLINE float F32_log10(float f) { return ::log10f(f); }
SLANG_FORCE_INLINE float F32_exp2(float f) { return ::exp2f(f); }
SLANG_FORCE_INLINE float F32_exp(float f) { return ::expf(f); }
SLANG_FORCE_INLINE float F32_abs(float f) { return ::fabsf(f); }
SLANG_FORCE_INLINE float F32_trunc(float f) { return ::truncf(f); }
SLANG_FORCE_INLINE float F32_sqrt(float f) { return ::sqrtf(f); }

SLANG_FORCE_INLINE bool F32_isnan(float f) { return SLANG_PRELUDE_STD isnan(f); }
SLANG_FORCE_INLINE bool F32_isfinite(float f) { return SLANG_PRELUDE_STD isfinite(f); }
SLANG_FORCE_INLINE bool F32_isinf(float f) { return SLANG_PRELUDE_STD isinf(f); }

// Binary
SLANG_FORCE_INLINE float F32_min(float a, float b) { return ::fminf(a, b); }
SLANG_FORCE_INLINE float F32_max(float a, float b) { return ::fmaxf(a, b); }
SLANG_FORCE_INLINE float F32_pow(float a, float b) { return ::powf(a, b); }
SLANG_FORCE_INLINE float F32_fmod(float a, float b) { return ::fmodf(a, b); }
SLANG_FORCE_INLINE float F32_remainder(float a, float b) { return ::remainderf(a, b); }
SLANG_FORCE_INLINE float F32_atan2(float a, float b) { return float(::atan2(a, b)); }

SLANG_FORCE_INLINE float F32_frexp(float x, float* e)
{
    int ei;
    float m = ::frexpf(x, &ei);
    *e = float(ei);
    return m;
}
SLANG_FORCE_INLINE float F32_modf(float x, float* ip)
{
    return ::modff(x, ip);
}

// Ternary
SLANG_FORCE_INLINE float F32_fma(float a, float b, float c) { return ::fmaf(a, b, c); }

#endif

SLANG_FORCE_INLINE float F32_calcSafeRadians(float radians)
{
    // Put 0 to 2pi cycles to cycle around 0 to 1 
	float a = radians * (1.0f /  float(SLANG_PRELUDE_PI * 2));
    // Get truncated fraction, as value in  0 - 1 range
    a = a - F32_floor(a);
    // Convert back to 0 - 2pi range
	return (a * float(SLANG_PRELUDE_PI * 2));
}

SLANG_FORCE_INLINE float F32_rsqrt(float f) { return 1.0f / F32_sqrt(f); }
SLANG_FORCE_INLINE float F32_sign(float f) { return ( f == 0.0f) ? f : (( f < 0.0f) ? -1.0f : 1.0f); } 
SLANG_FORCE_INLINE float F32_frac(float f) { return f - F32_floor(f); }

SLANG_FORCE_INLINE uint32_t F32_asuint(float f) { Union32 u; u.f = f; return u.u; }
SLANG_FORCE_INLINE int32_t F32_asint(float f) { Union32 u; u.f = f; return u.i; }

// ----------------------------- F64 -----------------------------------------

SLANG_FORCE_INLINE double F64_calcSafeRadians(double radians);

#ifdef SLANG_LLVM

SLANG_PRELUDE_EXTERN_C_START

// Unary 
double F64_ceil(double f);
double F64_floor(double f);
double F64_round(double f);
double F64_sin(double f);
double F64_cos(double f);
double F64_tan(double f);
double F64_asin(double f);
double F64_acos(double f);
double F64_atan(double f);
double F64_sinh(double f);
double F64_cosh(double f);
double F64_tanh(double f);
double F64_log2(double f);
double F64_log(double f);
double F64_log10(float f);
double F64_exp2(double f);
double F64_exp(double f);
double F64_abs(double f);
double F64_trunc(double f);
double F64_sqrt(double f);

bool F64_isnan(double f);
bool F64_isfinite(double f);
bool F64_isinf(double f);

// Binary
SLANG_FORCE_INLINE double F64_min(double a, double b) { return a < b ? a : b; }
SLANG_FORCE_INLINE double F64_max(double a, double b) { return a > b ? a : b; }
double F64_pow(double a, double b);
double F64_fmod(double a, double b);
double F64_remainder(double a, double b);
double F64_atan2(double a, double b);

double F64_frexp(double x, double* e);

double F64_modf(double x, double* ip);

// Ternary
SLANG_FORCE_INLINE double F64_fma(double a, double b, double c) { return a * b + c; }

SLANG_PRELUDE_EXTERN_C_END

#else // SLANG_LLVM

// Unary 
SLANG_FORCE_INLINE double F64_ceil(double f) { return ::ceil(f); }
SLANG_FORCE_INLINE double F64_floor(double f) { return ::floor(f); }
SLANG_FORCE_INLINE double F64_round(double f) { return ::round(f); }
SLANG_FORCE_INLINE double F64_sin(double f) { return ::sin(f); }
SLANG_FORCE_INLINE double F64_cos(double f) { return ::cos(f); }
SLANG_FORCE_INLINE double F64_tan(double f) { return ::tan(f); }
SLANG_FORCE_INLINE double F64_asin(double f) { return ::asin(f); }
SLANG_FORCE_INLINE double F64_acos(double f) { return ::acos(f); }
SLANG_FORCE_INLINE double F64_atan(double f) { return ::atan(f); }
SLANG_FORCE_INLINE double F64_sinh(double f) { return ::sinh(f); }
SLANG_FORCE_INLINE double F64_cosh(double f) { return ::cosh(f); }
SLANG_FORCE_INLINE double F64_tanh(double f) { return ::tanh(f); }
SLANG_FORCE_INLINE double F64_log2(double f) { return ::log2(f); }
SLANG_FORCE_INLINE double F64_log(double f) { return ::log(f); }
SLANG_FORCE_INLINE double F64_log10(float f) { return ::log10(f); }
SLANG_FORCE_INLINE double F64_exp2(double f) { return ::exp2(f); }
SLANG_FORCE_INLINE double F64_exp(double f) { return ::exp(f); }
SLANG_FORCE_INLINE double F64_abs(double f) { return ::fabs(f); }
SLANG_FORCE_INLINE double F64_trunc(double f) { return ::trunc(f); }
SLANG_FORCE_INLINE double F64_sqrt(double f) { return ::sqrt(f); }


SLANG_FORCE_INLINE bool F64_isnan(double f) { return SLANG_PRELUDE_STD isnan(f); }
SLANG_FORCE_INLINE bool F64_isfinite(double f) { return SLANG_PRELUDE_STD isfinite(f); }
SLANG_FORCE_INLINE bool F64_isinf(double f) { return SLANG_PRELUDE_STD isinf(f); }

// Binary
SLANG_FORCE_INLINE double F64_min(double a, double b) { return ::fmin(a, b); }
SLANG_FORCE_INLINE double F64_max(double a, double b) { return ::fmax(a, b); }
SLANG_FORCE_INLINE double F64_pow(double a, double b) { return ::pow(a, b); }
SLANG_FORCE_INLINE double F64_fmod(double a, double b) { return ::fmod(a, b); }
SLANG_FORCE_INLINE double F64_remainder(double a, double b) { return ::remainder(a, b); }
SLANG_FORCE_INLINE double F64_atan2(double a, double b) { return ::atan2(a, b); }

SLANG_FORCE_INLINE double F64_frexp(double x, double* e)
{
    int ei;
    double m = ::frexp(x, &ei);
    *e = float(ei);
    return m;
}

SLANG_FORCE_INLINE double F64_modf(double x, double* ip)
{
    return ::modf(x, ip);
}

// Ternary
SLANG_FORCE_INLINE double F64_fma(double a, double b, double c) { return ::fma(a, b, c); }

#endif // SLANG_LLVM

SLANG_FORCE_INLINE double F64_rsqrt(double f) { return 1.0 / F64_sqrt(f); }
SLANG_FORCE_INLINE double F64_sign(double f) { return (f == 0.0) ? f : ((f < 0.0) ? -1.0 : 1.0); }
SLANG_FORCE_INLINE double F64_frac(double f) { return f - F64_floor(f); }

SLANG_FORCE_INLINE void F64_asuint(double d, uint32_t* low, uint32_t* hi)
{
    Union64 u;
    u.d = d;
    *low = uint32_t(u.u);
    *hi = uint32_t(u.u >> 32);
}

SLANG_FORCE_INLINE void F64_asint(double d, int32_t* low, int32_t* hi)
{
    Union64 u;
    u.d = d;
    *low = int32_t(u.u);
    *hi = int32_t(u.u >> 32);
}

SLANG_FORCE_INLINE double F64_calcSafeRadians(double radians)
{
    // Put 0 to 2pi cycles to cycle around 0 to 1 
	double a = radians * (1.0f /  (SLANG_PRELUDE_PI * 2));
    // Get truncated fraction, as value in  0 - 1 range
    a = a - F64_floor(a);
    // Convert back to 0 - 2pi range
	return (a * (SLANG_PRELUDE_PI * 2));
}

// ----------------------------- I32 -----------------------------------------

SLANG_FORCE_INLINE int32_t I32_abs(int32_t f) { return (f < 0) ? -f : f; }

SLANG_FORCE_INLINE int32_t I32_min(int32_t a, int32_t b) { return a < b ? a : b; }
SLANG_FORCE_INLINE int32_t I32_max(int32_t a, int32_t b) { return a > b ? a : b; }

SLANG_FORCE_INLINE float I32_asfloat(int32_t x) { Union32 u; u.i = x; return u.f; }
SLANG_FORCE_INLINE uint32_t I32_asuint(int32_t x) { return uint32_t(x); }
SLANG_FORCE_INLINE double I32_asdouble(int32_t low, int32_t hi )
{
    Union64 u;
    u.u = (uint64_t(hi) << 32) | uint32_t(low);
    return u.d;
}

// ----------------------------- U32 -----------------------------------------

SLANG_FORCE_INLINE uint32_t U32_abs(uint32_t f) { return f; }

SLANG_FORCE_INLINE uint32_t U32_min(uint32_t a, uint32_t b) { return a < b ? a : b; }
SLANG_FORCE_INLINE uint32_t U32_max(uint32_t a, uint32_t b) { return a > b ? a : b; }

SLANG_FORCE_INLINE float U32_asfloat(uint32_t x) { Union32 u; u.u = x; return u.f; }
SLANG_FORCE_INLINE uint32_t U32_asint(int32_t x) { return uint32_t(x); } 

SLANG_FORCE_INLINE double U32_asdouble(uint32_t low, uint32_t hi)
{
    Union64 u;
    u.u = (uint64_t(hi) << 32) | low;
    return u.d;
}


SLANG_FORCE_INLINE uint32_t U32_countbits(uint32_t v)
{
#if SLANG_GCC_FAMILY && !defined(SLANG_LLVM)
    return __builtin_popcount(v);
#elif SLANG_PROCESSOR_X86_64 && SLANG_VC
    return __popcnt(v);
#else     
    uint32_t c = 0;
    while (v)
    {
        c++;
        v &= v - 1;
    }
    return c;
#endif
}

// ----------------------------- U64 -----------------------------------------

SLANG_FORCE_INLINE uint64_t U64_abs(uint64_t f) { return f; }

SLANG_FORCE_INLINE uint64_t U64_min(uint64_t a, uint64_t b) { return a < b ? a : b; }
SLANG_FORCE_INLINE uint64_t U64_max(uint64_t a, uint64_t b) { return a > b ? a : b; }

// TODO(JS): We don't define countbits for 64bit in stdlib currently.
// It's not clear from documentation if it should return 32 or 64 bits, if it exists. 
// 32 bits can always hold the result, and will be implicitly promoted. 
SLANG_FORCE_INLINE uint32_t U64_countbits(uint64_t v)
{
#if SLANG_GCC_FAMILY && !defined(SLANG_LLVM)   
    return uint32_t(__builtin_popcountl(v));
#elif SLANG_PROCESSOR_X86_64 && SLANG_VC
    return uint32_t(__popcnt64(v));
#else     
    uint32_t c = 0;
    while (v)
    {
        c++;
        v &= v - 1;
    }
    return c;
#endif
}

// ----------------------------- I64 -----------------------------------------

SLANG_FORCE_INLINE int64_t I64_abs(int64_t f) { return (f < 0) ? -f : f; }

SLANG_FORCE_INLINE int64_t I64_min(int64_t a, int64_t b) { return a < b ? a : b; }
SLANG_FORCE_INLINE int64_t I64_max(int64_t a, int64_t b) { return a > b ? a : b; }


// ----------------------------- Interlocked ---------------------------------

#if SLANG_LLVM

#else // SLANG_LLVM

#   ifdef _WIN32
#       include <intrin.h>
#   endif

void InterlockedAdd(uint32_t* dest, uint32_t value, uint32_t* oldValue)
{
#   ifdef _WIN32
    *oldValue = _InterlockedExchangeAdd((long*)dest, (long)value);
#   else
    *oldValue = __sync_fetch_and_add(dest, value);
#   endif
}

#endif // SLANG_LLVM


// ----------------------- fmod --------------------------
SLANG_FORCE_INLINE float _slang_fmod(float x, float y)
{
    return F32_fmod(x, y);
}
SLANG_FORCE_INLINE double _slang_fmod(double x, double y)
{
    return F64_fmod(x, y);
}

#ifdef SLANG_PRELUDE_NAMESPACE
} 
#endif

#endif


static const int kSlangTorchTensorMaxDim = 5;

struct TensorView
{
    uint8_t* data;
    uint32_t strides[kSlangTorchTensorMaxDim];
    uint32_t sizes[kSlangTorchTensorMaxDim];
    uint32_t dimensionCount;
};


TensorView make_tensor_view(torch::Tensor val, const char* name, torch::ScalarType targetScalarType)
{
    // Convert device and scalar types.
    if (!val.device().is_cuda())
        val = val.to(torch::kCUDA);
    if (val.dtype() != targetScalarType)
        val = val.to(targetScalarType);

    TensorView res = {};
    res.dimensionCount = val.dim();
    res.data = nullptr;
    size_t elementSize = 4;

    switch (val.scalar_type())
    {
    case torch::kInt8:
    case torch::kUInt8:
        elementSize = 1;
        res.data = (uint8_t*)val.data_ptr<uint8_t>();
        break;
    case torch::kBFloat16:
        elementSize = 2;
        res.data = (uint8_t*)val.data_ptr<torch::BFloat16>();
        break;
    case torch::kInt16:
        elementSize = 2;
        res.data = (uint8_t*)val.data_ptr<int16_t>();
        break;
    case torch::kFloat32:
        elementSize = 4;
        res.data = (uint8_t*)val.data_ptr<float>();
        break;
    case torch::kInt32:
        elementSize = 4;
        res.data = (uint8_t*)val.data_ptr<int32_t>();
        break;
    case torch::kFloat64:
        elementSize = 8;
        res.data = (uint8_t*)val.data_ptr<double>();
        break;
    case torch::kInt64:
        elementSize = 8;
        res.data = (uint8_t*)val.data_ptr<int64_t>();
        break;
    }

    if (val.dim() > kSlangTorchTensorMaxDim)
        throw std::runtime_error(std::string(name).append(": number of dimensions exceeds limit (").append(std::to_string(kSlangTorchTensorMaxDim)).append(")").c_str());

    for (int i = 0; i < val.dim(); ++i)
    {
        res.strides[i] = val.stride(i) * elementSize;
        res.sizes[i] = val.size(i);
    }
    if (!res.data)
        throw std::runtime_error(std::string(name).append(": data pointer is invalid.").c_str());
    return res;
}

size_t slangGetCudaKernelSharedMemSize(const void* func)
{
    cudaFuncAttributes attr = {};
    cudaFuncGetAttributes(&attr, func);
    AT_CUDA_CHECK(cudaGetLastError());
    return attr.sharedSizeBytes;
}

#define SLANG_PRELUDE_EXPORT


#line 6 "D:/git_repo/slang-python/tests/smoke.slang"
struct ReturnValue_0
{
    torch::Tensor v0_0;
    torch::Tensor v1_0;
};


#line 19
SLANG_PRELUDE_EXPORT
void myKernel(TensorView _0, TensorView _1, float _2);


#line 284 "diff.meta.slang"
torch::Tensor TorchTensor_zerosLike_0(torch::Tensor other_0)
{
    torch::Tensor result_0 = torch::empty_like(other_0, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
    (result_0).zero_();
    return result_0;
}


#line 28 "D:/git_repo/slang-python/tests/smoke.slang"
SLANG_PRELUDE_EXPORT
std::tuple<torch::Tensor, torch::Tensor> runCompute(std::tuple<torch::Tensor, float> inputs_0)
{

#line 28
    torch::Tensor _S1 = std::get<int(0)>(inputs_0);

#line 28
    float _S2 = std::get<int(1)>(inputs_0);


    torch::Tensor outValues_0 = TorchTensor_zerosLike_0(_S1);
    uint32_t _S3 = ((_S1).size((0U)));

#line 32
    uint32_t _S4 = ((_S1).size((1U)));

#line 32
    Vector<uint32_t, 3> _S5 = Vector<uint32_t, 3>(_S3, _S4, 1U);

#line 32
    Vector<uint32_t, 3> _S6 = Vector<uint32_t, 3>(1U, 1U, 1U);

#line 32
    TensorView _S7 = make_tensor_view(outValues_0, "outValues", torch::kFloat32);

#line 32
    TensorView _S8 = make_tensor_view(_S1, "_S1", torch::kFloat32);

#line 32
    FixedArray<void *, 3>  _S9;

#line 32
    *(&_S9[int(0)]) = &_S8;

#line 32
    TensorView _S10 = _S7;

#line 32
    *(&_S9[int(1)]) = &_S10;

#line 32
    float _S11 = _S2;

#line 32
    *(&_S9[int(2)]) = &_S11;

#line 32
    AT_CUDA_CHECK(cudaLaunchKernel((const void*)(myKernel), slang_bit_cast<dim3>(_S6), slang_bit_cast<dim3>(_S5), &_S9[int(0)], slangGetCudaKernelSharedMemSize((const void*)(myKernel)), ((cudaStream_t)at::cuda::getCurrentCUDAStream())));
    ReturnValue_0 retVal_0;
    (&retVal_0)->v0_0 = _S1;
    (&retVal_0)->v1_0 = outValues_0;
    return std::make_tuple(retVal_0.v0_0, retVal_0.v1_0);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("runCompute", &runCompute, "runCompute");
}
