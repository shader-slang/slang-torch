#include <cuda.h>
#include <mma.h>


// Performs an inline matmul (Out = In x Wt) assuming:
// I = WxC row_major
// W = CxO col_major
//
// Each warp is responsible for multiplying its input (which is WxC where C is the feature dimension)
// by the CxO matrix. The output is WxO.
// 
// C must be a multiple of 16
// 'In' matrix must be row-major, 'Wt' matrix must be col-major, 'Out' matrix will be row-major.
// 
// We assume that all threads in the warp are active. WMMA ops don't work otherwise.
// 
template<int W, int C, int O, int M, int N, int K>
__device__ void wmma_inline_matmul(float *in, float *wt, float *out)
{
   // Leading dimensions. Packed with no transpositions.
   int lda = C;
   int ldb = C;
 
   // Declare the fragments
   nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, M, N, K, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major> a_frag;
   nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, M, N, K, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major> b_frag;
   nvcuda::wmma::fragment<nvcuda::wmma::accumulator, M, N, K, float> acc_frag;

   // We'll go over each tile of MxN for the output matrix. 
   // (all these loops are unrollable since C is known at compile time)
   // 
   // For example, if W=O=C=32, M=16, N=16, K=8, the outer two loops will loop 4 times total, 
   // at each step computing a 16x16 tile of the output 32x32 matrix.
   // For each of those tiles, the inner loop will loop twice, accumulating the effect of 
   // a 16x8 tile of 'In' and a 8x16 tile of 'Wt'.
   // 
   
   #pragma unroll
   for (int ti = 0; ti < W; ti += M)
   {
      #pragma unroll
      for (int tj = 0; tj < O; tj += N)
      {
         // Reset the accumulator fragment
         nvcuda::wmma::fill_fragment(acc_frag, 0.0f);

         // Go over each tile of MxK sub-tile for the input matrix and corresponding 
         // KxN sub-tile for the weight matrix.
         // 
         #pragma unroll
         for (int tk = 0; tk < C; tk += K)
         {
            // Load the inputs
            nvcuda::wmma::load_matrix_sync(a_frag, in + tk + ti * lda, lda);
            nvcuda::wmma::load_matrix_sync(b_frag, wt + tk + tj * ldb, ldb);

            #pragma unroll
            for (int t = 0; t < a_frag.num_elements; t++)
               a_frag.x[t] =  nvcuda::wmma::__float_to_tf32(a_frag.x[t]);

            #pragma unroll
            for (int t = 0; t < b_frag.num_elements; t++)
               b_frag.x[t] =  nvcuda::wmma::__float_to_tf32(b_frag.x[t]);

            // Perform the matrix multiplication
            nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
         }

         // Store the output tile.
         nvcuda::wmma::store_matrix_sync(out + tj + ti * O, acc_frag, O, nvcuda::wmma::mem_row_major);
      }
   } 

}
