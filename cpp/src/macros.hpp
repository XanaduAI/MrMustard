#if (_OPENMP >= 202011)
#define MM_VECTORIZE _Pragma("omp simd")
#elif defined(__clang__)
#define MM_VECTORIZE _Pragma("clang loop vectorize(enable)")
#elif defined(__GNUC__)
#define MM_VECTORIZE _Pragma("GCC ivdep")
#endif
