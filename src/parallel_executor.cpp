#include "l2map/parallel_executor.hpp"
#ifndef L2MAP_NO_OPENMP
#include <omp.h>
#endif

namespace l2map {

ParallelExecutor::ParallelExecutor(int n_threads) {
#ifndef L2MAP_NO_OPENMP
    n_threads_ = (n_threads == -1) ? omp_get_max_threads() : n_threads;
#else
    n_threads_ = 1;
    (void)n_threads;
#endif
}

} // namespace l2map
