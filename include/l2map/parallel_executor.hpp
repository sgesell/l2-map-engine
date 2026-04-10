#pragma once
#ifndef L2MAP_NO_OPENMP
#include <omp.h>
#endif

namespace l2map {

class ParallelExecutor {
public:
    explicit ParallelExecutor(int n_threads = -1);

    // Execute func(i) for i in [0, n), in parallel.
    // func must be thread-safe (no shared mutable state).
    template <typename Func>
    void parallel_for(int n, Func&& func) const {
        // Dynamic scheduling because element work is heterogeneous —
        // boundary elements have fewer overlapping candidates than interior ones.
#ifndef L2MAP_NO_OPENMP
        #pragma omp parallel for schedule(dynamic, 1) num_threads(n_threads_)
#endif
        for (int i = 0; i < n; ++i) {
            func(i);
        }
    }

    int n_threads() const { return n_threads_; }

private:
    int n_threads_;
};

} // namespace l2map
