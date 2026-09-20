#pragma once

#include "common.h"

// Taken from my Project 2 (cis5650_stream_compaction_test, stream_compaction/).
// `scanDevice` is the work-efficient (Blelloch) exclusive scan that the path
// tracer uses to compact terminated rays; it runs in place on a power-of-two
// device array whose tail is zeroed.
namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scan(int n, int *odata, const int *idata);

        int compact(int n, int *odata, const int *idata);

        void scanDevice(int m, int *data);
    }
}
