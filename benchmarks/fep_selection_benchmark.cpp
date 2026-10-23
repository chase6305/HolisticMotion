#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "FEPSelection.h"

using namespace holistic_motion::robotics::fep_internal;

int main(int argc, char **argv) {
    const std::string backend = argc > 1 ? argv[1] : "CPU";
    if (argc > 2 || (backend != "CPU" && backend != "CUDA")) {
        std::cerr << "usage: fep_selection_benchmark [CPU|CUDA]\n";
        return 2;
    }
    if (backend == "CUDA") {
#ifdef HOLISTICMOTION_HAS_CUDA
        if (!CudaBackendAvailable()) {
            std::cerr << "CUDA device unavailable\n";
            return 2;
        }
#else
        std::cerr << "CUDA was not compiled\n";
        return 2;
#endif
    }
    const Eigen::VectorXd seed = Eigen::VectorXd::LinSpaced(7, -0.3, 0.3);
    std::mt19937 generator(82);
    std::uniform_real_distribution<double> angle(-25.0, 25.0);
    std::cout
        << "candidates,calls_per_sample,median_us,max_cpu_error,checksum,backend\n"
        << std::setprecision(12);
    for (int count : {1, 32, 129, 1024}) {
        std::vector<Eigen::VectorXd> candidates;
        for (int row = 0; row < count; ++row) {
            Eigen::VectorXd candidate(7);
            for (int joint = 0; joint < 7; ++joint)
                candidate[joint] = angle(generator);
            candidates.push_back(std::move(candidate));
        }
        const auto reference = ScoreCandidatesCPU(candidates, seed);
        std::vector<double> scores;
        const auto query = [&]() {
#ifdef HOLISTICMOTION_HAS_CUDA
            if (backend == "CUDA")
                return ScoreCandidatesCUDA(candidates, seed, scores);
#endif
            scores = ScoreCandidatesCPU(candidates, seed);
            return scores.size() == candidates.size();
        };
        for (int warmup = 0; warmup < 3; ++warmup)
            if (!query())
                return 1;
        const int calls = backend == "CUDA" ? 100 : std::max(1, 16384 / count);
        std::vector<double> elapsed;
        for (int sample = 0; sample < 7; ++sample) {
            const auto started = std::chrono::steady_clock::now();
            for (int call = 0; call < calls; ++call)
                if (!query())
                    return 1;
            elapsed.push_back(std::chrono::duration<double, std::micro>(
                                  std::chrono::steady_clock::now() - started)
                                  .count() /
                              calls);
        }
        if (scores.size() != reference.size())
            return 1;
        double error = 0.0, checksum = 0.0;
        for (std::size_t row = 0; row < scores.size(); ++row) {
            if (!std::isfinite(scores[row]))
                return 1;
            error = std::max(error, std::abs(scores[row] - reference[row]));
            checksum += scores[row];
        }
        if (error > 1e-11)
            return 1;
        std::sort(elapsed.begin(), elapsed.end());
        std::cout << count << ',' << calls << ',' << elapsed[3] << ',' << error << ','
                  << checksum << ',' << backend << '\n';
    }
}
