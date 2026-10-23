#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "holistic_motion/kinematics/fep/FEPKinematics.h"

using namespace holistic_motion::robotics;

int main(int argc, char **argv) {
    const std::string backend_name = argc > 1 ? argv[1] : "CPU";
    const FEPBackend backend = backend_name == "CPU"    ? FEPBackend::CPU
                               : backend_name == "CUDA" ? FEPBackend::CUDA
                                                        : FEPBackend::AUTO;
    if (argc > 2 ||
        (backend_name != "CPU" && backend_name != "CUDA" && backend_name != "AUTO")) {
        std::cerr << "usage: fep_batch_benchmark [CPU|CUDA|AUTO]\n";
        return 2;
    }
    if (backend == FEPBackend::CUDA && !FEPKinematics::HasCudaBackend()) {
        std::cerr << "CUDA benchmark requires an available native CUDA backend\n";
        return 2;
    }
    std::vector<JointNode> nodes;
    for (int joint = 0; joint < 7; ++joint) {
        nodes.emplace_back(SE3d(Eigen::Vector3d(0.02 * joint, -0.01, 0.1),
                                SO3d(0.03 * joint, -0.02, 0.04)),
                           Eigen::Vector3d::Unit(joint % 3), JointType::REVOLUTE, -2.0,
                           2.0);
    }
    nodes.emplace_back(SE3d(Eigen::Vector3d(0.03, -0.02, 0.08), SO3d()),
                       Eigen::Vector3d::Zero(), JointType::FIXED, 0.0, 0.0);
    FEPKinematics solver(nodes);
    solver.SetTCP(SE3d(Eigen::Vector3d(0.04, 0.02, -0.03), SO3d(0.2, -0.1, 0.3)));
    std::mt19937 generator(7);
    std::uniform_real_distribution<double> distribution(-1.5, 1.5);
    std::cout
        << "batch_size,calls_per_sample,median_us,max_scalar_error,checksum,backend\n"
        << std::setprecision(12);
    for (int batch_size : {1, 32, 128, 1024}) {
        Eigen::MatrixXd joints(batch_size, 7);
        for (Eigen::Index row = 0; row < joints.rows(); ++row)
            for (Eigen::Index col = 0; col < joints.cols(); ++col)
                joints(row, col) = distribution(generator);
        std::vector<Eigen::Matrix4d> poses;
        for (int warmup = 0; warmup < 3; ++warmup)
            if (!solver.ForwardBatch(joints, backend, poses))
                return 1;
        const int calls =
            backend == FEPBackend::CPU ? std::max(1, 16384 / batch_size) : 100;
        std::vector<double> elapsed;
        for (int sample = 0; sample < 7; ++sample) {
            const auto started = std::chrono::steady_clock::now();
            for (int call = 0; call < calls; ++call)
                if (!solver.ForwardBatch(joints, backend, poses))
                    return 1;
            elapsed.push_back(std::chrono::duration<double, std::micro>(
                                  std::chrono::steady_clock::now() - started)
                                  .count() /
                              calls);
        }
        double max_error = 0.0;
        double checksum = 0.0;
        for (int row = 0; row < batch_size; ++row) {
            SE3d scalar;
            if (!solver.GetFK(joints.row(row).transpose(), scalar))
                return 1;
            max_error = std::max(
                max_error, (poses[row] - scalar.GetTransform()).cwiseAbs().maxCoeff());
            checksum += poses[row].sum();
        }
        if (max_error > 1e-12)
            return 1;
        std::sort(elapsed.begin(), elapsed.end());
        std::cout << batch_size << ',' << calls << ',' << elapsed[3] << ',' << max_error
                  << ',' << checksum << ',' << backend_name << '\n';
    }
}
