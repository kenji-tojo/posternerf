#pragma once

#include <cstdint>
#include <vector>

#include <cuda_fp16.h>

namespace plnoct {

struct Tree {
public:
    Tree(size_t num_nodes,
         int data_dim,
         int basis_dim,
         const __half *data,
         const int32_t *child,
         const float shift[3],
         const float scale[3]);

    ~Tree();

    [[nodiscard]] const std::vector<float> &leaves_xyz() const { return m_leaves_xyz; }
    [[nodiscard]] int data_dim() const { return m_data_dim; }
    [[nodiscard]] int basis_dim() const { return m_basis_dim; }
    [[nodiscard]] const __half *data_CUDA() const { return m_data_CUDA; }
    [[nodiscard]] const int32_t *child_CUDA() const { return m_child_CUDA; }
    [[nodiscard]] const float *shift() const { return m_shift; }
    [[nodiscard]] const float *scale() const { return m_scale; }

private:
    std::vector<float> m_leaves_xyz;
    int m_data_dim = 0;
    int m_basis_dim = 0;
    float m_shift[3]{ 0 };
    float m_scale[3]{ 1 };

    __half *m_data_CUDA = nullptr;
    int32_t *m_child_CUDA = nullptr;

    void collect_leaves(int data_dim,
                        const __half *data,
                        const int32_t *child);
};

} // namespace plnoct