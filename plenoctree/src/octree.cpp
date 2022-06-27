#include "octree.h"

#include <iostream>

#include <cuda_runtime.h>


namespace plnoct {

Tree::Tree(size_t num_nodes,
           int data_dim,
           int basis_dim,
           const __half *data,
           const int32_t *child,
           const float shift[3],
           const float scale[3])
        : m_data_dim(data_dim)
        , m_basis_dim(basis_dim)
{
    collect_leaves(data_dim, data, child);

    if (shift != nullptr)
        for (int i=0; i<3; i++) m_shift[i] = shift[i];

    if (scale != nullptr)
        for (int i=0; i<3; i++) m_scale[i] = scale[i];

    const auto data_size = num_nodes * m_data_dim * sizeof(__half);
    cudaMalloc((void **)&m_data_CUDA, data_size);
    cudaMemcpy(m_data_CUDA, data, data_size, cudaMemcpyHostToDevice);

    const auto child_size = num_nodes * sizeof(int32_t);
    cudaMalloc((void **)&m_child_CUDA, child_size);
    cudaMemcpy(m_child_CUDA, child, child_size, cudaMemcpyHostToDevice);
}

Tree::~Tree() {
    if (m_data_CUDA != nullptr)
        cudaFree(m_data_CUDA);

    if (m_child_CUDA != nullptr)
        cudaFree(m_child_CUDA);
}


namespace {
typedef struct {
    float x; float y; float z; float scale; uint32_t ibase;
} node;
}

void Tree::collect_leaves(const int data_dim,
                          const __half *data,
                          const int32_t *child)
{
    std::cout << "collecting tree leaves via tree search" << std::endl;
    m_leaves_xyz.clear();

    std::vector<node> stack;
    stack.push_back({ 0.5f, 0.5f, 0.5f, 0.25f, 0 });
    while (!stack.empty()) {
        auto nd = stack.back();
        stack.pop_back();
        for (int32_t ix=0; ix<2; ++ix)
        for (int32_t iy=0; iy<2; ++iy)
        for (int32_t iz=0; iz<2; ++iz) {
            uint32_t morton = (ix << 2) + (iy << 1) + iz;
            uint32_t index = (nd.ibase << 3) + morton;
            uint32_t skip = child[index];
            node nd_child{
                nd.x + nd.scale * float(ix*2-1),
                nd.y + nd.scale * float(iy*2-1),
                nd.z + nd.scale * float(iz*2-1),
                0.5f * nd.scale,
                nd.ibase + skip
            };
            if (skip == 0) { // leaf node
                float sigma = __half2float(data[data_dim*(index+1)-1]);
                if (sigma >= 1e-2f) {
                    m_leaves_xyz.push_back(nd_child.x);
                    m_leaves_xyz.push_back(nd_child.y);
                    m_leaves_xyz.push_back(nd_child.z);
                }
            } else {
                stack.push_back(nd_child);
            }
        }
    }
}

} // namespace oct