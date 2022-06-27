#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "octree.h"
#include "viewer.h"
#include "sampling.h"


namespace py = pybind11;

namespace {

template <typename T>
struct pybuffer1D {
public:
    explicit  pybuffer1D(py::array &arr) { set_buffer(arr); }
    explicit  pybuffer1D(py::array_t<T> &arr) { set_buffer(arr); }

    [[nodiscard]] inline T *data() { return static_cast<T *>(m_buf.ptr); }
    [[nodiscard]] inline const T *data() const { return static_cast<const T *>(m_buf.ptr); }
    [[nodiscard]] inline py::ssize_t size() const { return m_buf.size; }

private:
    py::buffer_info m_buf;

    void set_buffer(py::array &arr) {
        if (arr.itemsize() != sizeof(T))
            throw std::runtime_error("itemsize does not match");
        if (arr.ndim() != 1)
            throw std::runtime_error("1D array is expected");

        m_buf = arr.request();
    }
};


void launch_viewer(int data_dim,
                   int basis_dim,
                   py::array &data,
                   py::array_t<int32_t> &child,
                   py::array_t<float> &shift,
                   py::array_t<float> &scale,
                   py::array_t<float> &palette,
                   const std::string &output_dir)
{
    plnoct::Viewer viewer;
    {
        const pybuffer1D<__half> data_buf{ data };
        const pybuffer1D child_buf{ child };
        const pybuffer1D shift_buf{ shift };
        const pybuffer1D scale_buf{ scale };
        const auto num_nodes = (size_t)child_buf.size();
        viewer.set_tree(num_nodes, data_dim, basis_dim,
                        data_buf.data(), child_buf.data(),
                        shift_buf.data(), scale_buf.data());

        const pybuffer1D palette_buf{ palette };
        if (palette_buf.size() > 0) {
            viewer.set_palette(std::vector<float>{ palette_buf.data(), palette_buf.data() + palette_buf.size() });
        }

        viewer.set_output_dir(output_dir);
    }
    viewer.launch(800, 800);
}

inline uint32_t RGB_to_bin_index(const float rgb[3], uint32_t bits_per_channel)
{
    uint32_t index = 0;
    for (int i=0; i<3; ++i) {
        auto c = rgb[i];
        c = std::fmaxf(0.0f, std::fminf(0.999f, c));
        index <<= bits_per_channel;
        index += uint32_t(c * float(1 << bits_per_channel));
    }
    return index;
}

py::tuple compute_RGB_histogram(
        py::array_t<float> &colors_rgb,
        py::array_t<float> &weights,
        int bits_per_channel)
{
    const int bpc = bits_per_channel;
    const int num_bins = 1 << (bpc * 3);
    py::array_t<double> bin_weights{ num_bins };
    py::array_t<float> bin_centers_rgb{ num_bins * 3 };

    pybuffer1D bin_wgt{ bin_weights };
    pybuffer1D bin_cen_rgb{ bin_centers_rgb };
    const pybuffer1D rgb{ colors_rgb };
    const pybuffer1D wgt{ weights };

    for (int i=0; i<num_bins; ++i)
        bin_wgt.data()[i] = 0;

    // compute bin weights
    const auto num_colors = rgb.size() / 3;
    for (int i=0; i<num_colors; ++i) {
        const auto ibin = RGB_to_bin_index(rgb.data() + i * 3, bpc);
        bin_wgt.data()[ibin] += (double)wgt.data()[i];
    }

    // compute the RGB colors at each bin center
    for (int ibin=0; ibin<num_bins; ++ibin) {
        auto code = (uint32_t)ibin;
        for (int i=0; i<3; ++i) {
            const auto c = float(code & ((1 << bpc) - 1)); // lowest bpc bits of the code

            bin_cen_rgb.data()[ibin * 3 + (2 - i)] = (c + 0.5f) / float(1 << bpc);

            code >>= bpc; // use next bpc bits in the following iteration
        }
    }

    return py::make_tuple(bin_weights, bin_centers_rgb.reshape({ num_bins, 3 }));
}

py::tuple sample_radiance(
        const int data_dim,
        const int basis_dim,
        py::array &data,
        py::array_t<int32_t> &child,
        py::array_t<float> &shift,
        py::array_t<float> &scale,
        const py::ssize_t num_dirs_final,
        py::array_t<float> &dirs)
{
    const pybuffer1D<__half> data_buf{ data };
    const pybuffer1D child_buf{ child };
    const pybuffer1D shift_buf{ shift };
    const pybuffer1D scale_buf{ scale };

    plnoct::Tree tree{ (size_t)child_buf.size(), data_dim, basis_dim,
                       data_buf.data(), child_buf.data(),
                       shift_buf.data(), scale_buf.data() };

    const auto num_points = py::ssize_t_cast(tree.leaves_xyz().size() / 3);

    const pybuffer1D dirs_buf{ dirs };
    const auto num_samples_per_dir = dirs_buf.size() / (num_dirs_final * 3);
    const auto num_dirs_total = num_dirs_final * num_samples_per_dir;

    std::cout << "N_sp = " << num_points << " and "
              << "N_dir = " << num_dirs_final << std::endl;

    py::array_t<float> colors_raw{ num_points * num_dirs_total * 3 };
    py::array_t<float>colors_aa{ num_points * num_dirs_final * 3 };
    py::array_t<float> weights{ num_points * num_dirs_final };

    pybuffer1D colors_raw_buf{ colors_raw };
    pybuffer1D colors_aa_buf{ colors_aa };
    pybuffer1D weights_buf{ weights };

    plnoct::sample_radiance_kernel(tree.leaves_xyz(),
                                   num_dirs_final,
                                   num_samples_per_dir,
                                   dirs_buf.data(),
                                   plnoct::TreeSpec{ tree.data_dim(), tree.basis_dim(),
                                                     tree.data_CUDA(), tree.child_CUDA() },
                                   tree.scale(),
                                   colors_raw_buf.data(),
                                   colors_aa_buf.data(),
                                   weights_buf.data());

    constexpr py::ssize_t num_channels = 3;
    return py::make_tuple(colors_raw.reshape({ num_points, num_dirs_total, num_channels }),
                          colors_aa.reshape({ num_points, num_dirs_final, num_channels }),
                          weights.reshape({ num_points, num_dirs_final }));
}

} // namespace

PYBIND11_MODULE(plenoctree, m) {

m.doc() = R"pbdoc(
    plenoctree utility library
)pbdoc";

m.def("launch_viewer", &::launch_viewer, R"pbdoc(
    run plenoctree viewer
)pbdoc");

m.def("compute_RGB_histogram", &::compute_RGB_histogram, R"pbdoc(
    compute the histogram of RGB and weight data
)pbdoc");

m.def("sample_radiance", &::sample_radiance, R"pbdoc(
    generate radiance samples
)pbdoc");

} // PYBIND11_MODULE
