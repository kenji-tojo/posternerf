#include "palette.h"

#include <iostream>

#include "math_utils.h"


namespace {

template<class STL_VECTOR_FLOAT>
void append_inner_labels(std::vector<float> &colors_rgb,
                         STL_VECTOR_FLOAT &&colors_lab,
                         size_t num_steps)
{
    float c[3];
    const auto palette_size = colors_rgb.size() / 3;

    for (int isrt = 0; isrt < palette_size; ++isrt)
    for (int iend = isrt+1; iend < palette_size; ++iend)
    for (int s = 1; s <= num_steps; ++s)
    {
        const float t = (float)s / (float)(num_steps + 1);
        c[0] = (1.0f-t) * colors_rgb[isrt*3+0] + t * colors_rgb[iend*3+0];
        c[1] = (1.0f-t) * colors_rgb[isrt*3+1] + t * colors_rgb[iend*3+1];
        c[2] = (1.0f-t) * colors_rgb[isrt*3+2] + t * colors_rgb[iend*3+2];
        colors_rgb.push_back(c[0]);
        colors_rgb.push_back(c[1]);
        colors_rgb.push_back(c[2]);

        if (!colors_lab.empty()) {
            plnoct::RGB_to_CIELAB(c);
            colors_lab.push_back(c[0]);
            colors_lab.push_back(c[1]);
            colors_lab.push_back(c[2]);
        }
    }
}

} // namespace


namespace plnoct {

void Palette::set_reference_rgb(std::vector<float> colors_rgb) {
    if (colors_rgb.size() / 3 > Palette::MAX_SIZE)
        colors_rgb.resize(Palette::MAX_SIZE * 3);

    m_reference_rgb = colors_rgb;

    std::cout << "palette size: " << this->size() << std::endl;

    std::vector<float> labels_lab = colors_rgb;
    for (int i=0; i<labels_lab.size()/3; ++i)
        RGB_to_CIELAB(labels_lab.data() + i * 3);

    std::vector<float> labels_rgb = std::move(colors_rgb); // will not use colors_rgb below

    ::append_inner_labels(labels_rgb, labels_lab, m_num_steps);
    if (labels_rgb.size() > Labels::MAX_SIZE * 3) {
        std::cerr << "labels will be truncated" << std::endl;
        labels_rgb.resize(Labels::MAX_SIZE * 3);
        labels_lab.resize(Labels::MAX_SIZE * 3);
    }

    for (int i=0; i<labels_rgb.size(); ++i) {
        m_labels.reference_rgb[i] = labels_rgb[i];
        m_labels.reference_lab[i] = labels_lab[i];
    }

    m_labels.num_labels = labels_rgb.size() / 3;
    std::cout << "number of labels: " << m_labels.num_labels << std::endl;

    m_needs_update = true;
}

void Palette::update_recolored_labels() {
    if (m_recolored_rgb.empty())
        return;

    auto labels_rgb = m_recolored_rgb;
    ::append_inner_labels(labels_rgb, std::vector<float>{ }, m_num_steps);

    if (labels_rgb.size() > Labels::MAX_SIZE * 3) {
        std::cerr << "labels will be truncated" << std::endl;
        labels_rgb.resize(Labels::MAX_SIZE * 3);
    }

    for (int i=0; i<labels_rgb.size(); ++i)
        m_labels.recolored_rgb[i] = labels_rgb[i];

    m_needs_update = true;
}

} // namespace plnoct