#pragma once

#include <vector>


namespace plnoct {

struct Labels {
    static constexpr size_t MAX_SIZE = 200;

    float reference_rgb[MAX_SIZE * 3];
    float reference_lab[MAX_SIZE * 3];
    float recolored_rgb[MAX_SIZE * 3];

    size_t num_labels;
};

} // namespace plnoct

namespace plnoct {

struct Palette {
public:
    static constexpr size_t MAX_SIZE = 10;

    explicit Palette() = default;

    void set_colors_rgb(const std::vector<float> &colors_rgb) {
        set_reference_rgb(colors_rgb);
        m_recolored_rgb = m_reference_rgb;
        update_recolored_labels();
    }

    [[nodiscard]] inline size_t num_steps() const { return m_num_steps; }
    inline void set_num_steps(size_t num_steps) {
        if (m_num_steps == num_steps || m_reference_rgb.empty())
            return;
        m_num_steps = num_steps;
        set_reference_rgb(m_reference_rgb);
        update_recolored_labels();
    }

    void write_recolored_rgb(size_t index, const float rgb[3]) {
        if (index >= size())
            return;
        m_recolored_rgb[index * 3 + 0] = rgb[0];
        m_recolored_rgb[index * 3 + 1] = rgb[1];
        m_recolored_rgb[index * 3 + 2] = rgb[2];
        update_recolored_labels();
    }
    void read_recolored_rgb(size_t index, float rgb[3]) {
        if (index >= size())
            return;
        rgb[0] = m_recolored_rgb[index * 3 + 0];
        rgb[1] = m_recolored_rgb[index * 3 + 1];
        rgb[2] = m_recolored_rgb[index * 3 + 2];
    }
    void reset_recolored_rgb() {
        m_recolored_rgb = m_reference_rgb;
        update_recolored_labels();
    }

    [[nodiscard]] inline bool needs_update() {
        if (m_needs_update) {
            m_needs_update = false;
            return true;
        }
        return false;
    }

    [[nodiscard]] inline size_t size() const { return m_reference_rgb.size() / 3; }
    [[nodiscard]] inline const Labels &labels() const { return m_labels; }

private:
    std::vector<float> m_reference_rgb; // reference palette colors in RGB
    std::vector<float> m_recolored_rgb; // recolored palette (same as the reference by default)
    size_t m_num_steps = 2;

    bool m_needs_update = false;
    Labels m_labels{ };

    void update_recolored_labels();
    void set_reference_rgb(std::vector<float> colors_rgb);
};

} // namespace plnoct