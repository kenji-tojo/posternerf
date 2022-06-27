/*
 * The viewer logic is adapted from https://github.com/sxyu/volrend
 */

#pragma once

#include <vector>
#include <array>
#include <string>
#include <memory>

#include <cuda_runtime.h>

#include "glad/glad.h"
#include "GLFW/glfw3.h"

#include "octree.h"
#include "render_options.h"
#include "viewer/palette.h"


namespace plnoct {

struct Camera;

class Viewer {
public:
    Viewer();
    ~Viewer();

    void launch(int width, int height);

    void set_output_dir(std::string output_dir) {
        if (output_dir.empty())
            return;
        if (output_dir.back() != '/')
            output_dir.push_back('/');
        m_output_dir = std::move(output_dir);
    }

    void set_tree(size_t num_nodes, int data_dim, int basis_dim,
                  const __half *data, const int32_t *child,
                  const float *shift, const float *scale);

    void set_palette(const std::vector<float> &colors_rgb) { m_palette.set_colors_rgb(colors_rgb); }

    void _mouse_button_callback(GLFWwindow *window, int button, int action, int mods);
    void _cursor_pos_callback(GLFWwindow* window, double x, double y);
    void _scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
    void _window_size_callback(GLFWwindow *window, int width, int height);
private:
    std::string m_output_dir;

    std::unique_ptr<Camera> m_camera;
    std::unique_ptr<Tree> m_tree;
    RenderOptions m_options;

    Palette m_palette;

    GLFWwindow *m_window = nullptr;
    int m_buf_index = 0;
    unsigned int m_FBO[2]{ 0 };
    unsigned int m_RBO[4]{ 0 };

    // CUDA resources
    std::array<cudaGraphicsResource_t, 4> m_CGR{ nullptr };
    std::array<cudaArray_t, 4> m_CA{ };
    cudaStream_t m_stream = nullptr;

    bool m_started = false;

    void draw_gui();
    void start();
    void resize(int width, int height);
    void render();
};

} // namespace plnoct