/*
 * The viewer logic is adapted from https://github.com/sxyu/volrend
 */

#include "viewer.h"

#include <cstdlib>
#include <iostream>

#include <cuda_gl_interop.h>

#include "glm/gtc/type_ptr.hpp"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_glfw.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "camera.h"
#include "data_spec.h"


#define GET_VIEWER(window) (static_cast<plnoct::Viewer *>(glfwGetWindowUserPointer(window)))

namespace {

void glfw_update_title(GLFWwindow* window) {
    // static fps counters
    // Source: http://antongerdelan.net/opengl/glcontext2.html
    static double stamp_prev = 0.0;
    static int frame_count = 0;

    const double stamp_curr = glfwGetTime();
    const double elapsed = stamp_curr - stamp_prev;

    if (elapsed > 0.5) {
        stamp_prev = stamp_curr;

        const double fps = (double)frame_count / elapsed;

        char tmp[128];
        sprintf(tmp, "plenoctree viewer - FPS: %.2f", fps);
        glfwSetWindowTitle(window, tmp);
        frame_count = 0;
    }

    frame_count++;
}

void glfw_error_callback(int error, const char* description) {
    std::cerr << description << std::endl;
}

void glfw_mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    GET_VIEWER(window)->_mouse_button_callback(window, button, action, mods);
}
void glfw_cursor_pos_callback(GLFWwindow* window, double x, double y) {
    GET_VIEWER(window)->_cursor_pos_callback(window, x, y);
}
void glfw_scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    GET_VIEWER(window)->_scroll_callback(window, xoffset, yoffset);
}
void glfw_window_size_callback(GLFWwindow *window, int width, int height) {
    GET_VIEWER(window)->_window_size_callback(window, width, height);
}

GLFWwindow *glfw_init(const int width, const int height) {
    glfwSetErrorCallback(::glfw_error_callback);

    if (!glfwInit()) { std::exit(EXIT_FAILURE); }

    glfwWindowHint(GLFW_DEPTH_BITS, GL_TRUE);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    GLFWwindow *window = glfwCreateWindow(width, height, "plenoctree viewer", nullptr, nullptr);

    if (window == nullptr) {
        glfwTerminate();
        std::exit(EXIT_FAILURE);
    }

    glfwMakeContextCurrent(window);

    if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress)) {
        std::cout << "Failed to initialize OpenGL context" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // ignore vsync for now
    glfwSwapInterval(1);

    // only copy r/g/b
    glClearDepth(1.0);
    glDepthFunc(GL_LESS);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_FALSE);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    ImGui_ImplGlfw_InitForOpenGL(window, false);
    char* glsl_version = nullptr;
    ImGui_ImplOpenGL3_Init(glsl_version);
    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    ImGui::GetIO().IniFilename = nullptr;
    glfwSetCharCallback(window, ImGui_ImplGlfw_CharCallback);

    return window;
}

} // namespace

namespace plnoct {

void Viewer::_mouse_button_callback(GLFWwindow *window, int button, int action, int mods) {
    ImGui_ImplGlfw_MouseButtonCallback(window, button, action, mods);
    if (ImGui::GetIO().WantCaptureMouse) return;
    auto &cam = *m_camera;
    double x, y;
    glfwGetCursorPos(window, &x, &y);
    if (action == GLFW_PRESS) {
        const bool SHIFT = mods & GLFW_MOD_SHIFT;
        cam.begin_drag((float)x, (float)y,
                       SHIFT || button == GLFW_MOUSE_BUTTON_MIDDLE,
                       button == GLFW_MOUSE_BUTTON_RIGHT ||
                       (button == GLFW_MOUSE_BUTTON_MIDDLE && SHIFT));
    } else if (action == GLFW_RELEASE) {
        cam.end_drag();
    }
}

void Viewer::_cursor_pos_callback(GLFWwindow *window, double x, double y) {
    m_camera->drag_update((float)x, (float)y);
}

void Viewer::_scroll_callback(GLFWwindow *window, double xoffset, double yoffset) {
    ImGui_ImplGlfw_ScrollCallback(window, xoffset, yoffset);
    if (ImGui::GetIO().WantCaptureMouse) return;
    auto &cam = *m_camera;
    const float speed_fact = 1e-1f;
    cam.move(cam.v_back * ((yoffset < 0.f) ? speed_fact : -speed_fact));
}

void Viewer::_window_size_callback(GLFWwindow *window, int width, int height) {
    glViewport(0, 0, width, height);
    resize(width, height);
}


Viewer::Viewer() : m_camera(std::make_unique<Camera>()) { }

Viewer::~Viewer() {
    if (m_started) {
        // unregister CUDA resources
        for (auto res : m_CGR) {
            if (res != nullptr) { cudaGraphicsUnregisterResource(res); }
        }
        glDeleteRenderbuffers(4, m_RBO);
        glad_glDeleteFramebuffers(2, m_FBO);
        cudaStreamDestroy(m_stream);
    }
    if (m_window != nullptr) {
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        glfwDestroyWindow(m_window);
        glfwTerminate();
    }
}

void Viewer::start() {
    if (m_started) { return; }
    cudaStreamCreateWithFlags(&m_stream, cudaStreamDefault);

    glCreateRenderbuffers(4, m_RBO);
    glCreateFramebuffers(2, m_FBO);

    // Attach RBO to FBO
    const GLenum attachments[]{ GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
    for (int index = 0; index < 2; ++index) {
        glNamedFramebufferRenderbuffer(m_FBO[index], GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, m_RBO[index*2+0]);
        glNamedFramebufferRenderbuffer(m_FBO[index], GL_COLOR_ATTACHMENT1, GL_RENDERBUFFER, m_RBO[index*2+1]);
        glNamedFramebufferDrawBuffers(m_FBO[index], 2, attachments);
    }

    m_started = true;
}

void Viewer::resize(int width, int height) {
    if (width == m_camera->width && height == m_camera->height) { return; }

    start();
    m_camera->width = width;
    m_camera->height = height;

    // unregister resource
    for (auto res : m_CGR) {
        if (res != nullptr) { cudaGraphicsUnregisterResource(res); }
    }

    // resize color buffer
    for (int index = 0; index < 2; ++index) {
        // resize RBO
        glNamedRenderbufferStorage(m_RBO[index*2+0], GL_RGBA8, width, height);
        glNamedRenderbufferStorage(m_RBO[index*2+1], GL_RGBA8, width, height);
        const GLenum attachments[]{ GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
        glNamedFramebufferDrawBuffers(m_FBO[index], 2, attachments);

        // register RBO
        cudaGraphicsGLRegisterImage(&m_CGR[index*2+0], m_RBO[index*2+0], GL_RENDERBUFFER,
                                    cudaGraphicsRegisterFlagsSurfaceLoadStore |
                                    cudaGraphicsRegisterFlagsWriteDiscard);
        cudaGraphicsGLRegisterImage(&m_CGR[index*2+1], m_RBO[index*2+1], GL_RENDERBUFFER,
                                    cudaGraphicsRegisterFlagsSurfaceLoadStore |
                                    cudaGraphicsRegisterFlagsWriteDiscard);
    }

    cudaGraphicsMapResources((int)m_CGR.size(), m_CGR.data(), nullptr);
    for (int index = 0; index < m_CGR.size(); ++index) {
        cudaGraphicsSubResourceGetMappedArray(&m_CA[index], m_CGR[index], 0, 0);
    }
    cudaGraphicsUnmapResources((int)m_CGR.size(), m_CGR.data(), nullptr);
}


inline CameraSpec make_camera_spec(const Camera &cam) {
    return CameraSpec{
        cam.width, cam.height, cam.fx, cam.fy, glm::value_ptr(cam.transform)
    };
}

TreeSpec make_tree_spec(const Tree &tree) {
    return TreeSpec{
        tree.data_dim(), tree.basis_dim(), tree.data_CUDA(), tree.child_CUDA()
    };
}

// implemented in viewer.cu
extern void render_kernel(CameraSpec &&cam,
                          TreeSpec &&tree,
                          const float *shift,
                          const float *scale,
                          Palette &palette,
                          const RenderOptions &opts,
                          cudaArray_t CA[2],
                          cudaStream_t stream);

void Viewer::render() {
    start();
    const GLfloat clear_color[]{
        fixed_render_options::background_brightness,
        fixed_render_options::background_brightness,
        fixed_render_options::background_brightness, 1.0f
    };
    glClearDepth(1.0f);
    glClearNamedFramebufferfv(m_FBO[m_buf_index], GL_COLOR, 0, clear_color);
    glClearNamedFramebufferfv(m_FBO[m_buf_index], GL_COLOR, 1, clear_color);

    m_camera->_update();

    if (m_tree != nullptr) {
        cudaGraphicsMapResources(2, &m_CGR[m_buf_index*2], m_stream);

        render_kernel(make_camera_spec(*m_camera),
                      make_tree_spec(*m_tree),
                      m_tree->shift(),
                      m_tree->scale(),
                      m_palette,
                      m_options,
                      &m_CA[m_buf_index*2], m_stream);

        cudaGraphicsUnmapResources(2, &m_CGR[m_buf_index*2], m_stream);
    }

    glNamedFramebufferReadBuffer(m_FBO[m_buf_index], GL_COLOR_ATTACHMENT0);
    glBlitNamedFramebuffer(m_FBO[m_buf_index],
                           0, 0, 0, m_camera->width,
                           m_camera->height, 0, m_camera->height, m_camera->width, 0,
                           GL_COLOR_BUFFER_BIT, GL_NEAREST);

    m_buf_index ^= 1; // swap between 0 and 1
}

void Viewer::launch(int nw, int nh) {
    if (m_window != nullptr) { return; }

    std::cout << "launch viewer" << std::endl;
    m_window = ::glfw_init(nw, nh);
    int width, height;
    glfwGetFramebufferSize(m_window, &width, &height);
    resize(width, height);

    // set user pointer and callbacks
    glfwSetWindowUserPointer(m_window, this);
    glfwSetMouseButtonCallback(m_window, ::glfw_mouse_button_callback);
    glfwSetCursorPosCallback(m_window, ::glfw_cursor_pos_callback);
    glfwSetScrollCallback(m_window, ::glfw_scroll_callback);
    glfwSetFramebufferSizeCallback(m_window, ::glfw_window_size_callback);

    while (!glfwWindowShouldClose(m_window)) {
        glEnable(GL_DEPTH_TEST);
        ::glfw_update_title(m_window);

        render();
        draw_gui();

        glfwSwapBuffers(m_window);
        glFinish();
        glfwPollEvents();
    }
}

void Viewer::set_tree(size_t num_nodes, int data_dim, int basis_dim,
                      const __half *data, const int32_t *child,
                      const float *shift, const float *scale)
{
    m_tree = std::make_unique<Tree>(num_nodes, data_dim, basis_dim,
                                    data, child,
                                    shift, scale);
}

void Viewer::draw_gui() {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::SetNextWindowPos(ImVec2(20.f, 20.f), ImGuiCond_Once);
    ImGui::SetNextWindowSize(ImVec2(300., 400.f), ImGuiCond_Once);
    ImGui::Begin("GUI");

    if (!m_output_dir.empty()) {
        if (ImGui::Button("save screenshot (png)")) {
            const auto output_path = m_output_dir + "screenshot.png";
            std::cout << "saving screenshot in " << output_path << std::endl;

            int width = m_camera->width, height = m_camera->height;
            std::vector<unsigned char> window_pixels(4 * width * height);
            glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, window_pixels.data());

            std::vector<unsigned char> flipped_pixels(4 * width * height);
            for (int row=0; row<height; ++row) {
                ::memcpy(&flipped_pixels[row * width * 4],
                         &window_pixels[(height - row - 1) * width * 4], 4 * width);
            }

            stbi_write_png(output_path.c_str(),
                           width, height, 4,
                           (const void *)flipped_pixels.data(),
                           width * 4);
        }
    }

    ImGui::SetNextTreeNodeOpen(true, ImGuiCond_Once);
    if (ImGui::CollapsingHeader("Posterization")) {
        ImGui::Checkbox("enable", &m_options.posterize);

        if (ImGui::TreeNode("Recoloring")) {
            static int index = 1; // index starts from 1 rather than 0 in this GUI
            static int index_prev = 0;
            static float rgb[3];

            if (index != index_prev)
                m_palette.read_recolored_rgb(index-1, rgb);
            index_prev = index;
            ImGui::SliderInt("index", &index, 1, (int)m_palette.size());

            ImGui::ColorPicker3("palette color", rgb);

            if (ImGui::Button("apply")) {
                m_palette.write_recolored_rgb(index-1, rgb);
            }

            ImGui::SameLine();
            
            if (ImGui::Button("reset")) {
                m_palette.reset_recolored_rgb();
                index_prev = 0;
            }

            ImGui::TreePop();
        }

        if (ImGui::TreeNode("Label projection")) {
            auto num_steps = (int)m_palette.num_steps();
            ImGui::SliderInt("num steps (d)", &num_steps, 0, 3);
            m_palette.set_num_steps(num_steps);

            ImGui::SliderFloat("step width (D)", &m_options.step_delta, 0.0f, 0.4f);
            ImGui::TreePop();
        }

        if (ImGui::TreeNode("Bilateral filtering")) {
            ImGui::SliderInt("iterations", &m_options.num_bilateral_iter, 1, 5);
            ImGui::SliderInt("radius", &m_options.filter_radius, 1, 4);
            ImGui::SliderFloat("spatial sigma", &m_options.sigma_spatial, 1.0f, 10.0f);
            ImGui::SliderFloat("CIELAB sigma", &m_options.sigma_CIELAB, 1.0f, 10.0f);
            ImGui::TreePop();
        }

        if (ImGui::TreeNode("alpah thresh (lambda)")) {
            ImGui::SliderInt("thresh", &m_options.alpha_thresh, 1, 256);
            ImGui::TreePop();
        }
    }

    ImGui::End();
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}


} // namespace plnoct
