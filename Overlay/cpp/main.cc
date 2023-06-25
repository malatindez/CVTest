#include "render/present-swapchain.hpp"
#include "core/engine.hpp"
#include "overlay-window.hpp"
#include "overlay-render-pipeline.hpp"
using namespace engine;
using namespace core;
using namespace math;
using namespace platform;

// the entry point for any Windows program
INT WINAPI wWinMain(HINSTANCE, HINSTANCE, PWSTR, int)
{
    Engine::Init();
    Engine& app = Engine::Get();
#ifndef _DEBUG
    try
    {
#endif
        // Initialize in-engine layers that we need
        auto shader_manager = ShaderManager::instance();
        app.PushLayer(shader_manager);

        std::shared_ptr<OverlayWindow> window = std::make_shared<OverlayWindow>(
            core::Window::Props{
                "Overlay",
                { 1920, 1080 },
                {0, 0}
            });
        window->SetEventCallback(Engine::event_function());
        window->Initialize();
        app.PushLayer(window);

        auto swapchain_render_target = std::make_shared<direct3d::SwapchainRenderTarget>();
        swapchain_render_target->init(window->handle(), window->size());

        auto render_pipeline = std::make_shared<OverlayRenderPipeline>(
            std::static_pointer_cast<core::Window>(window),
            swapchain_render_target
            );

        render_pipeline->WindowSizeChanged(window->size());
        auto scene_ptr = std::make_shared<core::Scene>();
        scene_ptr->main_camera = std::make_unique<CameraController>(&scene_ptr->registry, scene_ptr->);
        scene_ptr->renderer = std::make_unique<render::Renderer>();
        render_pipeline->SetScene(scene_ptr);
        app.SetScene(scene_ptr);

        app.PushLayer(render_pipeline);

        shader_manager = nullptr;
        render_pipeline = nullptr;
        app.Run();
#ifndef _DEBUG
    }
    catch (std::exception e)
    {
        spdlog::critical(e.what());
        spdlog::critical("Exception occurred within the engine. Shutting down.");
        std::this_thread::sleep_for(std::chrono::milliseconds(5000));
        app.Exit();
    }
    catch (...)
    {
        spdlog::critical("Unknown exception occurred within the engine. Shutting down.");
        std::this_thread::sleep_for(std::chrono::milliseconds(5000));
        app.Exit();
    }
#endif
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    Engine::Deinit();
    std::this_thread::sleep_for(std::chrono::milliseconds(750));
    return 0;
}