#include "overlay-render-pipeline.hpp"
#include "components/components.hpp"
#include "subsystems/core/input-layer.hpp"
#include "subsystems/render/skybox-manager.hpp"
#include "core/engine.hpp"

using namespace direct3d;

OverlayRenderPipeline::OverlayRenderPipeline(std::shared_ptr<core::Window> window, std::shared_ptr<SwapchainRenderTarget> const& output_target)
    : core::RenderPipeline(),
    output_target_{ output_target }
{
    viewport_.MinDepth = 0.0f;
    viewport_.MaxDepth = 1.0f;
    viewport_.TopLeftX = 0;
    viewport_.TopLeftY = 0;
    InitImGuiLayer(window);
    window_ = window;
    swapchain_present_ = std::make_shared<render::PresentSwapchain>();
}
void OverlayRenderPipeline::WindowSizeChanged(core::math::ivec2 const& size)
{
    output_target_->SizeResources(size);

    // Set up the viewport.
    viewport_.Width = static_cast<float>(size.x);
    viewport_.Height = static_cast<float>(size.y);
    direct3d::api().devcon4->RSSetViewports(1, &viewport_);
}
void OverlayRenderPipeline::OnRender()
{
    FrameBegin();

    direct3d::api().devcon4->PSSetSamplers(0, 1, &direct3d::states().bilinear_wrap_sampler.ptr());
    direct3d::api().devcon4->PSSetSamplers(1, 1, &direct3d::states().anisotropic_wrap_sampler.ptr());
    direct3d::api().devcon4->PSSetSamplers(2, 1, &direct3d::states().bilinear_clamp_sampler.ptr());
    direct3d::api().devcon4->PSSetSamplers(3, 1, &direct3d::states().comparison_linear_clamp_sampler.ptr());

    direct3d::api().devcon4->VSSetSamplers(0, 1, &direct3d::states().bilinear_wrap_sampler.ptr());
    direct3d::api().devcon4->VSSetSamplers(1, 1, &direct3d::states().anisotropic_wrap_sampler.ptr());
    direct3d::api().devcon4->VSSetSamplers(2, 1, &direct3d::states().bilinear_clamp_sampler.ptr());
    direct3d::api().devcon4->VSSetSamplers(3, 1, &direct3d::states().comparison_linear_clamp_sampler.ptr());

    direct3d::api().devcon4->CSSetSamplers(0, 1, &direct3d::states().bilinear_wrap_sampler.ptr());
    direct3d::api().devcon4->CSSetSamplers(1, 1, &direct3d::states().anisotropic_wrap_sampler.ptr());
    direct3d::api().devcon4->CSSetSamplers(2, 1, &direct3d::states().bilinear_clamp_sampler.ptr());
    direct3d::api().devcon4->CSSetSamplers(3, 1, &direct3d::states().comparison_linear_clamp_sampler.ptr());

    scene_->FrameBegin();

    api().devcon4->RSSetViewports(1, &viewport_);
    direct3d::api().devcon4->OMSetDepthStencilState(direct3d::states().no_depth_stencil_read, 1);

    imgui_layer_->Begin();
    OnGuiRender();
    imgui_layer_->End();
    // End frame
    FrameEnd();
}
void OverlayRenderPipeline::OnUpdate()
{
    core::RenderPipeline::OnUpdate();
    scene_->Update();
}
void OverlayRenderPipeline::OnTick(float dt)
{
    core::RenderPipeline::OnTick(dt);
    scene_->Tick(dt);
}
void OverlayRenderPipeline::OnEvent(core::events::Event& e)
{
    if (e.type() == core::events::EventType::WindowResize)
    {
        auto& event = static_cast<core::events::WindowResizeEvent&>(e);
        WindowSizeChanged(event.size());
    }
    core::RenderPipeline::OnEvent(e);
}

void OverlayRenderPipeline::FrameBegin()
{
    const core::math::vec4 empty_vec{ 0.0f, 0.0f, 0.0f, 1.0f };

    auto const& camera = scene_->main_camera->camera();
    per_frame_.view = camera.view;
    per_frame_.projection = camera.projection;
    per_frame_.view_projection = camera.view_projection;
    per_frame_.inv_view = camera.inv_view;
    per_frame_.inv_projection = camera.inv_projection;
    per_frame_.inv_view_projection = camera.inv_view_projection;
    per_frame_.screen_resolution = core::math::vec2{ viewport_.Width, viewport_.Height };
    per_frame_.mouse_position = core::math::vec2{ core::InputLayer::instance()->mouse_position() };
    per_frame_.time_now = core::Engine::TimeFromStart();
    per_frame_.time_since_last_frame = timer.elapsed();
    timer.reset();

    per_frame_buffer_.Bind(ShaderType::VertexShader, 0);
    per_frame_buffer_.Bind(ShaderType::HullShader, 0);
    per_frame_buffer_.Bind(ShaderType::DomainShader, 0);
    per_frame_buffer_.Bind(ShaderType::GeometryShader, 0);
    per_frame_buffer_.Bind(ShaderType::PixelShader, 0);
    per_frame_buffer_.Bind(ShaderType::ComputeShader, 0);
    per_frame_buffer_.Update(per_frame_);
}
void OverlayRenderPipeline::PostProcess()
{
}
void OverlayRenderPipeline::FrameEnd()
{
    swapchain_present_->OnFrameEnd(static_cast<direct3d::RenderTargetBase&>(*output_target_));
}