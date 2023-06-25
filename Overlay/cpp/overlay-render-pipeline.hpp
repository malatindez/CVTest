#pragma once
#include "direct3d11/direct3d11.hpp"
#include "core/layers/render-pipeline.hpp"
#include "render/common.hpp"
#include "subsystems/render/post-processing.hpp"
#include "render/hdr-to-ldr-layer.hpp"
#include "render/deferred-resolve.hpp"
#include "render/present-swapchain.hpp"
#include "render/renderer.hpp"
using namespace engine;
using namespace core;
class OverlayRenderPipeline : public RenderPipeline
{
public:
    OverlayRenderPipeline(std::shared_ptr<core::Window> window, std::shared_ptr<direct3d::SwapchainRenderTarget> const& output_target);
    OverlayRenderPipeline(OverlayRenderPipeline const&) = delete;
    OverlayRenderPipeline(OverlayRenderPipeline&&) = delete;
    OverlayRenderPipeline& operator=(OverlayRenderPipeline const&) = delete;
    OverlayRenderPipeline& operator=(OverlayRenderPipeline&&) = delete;

    [[nodiscard]] inline render::PerFrame const& per_frame() const noexcept { return per_frame_; }
    [[nodiscard]] inline render::PerFrame& per_frame() noexcept { return per_frame_; }
    [[nodiscard]] inline std::shared_ptr<core::Window> window() noexcept { return window_; }
    void OnRender() override;
    void OnUpdate() override;
    void OnTick(float) override;
    void OnEvent(core::events::Event&) override;
    void WindowSizeChanged(core::math::ivec2 const& size);

private:
    void FrameBegin() override;
    void FrameEnd() override;
    void PostProcess();
    utils::HighResolutionTimer timer;

    core::math::uivec4 sky_color_{ 0, 0, 0, 0 };

    render::PerFrame per_frame_;
    direct3d::DynamicUniformBuffer<engine::render::PerFrame> per_frame_buffer_{};

    std::shared_ptr<direct3d::SwapchainRenderTarget> output_target_;
    D3D11_VIEWPORT viewport_;
    std::shared_ptr<render::PresentSwapchain> swapchain_present_ = nullptr;
    std::shared_ptr<core::Window> window_ = nullptr;
};