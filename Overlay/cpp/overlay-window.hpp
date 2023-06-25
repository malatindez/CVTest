#pragma once
#include "platform/windows/windows-window.hpp"

class OverlayWindow : public engine::platform::windows::Window
{
public:
    OverlayWindow(engine::core::Window::Props const& props);
    virtual ~OverlayWindow() = default;
    void Initialize();

private:
    OverlayWindow(OverlayWindow&&) = delete;
    OverlayWindow& operator=(OverlayWindow&&) = delete;
    OverlayWindow(OverlayWindow const&) = delete;
    OverlayWindow& operator=(OverlayWindow const&) = delete;
};