#include "overlay-window.hpp"

OverlayWindow::OverlayWindow(Props const& props) : Window(props)
{

}

void OverlayWindow::Initialize()
{    
    // Modify the window style to be transparent, unclickable and always on top
    SetWindowLongPtr(handle(), GWL_EXSTYLE, WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST);

    // Set transparency
    SetLayeredWindowAttributes(handle(), RGB(0, 0, 0), 0, LWA_COLORKEY);

    // Remove the window border
    SetWindowLongPtr(handle(), GWL_STYLE, WS_POPUP);

    // Update the window with new style
    SetWindowPos(handle(), HWND_TOPMOST, position_.x, position_.y, size_.x, size_.y, SWP_FRAMECHANGED | SWP_NOACTIVATE);
}