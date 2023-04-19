import win32gui
import win32con
import win32api
import ctypes
from ctypes import wintypes
from OpenGL.GL import (
    GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
    GL_TRIANGLES, GL_TRUE,
    glBegin, glEnd, glClearColor, glClear, glColor3f, glVertex3f, glColorMask
)
from OpenGL.WGL import (
    PIXELFORMATDESCRIPTOR, ChoosePixelFormat, SetPixelFormat, wglCreateContext, wglMakeCurrent, wglDeleteContext
)
user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32
gdi32 = ctypes.windll.gdi32

PFD_DRAW_TO_WINDOW = 0x00000004
PFD_SUPPORT_OPENGL = 0x00000020
PFD_DOUBLEBUFFER = 0x00000001
PFD_TYPE_RGBA = 0
PFD_MAIN_PLANE = 0

def create_window():
    # Define the window class
    class_name = "TransparentOverlay"
    wnd_class = win32gui.WNDCLASS()
    wnd_class.lpszClassName = class_name
    wnd_class.lpfnWndProc = win32gui.DefWindowProc

    # Register the window class
    wnd_class_atom = win32gui.RegisterClass(wnd_class)

    # Create the window
    hwnd = win32gui.CreateWindowEx(
        win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT | win32con.WS_EX_TOPMOST,
        wnd_class_atom,
        "Transparent Overlay",
        win32con.WS_POPUP,
        0,
        0,
        win32api.GetSystemMetrics(win32con.SM_CXSCREEN),
        win32api.GetSystemMetrics(win32con.SM_CYSCREEN),
        None,
        None,
        None,
        None
    )

    # Set the window's transparency
    win32gui.SetLayeredWindowAttributes(hwnd, win32api.RGB(0, 0, 0), 180, win32con.LWA_ALPHA)

    return hwnd, wnd_class_atom

def create_context(hwnd):
    hdc = win32gui.GetDC(hwnd)

    pfd = PIXELFORMATDESCRIPTOR()
    pfd.nSize = ctypes.sizeof(PIXELFORMATDESCRIPTOR)
    pfd.nVersion = 1
    pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER
    pfd.iPixelType = PFD_TYPE_RGBA
    pfd.cColorBits = 32
    pfd.cDepthBits = 24
    pfd.cStencilBits = 8
    pfd.iLayerType = PFD_MAIN_PLANE

    pixel_format = ChoosePixelFormat(hdc, ctypes.byref(pfd))
    SetPixelFormat(hdc, pixel_format, ctypes.byref(pfd))

    hglrc = wglCreateContext(hdc)
    wglMakeCurrent(hdc, hglrc)

    return hdc, hglrc

def draw_triangle():
    glClearColor(0.0, 0.0, 0.0, 0.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glBegin(GL_TRIANGLES)
    glColor3f(1, 0, 0)
    glVertex3f(-0.6, -0.6, 0)
    glColor3f(0, 1, 0)
    glVertex3f(0.6, -0.6, 0)
    glColor3f(0, 0, 1)
    glVertex3f(0, 0.6, 0)
    glEnd()

def main():
    hwnd, wnd_class_atom = create_window()
    hdc, hglrc = create_context(hwnd)

    # Show the window
    win32gui.ShowWindow(hwnd, win32con.SW_SHOW)

    # Message loop
    msg = wintypes.MSG()
    running = True
    while running:
        while user32.PeekMessageW(ctypes.byref(msg), hwnd, 0, 0, win32con.PM_REMOVE) != 0:
            if msg.message == win32con.WM_QUIT:
                running = False
                break
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))

        if not running:
            break

        # Draw the triangle
        draw_triangle()

        # Swap buffers
        gdi32.SwapBuffers(hdc)

    # Clean up
    win32gui.UnregisterClass(wnd_class_atom, None)
    wglMakeCurrent(None, None)
    wglDeleteContext(hglrc)
    win32gui.ReleaseDC(hwnd, hdc)

if __name__ == "__main__":
    main()
