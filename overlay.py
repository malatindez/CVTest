import win32gui
import win32con
import win32api
import ctypes
import moderngl
import numpy as np
from PIL import Image
from ctypes import wintypes
user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32
gdi32 = ctypes.windll.gdi32

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
        win32con.WS_VISIBLE | win32con.WS_EX_TOPMOST,
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
    win32gui.SetLayeredWindowAttributes(hwnd, win32api.RGB(0, 0, 0), 128, win32con.LWA_ALPHA)

    return hwnd, wnd_class_atom

def create_context(hwnd):
    hdc = win32gui.GetDC(hwnd)
    hglrc = wglCreateContext(hdc)
    wglMakeCurrent(hdc, hglrc)
    ctx = moderngl.create_standalone_context()
    return ctx, hdc, hglrc

def wglCreateContext(hdc):
    return ctypes.windll.opengl32.wglCreateContext(hdc)

def wglMakeCurrent(hdc, hglrc):
    return ctypes.windll.opengl32.wglMakeCurrent(hdc, hglrc)

def create_vao(ctx):
    prog = ctx.program(
        vertex_shader="""
        #version 330
        in vec2 in_vert;
        in vec3 in_color;
        out vec3 color;
        void main() {
            gl_Position = vec4(in_vert, 0.0, 1.0);
            color = in_color;
        }
        """,
        fragment_shader="""
        #version 330
        in vec3 color;
        out vec4 fragColor;
        void main() {
            fragColor = vec4(color, 1.0);
        }
        """
    )
    vertices = np.array([
        -0.6, -0.6, 1.0, 0.0, 0.0,
         0.6, -0.6, 0.0, 1.0, 0.0,
         0.0,  0.6, 0.0, 0.0, 1.0,
    ], dtype='f4')
    vbo = ctx.buffer(vertices)
    vao = ctx.simple_vertex_array(prog, vbo, 'in_vert', 'in_color')
    return vao

def draw(ctx, vao, hdc):
    ctx.clear(0.0, 0.0, 0.0, 0.9)
    vao.render(moderngl.TRIANGLES)
    gdi32.SwapBuffers(hdc)

def main():
    hwnd, wnd_class_atom = create_window()
    ctx, hdc, hglrc = create_context(hwnd)

    # Show the window
    user32.ShowWindow(hwnd, win32con.SW_SHOW)

    vao = create_vao(ctx)

    # Message loop
    msg = ctypes.wintypes.MSG()
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
        draw(ctx, vao, hdc)

    # Clean up
    win32gui.UnregisterClass(wnd_class_atom, None)
    wglMakeCurrent(None, None)
    ctypes.windll.opengl32.wglDeleteContext(hglrc)
    win32gui.ReleaseDC(hwnd, hdc)

if __name__ == "__main__":
    main()
