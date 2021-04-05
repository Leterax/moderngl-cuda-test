import moderngl_window as mglw
import numpy as np
from moderngl_window.opengl.vao import VAO
import moderngl as mgl
from moderngl_window.scene.camera import KeyboardCamera
from pycuda.compiler import SourceModule
import pycuda.driver as cuda_driver


class CameraWindow(mglw.WindowConfig):
    """Base class with built in 3D camera support"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.camera = KeyboardCamera(self.wnd.keys, aspect_ratio=self.wnd.aspect_ratio)
        self.camera_enabled = True

    def key_event(self, key, action, modifiers):
        keys = self.wnd.keys

        if self.camera_enabled:
            self.camera.key_input(key, action, modifiers)

        if action == keys.ACTION_PRESS:
            if key == keys.C:
                self.camera_enabled = not self.camera_enabled
                self.wnd.mouse_exclusivity = self.camera_enabled
                self.wnd.cursor = not self.camera_enabled
            if key == keys.SPACE:
                self.timer.toggle_pause()

    def mouse_position_event(self, x: int, y: int, dx, dy):
        if self.camera_enabled:
            self.camera.rot_state(-dx, -dy)

    def resize(self, width: int, height: int):
        self.camera.projection.update(aspect_ratio=self.wnd.aspect_ratio)


class MyWindow(CameraWindow):
    resource_dir = "."
    N = 50_000_000
    vsync = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.camera.projection.update(near=0.1, far=10)
        # we have to import them here, they require a running context
        from pycuda import autoinit
        from pycuda.gl import autoinit
        import pycuda.gl as cuda_gl

        # data
        positions = np.random.random((self.N, 3)).astype("f4")
        self.pbuffer = self.ctx.buffer(positions)
        self.cbuffer = self.ctx.buffer(
            reserve=4 * self.N * 4
        )  # self.N*4 floats (4 bytes per float)

        # render program
        self.render_prog = self.load_program("file.glsl")

        # vao
        self.vao = VAO(mode=mgl.POINTS)
        self.vao.buffer(self.pbuffer, "3f", ["in_position"])
        self.vao.buffer(self.cbuffer, "4f", ["in_color"])

        # pycuda stuff
        self.mod = SourceModule(
            """
        __global__ void color_them(float4 *dest, int n, float time)
        {
        const int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= n) {return;}
        float v = (float)tid / (float)n;

        dest[tid] = make_float4(__cosf(time*3.), __sinf(time*2.), __sinf(time), 1.);
        }
        """,
            keep=True,
            cache_dir="./cache",
        )

        self.buffer_cu = cuda_gl.RegisteredBuffer(self.cbuffer._glo)

        self.color_them = self.mod.get_function("color_them")

    def process(self, time):
        # map the buffer
        dst_mapping = self.buffer_cu.map()
        ptr = np.uintp(dst_mapping.device_ptr_and_size()[0])
        self.color_them(
            ptr,
            np.int32(self.N),
            np.float32(time),
            grid=(self.N // 1024 + 1, 1, 1),
            block=(1024, 1, 1),
        )
        cuda_driver.Context.synchronize()
        dst_mapping.unmap()

    def render(self, time: float, frame_time: float):
        self.render_prog["m_proj"].write(self.camera.projection.matrix)
        self.render_prog["m_camera"].write(self.camera.matrix)
        self.process(time)
        self.vao.render(self.render_prog)


MyWindow.run()
