import numpy as np
from vispy import app, gloo, use
from PIL import Image
import imageio

# Set the backend to PyQt5
use('pyqt5')

# Define vertex shader with GLSL version 120
vertex_shader = """
attribute vec2 a_position;
varying vec2 v_position;
void main() {
    v_position = a_position;
    gl_Position = vec4(a_position, 0.0, 1.0);
}
"""

# Define fragment shader with GLSL version 120
color_shader = """ 
    precision highp float;

    varying vec2 v_position;

    vec3 hsv_to_rgb(float h, float s, float v) {
        vec3 rgb = clamp(abs(mod(h*6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);
        return v * mix(vec3(1.0), rgb, s);
    }

    void main() {
        float color = 0.5*(v_position.x+1);
        vec3 rgb = hsv_to_rgb(color, 1.0, color < 1.0 ? 1.0 : 0.0);
        gl_FragColor = vec4(rgb, 1.0);
    }
"""
fragment_shader = """
precision highp float;

varying vec2 v_position;
uniform float u_zoom;
uniform vec2 u_offset;
uniform int u_max_iterations;
uniform sampler2D u_texture;


vec3 hsv_to_rgb(float h, float s, float v) {
    vec3 rgb = clamp(abs(mod(h*6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);
    return v * mix(vec3(1.0), rgb, s);
}
vec4 mapColor(float mcol) {
    return vec4(0.5 + 0.5*cos(2.7+mcol*30.0 + vec3(0.6, .0, 1.0)), 1.0);
}

void main() {
    vec2 c = v_position * u_zoom + u_offset;
    vec2 z = vec2(0.0, 0.0);
    int iterations;
    int norm_iterations = 2500;
    for(iterations = 0; iterations < u_max_iterations; iterations++) {
        if (dot(z, z) > 7.0) break;
        z = vec2(
            z.x * z.x - z.y * z.y + c.x,
            2.0 * z.x * z.y + c.y
        );
    }
    
    
    float color = dot(z,z) > 7 ? (float(iterations) - log2(log2(dot(z,z))) + 4.0)*0.0025: 0.0;
    vec3 rgb = mapColor(color).rgb;

    // float color = float(mod(iterations,norm_iterations)) / float(norm_iterations);
    //vec3 rgb = hsv_to_rgb(color, 1.0, iterations < u_max_iterations ? 1.0 : 0.0);
    
    gl_FragColor = vec4(rgb, 1.0);

    //vec2 texture_coords = mod(vec2(z.x, 1.0-z.y), vec2(1.0, 1.0));
    //vec3 tex_color = texture2D(u_texture, texture_coords).rgb;
    // tex_color = iterations < u_max_iterations ? tex_color: vec3(0.0, 0.0, 0.0);
    //gl_FragColor = vec4(tex_color, 1.0);    


}
"""


image_path = 'nahida.jpg'
image = Image.open(image_path)
image_data = np.array(image)


class MandelbrotCanvas(app.Canvas):
    def __init__(self, writer):
        app.Canvas.__init__(self, size=(1100, 1100), title='Mandelbrot Set Zoom Animation')
        self.vertices = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=np.float32)
        self.vbo = gloo.VertexBuffer(self.vertices)
        self.color_texture = gloo.Texture2D(image_data)
        self.program = gloo.Program(vertex_shader, fragment_shader)
        self.program['a_position'] = self.vbo
        self.program['u_texture'] = self.color_texture
        
        gloo.set_clear_color('black')
        self.zoom = 2.0
        # self.zoom = 1.4831937955578253e-06

        self.offset = np.array([-0.7453, 0.1127])
        # self.offset = np.array([0.25740134, 0.00124048])

        self.max_iterations = 2500

        self.program['u_max_iterations'] = self.max_iterations
        self.program['u_zoom'] = self.zoom
        self.program['u_offset'] = self.offset
        self.show()

        self.writer = writer  # Writer object to write frames directly



        # 2500
    def on_draw(self, event):
        gloo.clear()
        gloo.set_state(blend=False)
        self.program.draw('triangle_strip')
        frame = gloo.read_pixels()  # Capture frame
        self.writer.append_data(frame)  # Write frame to video        

    def on_mouse_wheel(self, event):
        # Determine the zoom factor (you might adjust the rate of zoom here)
        zoom_factor = 1.1 if -event.delta[1] > 0 else 0.9
        old_zoom = self.zoom
        new_zoom = old_zoom * zoom_factor

        # Get the cursor position in normalized device coordinates (NDC)
        x, y = event.pos
        ndc_x = (x / self.size[0]) * 2 - 1  # convert pixel x to NDC
        ndc_y = 1 - (y / self.size[1]) * 2  # convert pixel y to NDC, flipped y-axis

        # Calculate how much we need to shift the offset to keep the cursor's position "stable" in world coordinates
        cursor_pos_world = (ndc_x * old_zoom + self.offset[0], ndc_y * old_zoom + self.offset[1])
        new_offset_x = cursor_pos_world[0] - ndc_x * new_zoom
        new_offset_y = cursor_pos_world[1] - ndc_y * new_zoom

        # Apply the new zoom and offset
        self.zoom = new_zoom
        self.offset = np.array([new_offset_x, new_offset_y])

        # Update the uniform values and redraw
        self.program['u_zoom'] = self.zoom
        self.program['u_offset'] = self.offset
        self.update()
    # def on_mouse_move(self, event):
    #     if event.is_dragging:
    #         # Convert pixel coordinates to Mandelbrot coordinate offsets
    #         dx, dy = event.pos - event.last_event.pos
    #         self.offset -= np.array([dx, dy]) * (4.0 / self.size[0]) / self.zoom
    #         self.program['u_offset'] = self.offset
    #         self.update()

    def on_mouse_move(self, event):
        if event.is_dragging:
            dx, dy = -(event.pos - event.last_event.pos)
            # Normalize panning speed according to the zoom level
            # Ensure dx, dy movements are inversely proportional to the zoom level
            scale_factor = 4.0 / self.size[0]  # Adjust the scale factor based on canvas size and desired sensitivity
            dx = (dx * scale_factor) * self.zoom
            dy = (dy * scale_factor) * self.zoom
            # Update offset according to adjusted delta values
            self.offset += np.array([dx, -dy])  # invert dy for intuitive 'dragging' behavior
            self.program['u_offset'] = self.offset
            self.update()
    def on_key_press(self, event):
        # Handle arrow keys for fine control of offset
        step = 0.01 / self.zoom
        if event.key == 'Left':
            self.offset[0] -= step
        elif event.key == 'Right':
            self.offset[0] += step
        elif event.key == 'Up':
            self.offset[1] += step
        elif event.key == 'Down':
            self.offset[1] -= step
        elif event.key == 'p':
            self.max_iterations  += 2500

        elif event.key == 'm':
            self.max_iterations  -= 2500 
            self.max_iterations  = max(self.max_iterations ,2500)
        
        self.program['u_max_iterations'] = self.max_iterations
        print(self.program['u_max_iterations'])
        self.program['u_offset'] = self.offset
        self.update()

    def on_resize(self, event):
        gloo.set_viewport(0, 0, *event.physical_size)
if __name__ == '__main__':
    # canvas = MandelbrotCanvas()
    # app.run()

    with imageio.get_writer('mandelbrot_animation.mp4', fps=30) as writer:
        canvas = MandelbrotCanvas(writer)
        app.run()
    print("Animation saved to mandelbrot_animation.mp4")    