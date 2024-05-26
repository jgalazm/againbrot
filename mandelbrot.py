import numpy as np
from vispy import app, gloo, use
from PIL import Image

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
uniform sampler2D u_texture;


vec3 hsv_to_rgb(float h, float s, float v) {
    vec3 rgb = clamp(abs(mod(h*6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);
    return v * mix(vec3(1.0), rgb, s);
}

void main() {
    vec2 c = v_position * u_zoom + u_offset;
    vec2 z = vec2(0.0, 0.0);
    int max_iterations = 2500;
    int iterations;
    
    for(iterations = 0; iterations < max_iterations; iterations++) {
        if (dot(z, z) > 4.0) break;
        z = vec2(
            z.x * z.x - z.y * z.y + c.x,
            2.0 * z.x * z.y + c.y
        );
    }
    
    float color = float(iterations) / float(max_iterations);
    vec3 rgb = hsv_to_rgb(color, 1.0, color < 1.0 ? 1.0 : 0.0);
    gl_FragColor = vec4(rgb, 1.0);

    vec2 texture_coords = mod(vec2(z.x, z.y)/10.0, vec2(1.0, 1.0));
    vec3 tex_color = texture2D(u_texture, texture_coords).rgb;
    // tex_color = color < 1.0 ? tex_color: vec3(0.0, 0.0, 0.0);
    gl_FragColor = vec4(tex_color, 1.0);    
}
"""


image_path = 'nahida.jpg'
image = Image.open(image_path)
image_data = np.array(image)


class MandelbrotCanvas(app.Canvas):
    def __init__(self):
        app.Canvas.__init__(self, size=(800, 800), title='Mandelbrot Set Zoom Animation')
        # Define the square's vertices covering the entire screen
        self.vertices = np.array([[-1, -1], 
                                  [-1,  1], 
                                  [ 1, -1], 
                                  [ 1,  1]], dtype=np.float32)
        
        # Create Vertex Buffer Object (VBO)
        self.vbo = gloo.VertexBuffer(self.vertices)        

        # Create a framebuffer to render the texture
        # self.color_texture = gloo.Texture2D(shape=(800, 800, 3), interpolation='linear') # Create a texture from the image data
        self.color_texture = gloo.Texture2D(image_data)
        # self.framebuffer = gloo.FrameBuffer(color=self.color_texture)

                
        
        self.program = gloo.Program(vertex_shader, fragment_shader)
        self.program['a_position'] = self.vbo
        self.program['u_zoom'] = 2.0
        self.program['u_offset'] = np.array([-0.7453+0.5, 0.1127])  # Centered on a mini Mandelbrot set
        self.program['u_offset'] = np.array([-0.7453, 0.1127])
        self.program['u_texture'] = self.color_texture

        self.color_program = gloo.Program(vertex_shader, color_shader)
        self.color_program['a_position'] = self.vbo

        
        # Set the clear color to black
        gloo.set_clear_color('black')
        
        self.zoom_direction = -1  # Start with zooming out
        self.zoom = 2.0  # Start with the final zoom level
        self.target_zoom_in = 1.0/50000.0
        self.original_zoom = 2.0
        self.timer = app.Timer('auto', self.on_timer)
        self.timer.start()
        
        self.t = 0  # Time variable for smooth animation
        self.zoom_duration = 5.0  # Duration of one zoom-in or zoom-out cycle in seconds
        
        self.texture_rendered = False  # Flag to check if the texture is rendered

        self.show()
    
    def on_draw(self, event):
        # if not self.texture_rendered:
        #     # Render the color texture once
        #     with self.framebuffer:
        #         gloo.clear()
        #         self.color_program.draw('triangle_strip')
        #     self.texture_rendered = True        
        gloo.clear()
        # Disable blending to ensure the color is not affected
        gloo.set_state(blend=False)
        self.program.draw('triangle_strip')
    
    def on_timer(self, event):
        # Exponential zoom in and out animation logic
        cycle_time = self.zoom_duration * 2
        half_cycle = self.zoom_duration
        
        # Update time variable
        self.t += event.dt
        cycle_pos = self.t % cycle_time
        
        if cycle_pos < half_cycle:
            # Zooming out
            progress = cycle_pos / half_cycle
            self.zoom = self.target_zoom_in * ((self.original_zoom / self.target_zoom_in) ** progress)
        else:
            # Zooming in
            progress = (cycle_time - cycle_pos) / half_cycle
            self.zoom = self.target_zoom_in * ((self.original_zoom / self.target_zoom_in) ** progress)
        
        self.program['u_zoom'] = self.zoom
        self.update()
    
    def on_resize(self, event):
        gloo.set_viewport(0, 0, *event.physical_size)

if __name__ == '__main__':
    canvas = MandelbrotCanvas()
    app.run()