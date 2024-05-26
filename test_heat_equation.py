import numpy as np
from vispy import app, gloo, use

# Set the backend to PyQt5
use('pyqt5')

# Define vertex shader with GLSL version 120
vertex_shader = """
#version 120
attribute vec2 a_position;
varying vec2 v_position;
void main() {
    v_position = a_position;
    gl_Position = vec4(a_position, 0.0, 1.0);
}
"""

# Define fragment shader with GLSL version 120
fragment_shader = """
#version 120
varying vec2 v_position;
uniform float u_zoom;
uniform vec2 u_offset;

vec3 hsv_to_rgb(float h, float s, float v) {
    vec3 rgb = clamp(abs(mod(h*6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);
    return v * mix(vec3(1.0), rgb, s);
}

void main() {
    vec2 c = v_position * u_zoom + u_offset;
    vec2 z = vec2(0.0, 0.0);
    int max_iterations = 100;
    int iterations;
    
    for(iterations = 0; iterations < max_iterations; iterations++) {
        if (dot(z, z) > 4.0) break;
        z = vec2(
            z.x * z.x - z.y * z.y + c.x,
            2.0 * z.x * z.y + c.y
        );
    }
    
    float color = float(iterations) / float(max_iterations);
    vec3 rgb = hsv_to_rgb(color * 360.0, 1.0, color < 1.0 ? 1.0 : 0.0);
    gl_FragColor = vec4(rgb, 1.0);
}
"""

class MandelbrotCanvas(app.Canvas):
    def __init__(self):
        app.Canvas.__init__(self, size=(800, 800), title='Mandelbrot Set')
        
        self.program = gloo.Program(vertex_shader, fragment_shader)
        
        # Define the square's vertices covering the entire screen
        self.vertices = np.array([[-1, -1], 
                                  [-1,  1], 
                                  [ 1, -1], 
                                  [ 1,  1]], dtype=np.float32)
        
        # Create Vertex Buffer Object (VBO)
        self.vbo = gloo.VertexBuffer(self.vertices)
        
        # Set attribute locations
        self.program['a_position'] = self.vbo
        self.program['u_zoom'] = 2.0
        self.program['u_offset'] = (-0.5, 0.0)
        
        self.show()
    
    def on_draw(self, event):
        gloo.clear()
        self.program.draw('triangle_strip')
    
    def on_resize(self, event):
        gloo.set_viewport(0, 0, *event.physical_size)

if __name__ == '__main__':
    canvas = MandelbrotCanvas()
    app.run()