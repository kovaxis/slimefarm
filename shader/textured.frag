#version 130

uniform sampler2D tex;

in vec2 tex_coords;

out vec4 out_color;

void main() {
    out_color = texture2D(tex, tex_coords);
}