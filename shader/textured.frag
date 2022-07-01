#version 130

uniform sampler2D tex;
uniform vec4 tint;

in vec2 tex_coords;

out vec4 out_color;

void main() {
    vec4 color = texture2D(tex, tex_coords) * tint;
    if (color.a < 0.01) {
        discard;
    }
    out_color = color;
}