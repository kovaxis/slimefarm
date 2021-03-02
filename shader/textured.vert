#version 130

uniform mat4 mvp;

in vec2 pos;
in vec2 tex;

out vec2 tex_coords;

void main() {
    tex_coords = tex;
    gl_Position = mvp * vec4(pos, 0, 1);
}