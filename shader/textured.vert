#version 110

uniform mat4 mvp;

attribute vec2 pos;
attribute vec2 tex;

varying vec2 tex_coords;

void main() {
    tex_coords = tex;
    gl_Position = mvp * vec4(pos, 0, 1);
}