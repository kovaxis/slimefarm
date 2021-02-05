#version 110

uniform mat4 mvp;
uniform vec4 tint;

attribute vec3 pos;
attribute vec4 color;

varying vec4 v_color;

void main() {
    v_color = color * tint;
    gl_Position = mvp * vec4(pos, 1);
}
