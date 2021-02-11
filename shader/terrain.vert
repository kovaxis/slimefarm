#version 110

uniform vec3 offset;
uniform mat4 mvp;

attribute vec3 pos;
attribute vec4 color;

varying vec4 v_color;

void main() {
    v_color = color;
    gl_Position = mvp * vec4(pos + offset, 1);
}