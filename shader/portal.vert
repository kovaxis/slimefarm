#version 130

uniform mat4 mvp;
uniform vec3 offset;
uniform vec4 nclip;
uniform mat4 clip;

in vec3 pos;
in vec3 normal;
in vec4 color;

void main() {
    vec4 posh = vec4(pos + offset, 1);
    
    gl_ClipDistance[0] = dot(posh, nclip);
    vec4 cd = posh * clip;
    gl_ClipDistance[1] = cd.x;
    gl_ClipDistance[2] = cd.y;
    gl_ClipDistance[3] = cd.z;
    gl_ClipDistance[4] = cd.w;
    
    gl_Position = mvp * posh;
}
