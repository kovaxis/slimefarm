#version 130

uniform vec3 offset;
uniform mat4 mvp;

in vec4 pos;
in vec2 cuv;
in vec2 luv;

smooth out vec2 v_cuv;
smooth out vec2 v_luv;

void main() {
    v_cuv = cuv;
    v_luv = luv;

    vec4 posh = vec4(pos.xyz + offset, 1);
    gl_Position = mvp * posh;
}


/*
uniform vec3 offset;
uniform mat4 mvp;
uniform mat4 mv;
uniform vec3 l_dir;
uniform mat4 clip;
uniform vec4 nclip;

out vec3 v_pos;

void main() {
    //Compute real normal using modelview matrix
    vec3 real_normal = vec3(mv * normal);

    //Calculate how aligned are the normal and the light direction (for diffuse lighting)
    float diffuse = -dot(real_normal, l_dir);
    v_diffuse = diffuse < 0 ? diffuse : diffuse * 0.25;

    //Calculate the reflection of the global light on this normal
    v_light_dir = l_dir + 2 * real_normal * dot(l_dir, real_normal);

    //Send color
    v_color = color;
    vec4 posh = vec4(vec3(pos) + offset, 1);

    //Send the view position of the vertex
    v_pos = vec3(mv * posh);

    //Calculate distance to clipping planes
    gl_ClipDistance[0] = dot(nclip, posh);
    vec4 clip_dists = posh * clip;
    gl_ClipDistance[1] = clip_dists.x;
    gl_ClipDistance[2] = clip_dists.y;
    gl_ClipDistance[3] = clip_dists.z;
    gl_ClipDistance[4] = clip_dists.w;

    //Send the projected position of the vertex
    gl_Position = mvp * posh;
}
*/