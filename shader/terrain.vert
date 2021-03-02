#version 130

uniform vec3 offset;
uniform mat4 mvp;
uniform mat4 mv;
uniform vec3 l_dir;

in vec3 pos;
in uint color;

out vec3 v_color;
out vec3 v_light_dir;
out float v_diffuse;
out vec3 v_pos;

const vec3 NORMALS[6] = vec3[6](
    vec3(1., 0., 0.),
    vec3(-1., 0., 0.),
    vec3(0., 1., 0.),
    vec3(0., -1., 0.),
    vec3(0., 0., 1.),
    vec3(0., 0., -1.)
);

void main() {
    //Extract R, G, B and normal index from a single 32-bit input
    float r = float((color >> 24u) & 0xffu) * (1./256.);
    float g = float((color >> 16u) & 0xffu) * (1./256.);
    float b = float((color >> 8u) & 0xffu) * (1./256.);
    vec3 raw_normal = NORMALS[color & 0xffu];

    //Compute normal using modelview matrix
    vec3 normal = vec3(mv * vec4(raw_normal, 0));

    //Calculate how aligned are the normal and the light direction (for diffuse lighting)
    float diffuse = -dot(normal, l_dir);
    if (diffuse < 0) {
        v_diffuse = diffuse * 0.1;
    }else{
        v_diffuse = diffuse;
    }

    //Calculate the reflection of the global light on this normal
    v_light_dir = l_dir + 2 * normal * dot(l_dir, normal);

    //Send color
    v_color = vec3(r, g, b);

    //Send the view position of the vertex
    v_pos = vec3(mv * vec4(pos + offset, 1));

    //Send the projected position of the vertex
    gl_Position = mvp * vec4(pos + offset, 1);
}