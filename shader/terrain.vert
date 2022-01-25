#version 130

uniform vec3 offset;
uniform mat4 mvp;
uniform mat4 mv;
uniform vec3 l_dir;

in vec3 pos;
in vec4 normal;
in vec4 color;

out vec4 v_color;
out vec3 v_light_dir;
out float v_diffuse;
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

    //Send the view position of the vertex
    v_pos = vec3(mv * vec4(pos + offset, 1));

    //Send the projected position of the vertex
    gl_Position = mvp * vec4(pos + offset, 1);
}