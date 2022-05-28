#version 130

uniform vec3 offset;

uniform mat4 mvp;
uniform mat4 invp;
uniform mat4 clip;
uniform vec4 nclip;

uniform vec3 sun_dir;

in vec4 pos;
in vec2 cuv;
in vec2 luv;

smooth out vec2 v_cuv;
smooth out vec2 v_luv;

smooth out float v_diffuse;
smooth out vec3 v_light_dir;
smooth out vec3 v_pos;
smooth out float v_dist;

void main() {
    v_cuv = cuv;
    v_luv = luv;

    // Compute the position in relative space (absolute orientation, but origin at the camera)
    vec4 posh = vec4(pos.xyz + offset, 1);

    // Model matrix
    mat4 mv = invp * mvp;

    // Compute the position in camera space (world orientation, origin at camera, no perspective)
    vec3 cpos = (mv * posh).xyz;

    // Compute normal in camera space
    const vec4 NORMAL_TABLE[6] = vec4[6](
        vec4(1., 0., 0., 0.),
        vec4(-1., 0., 0., 0.),
        vec4(0., 1., 0., 0.),
        vec4(0., -1., 0., 0.),
        vec4(0., 0., 1., 0.),
        vec4(0., 0., -1., 0.)
    );
    vec3 normal = normalize((mv * NORMAL_TABLE[int(pos.w)]).xyz);

    // Direction of the light rays
    vec3 l_dir = -sun_dir;

    // Compute the diffuse lighting on this surface
    float diffuse = -dot(normal, l_dir);
    v_diffuse = diffuse < 0 ? diffuse : diffuse * 0.25;

    // Compute the specular light direction
    v_light_dir = l_dir - 2 * normal * dot(l_dir, normal);

    // Store the position in camera space
    v_pos = cpos;
    v_dist = length(cpos);

    // Compute the position in screenspace
    gl_Position = mvp * posh;
}
