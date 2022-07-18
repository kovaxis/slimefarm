#version 130

uniform mat4 mvp;
uniform mat4 invp;
uniform mat4 clip;
uniform vec4 nclip;

uniform vec3 sun_dir;

in vec3 pos;
in vec4 normal;
in vec4 color;

in vec3 ppos;
in mat3 prot;
in vec4 pcol;
in float psize;

flat out vec4 v_color;

smooth out float v_diffuse;
smooth out vec3 v_light_dir;
smooth out vec3 v_pos;
smooth out float v_dist;

void main() {
    float gamma = 2.2;

    // Compute the position in relative space (absolute orientation, but origin at the camera)
    vec4 posh = vec4(psize * (prot * pos) + ppos, 1);

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
    vec3 normal = normalize((mv * normal).xyz);

    // Store the color
    v_color = pow(pcol, vec4(gamma));

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

    // TODO: Apply clip planes
    gl_ClipDistance[0] = 0;
    gl_ClipDistance[1] = 0;
    gl_ClipDistance[2] = 0;
    gl_ClipDistance[3] = 0;
    gl_ClipDistance[4] = 0;
}
