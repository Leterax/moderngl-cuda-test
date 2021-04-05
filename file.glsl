#version 330

#if defined VERTEX_SHADER

in vec3 in_position;
in vec4 in_color;

uniform mat4 m_camera;
uniform mat4 m_proj;

out vec4 color;

void main() {
    gl_Position =  m_proj * m_camera * vec4(in_position, 1.0);;
    color = in_color;
}

#elif defined FRAGMENT_SHADER

out vec4 fragColor;
in vec4 color;

void main() {
    fragColor = vec4(color.xyz, 1.);
}
#endif