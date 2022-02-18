use crate::{
    lua::{CameraFrame, CameraStack, MatrixStack},
    prelude::*,
};
use common::{lua_assert, lua_bail, lua_func, lua_lib, lua_type};

#[derive(Clone)]
pub(crate) enum BufferRef {
    NoBuf,
    Buf2d(Rc<Buffer2d>),
    Buf3d(Rc<Buffer3d>),
}
unsafe impl Send for BufferRef {}
unsafe impl Sync for BufferRef {}
impl LuaUserData for BufferRef {}

pub(crate) enum UniformVal {
    Float(f32),
    Vec2([f32; 2]),
    Vec3([f32; 3]),
    Vec4([f32; 4]),
    Mat4([[f32; 4]; 4]),
    Texture2d(TextureRef, SamplerBehavior),
}
impl UniformVal {
    fn as_uniform(&self) -> UniformValue {
        match self {
            &Self::Float(v) => UniformValue::Float(v),
            &Self::Vec2(v) => UniformValue::Vec2(v),
            &Self::Vec3(v) => UniformValue::Vec3(v),
            &Self::Vec4(v) => UniformValue::Vec4(v),
            &Self::Mat4(v) => UniformValue::Mat4(v),
            Self::Texture2d(tex, sampling) => UniformValue::Texture2d(&tex.tex, Some(*sampling)),
        }
    }
}

#[derive(Clone)]
pub(crate) struct UniformStorage {
    pub vars: AssertSync<Rc<RefCell<Vec<(String, UniformVal)>>>>,
}
impl Uniforms for UniformStorage {
    fn visit_values<'a, F: FnMut(&str, UniformValue<'a>)>(&self, mut visit: F) {
        for (name, val) in self.vars.borrow().iter() {
            let as_uniform: UniformValue<'a> = unsafe {
                let as_uniform: UniformValue = val.as_uniform();
                mem::transmute(as_uniform)
            };
            visit(name, as_uniform);
        }
    }
}
lua_type! {UniformStorage,
    fn add(lua, this, name: String) {
        let mut vars = this.vars.0.borrow_mut();
        let idx = vars.len();
        vars.push((name, UniformVal::Float(0.)));
        idx
    }

    fn set_float(lua, this, (idx, val): (usize, f32)) {
        let mut vars = this.vars.borrow_mut();
        vars
            .get_mut(idx)
            .ok_or("index out of range")
            .to_lua_err()?
            .1 = UniformVal::Float(val);
    }
    fn set_vec2(lua, this, (idx, x, y): (usize, f32, f32)) {
        let mut vars = this.vars.borrow_mut();
        vars
            .get_mut(idx)
            .ok_or("index out of range")
            .to_lua_err()?
            .1 = UniformVal::Vec2([x, y]);
    }
    fn set_vec3(lua, this, (idx, x, y, z): (usize, f32, f32, f32)) {
        let mut vars = this.vars.borrow_mut();
        vars
            .get_mut(idx)
            .ok_or("index out of range")
            .to_lua_err()?
            .1 = UniformVal::Vec3([x, y, z]);
    }
    fn set_vec4(lua, this, (idx, x, y, z, w): (usize, f32, f32, f32, f32)) {
        let mut vars = this.vars.borrow_mut();
        vars
            .get_mut(idx)
            .ok_or("index out of range")
            .to_lua_err()?
            .1 = UniformVal::Vec4([x, y, z, w]);
    }

    fn set_matrix(lua, this, (idx, mat): (usize, MatrixStack)) {
        let (_, top) = *mat.stack.borrow();
        let mut vars = this.vars.borrow_mut();
        vars
            .get_mut(idx)
            .ok_or("index out of range")
            .to_lua_err()?
            .1 = UniformVal::Mat4(top.into());
    }

    fn set_texture(lua, this, (idx, tex): (usize, TextureRef)) {
        let mut vars = this.vars.borrow_mut();
        let sampling = tex.sampling;
        vars
            .get_mut(idx)
            .ok_or("index out of range")
            .to_lua_err()?
            .1 = UniformVal::Texture2d(tex, sampling);
    }
}

pub(crate) struct Font {
    pub state: Rc<State>,
    pub text: RefCell<TextDisplay<Rc<FontTexture>>>,
}
impl Font {
    fn new(state: &Rc<State>, font_data: &[u8], size: u32) -> Result<Font> {
        let state = state.clone();
        let tex = Rc::new(FontTexture::new(
            &state.display,
            font_data,
            size,
            (0..0x250).filter_map(|i| std::char::from_u32(i)),
        )?);
        let text = RefCell::new(TextDisplay::new(&state.text_sys, tex, ""));
        Ok(Font { state, text })
    }

    fn draw(&self, text: &str, mvp: Mat4, color: [f32; 4], draw_params: &DrawParameters) {
        let mut frame = self.state.frame.borrow_mut();
        let mut text_disp = self.text.borrow_mut();
        text_disp.set_text(text);
        glium_text_rusttype::draw_with_params(
            &text_disp,
            &self.state.text_sys,
            &mut *frame,
            mvp,
            (color[0], color[1], color[2], color[3]),
            SamplerBehavior {
                minify_filter: MinifySamplerFilter::Nearest,
                magnify_filter: MagnifySamplerFilter::Nearest,
                ..default()
            },
            &draw_params,
        )
        .unwrap();
    }
}

#[derive(Clone)]
pub(crate) struct FontRef {
    pub rc: AssertSync<Rc<Font>>,
}
lua_type! {FontRef,
    fn draw(lua, this, (text, mvp, draw_params, r, g, b, a): (LuaString, MatrixStack, LuaDrawParams, f32, f32, f32, Option<f32>)) {
        let (_, mvp) = &*mvp.stack.borrow();
        this.rc.draw(text.to_str()?, *mvp, [r, g, b, a.unwrap_or(1.)], &draw_params.params);
    }
}
#[derive(Clone)]
pub(crate) struct ShaderRef {
    pub program: AssertSync<Rc<Program>>,
}
impl LuaUserData for ShaderRef {}

#[derive(Clone)]
pub(crate) struct TextureRef {
    pub tex: AssertSync<Rc<Texture2d>>,
    pub sampling: SamplerBehavior,
}
impl TextureRef {
    fn new(tex: Texture2d) -> Self {
        use glium::uniforms::{MagnifySamplerFilter, MinifySamplerFilter, SamplerWrapFunction};
        Self {
            tex: AssertSync(Rc::new(tex)),
            sampling: SamplerBehavior {
                minify_filter: MinifySamplerFilter::Nearest,
                magnify_filter: MagnifySamplerFilter::Nearest,
                wrap_function: (
                    SamplerWrapFunction::Repeat,
                    SamplerWrapFunction::Repeat,
                    SamplerWrapFunction::Repeat,
                ),
                depth_texture_comparison: None,
                max_anisotropy: 1,
            },
        }
    }
}
lua_type! {TextureRef,
    fn dimensions(lua, this, ()) {
        (this.tex.width(), this.tex.height())
    }

    mut fn set_min(lua, this, filter: LuaString) {
        use glium::uniforms::MinifySamplerFilter::*;
        this.sampling.minify_filter = match filter.as_bytes() {
            b"linear" => Linear,
            b"nearest" => Nearest,
            _ => lua_bail!("unknown minify filter '{}'", filter.to_str().unwrap_or_default())
        };
    }

    mut fn set_mag(lua, this, filter: LuaString) {
        use glium::uniforms::MagnifySamplerFilter::*;
        this.sampling.magnify_filter = match filter.as_bytes() {
            b"linear" => Linear,
            b"nearest" => Nearest,
            _ => lua_bail!("unknown magnify filter '{}'", filter.to_str().unwrap_or_default())
        };
    }

    mut fn set_wrap(lua, this, wrap: LuaString) {
        use glium::uniforms::SamplerWrapFunction::*;
        let func = match wrap.as_bytes() {
            b"repeat" => Repeat,
            b"mirror" => Mirror,
            b"clamp" => Clamp,
            _ => lua_bail!("unknown wrap function '{}'", wrap.to_str().unwrap_or_default())
        };
        this.sampling.wrap_function = (func, func, func);
    }
}

#[derive(Clone, Default)]
pub(crate) struct LuaDrawParams {
    pub params: AssertSync<DrawParameters<'static>>,
}
lua_type! {LuaDrawParams,
    mut fn set_depth(lua, this, (test, write, clamp, near, far): (LuaString, bool, Option<LuaString>, Option<f32>, Option<f32>)) {
        use glium::draw_parameters::{DepthTest::*, DepthClamp::*, Depth};
        let test = match test.as_bytes() {
            b"always_fail" => Ignore,
            b"always_pass" => Overwrite,
            b"if_equal" => IfEqual,
            b"if_not_equal" => IfNotEqual,
            b"if_more" => IfMore,
            b"if_more_or_equal" => IfMoreOrEqual,
            b"if_less" => IfLess,
            b"if_less_or_equal" => IfLessOrEqual,
            _ => lua_bail!("unknown depth test"),
        };
        let clamp = match clamp.as_ref().map(|s| s.as_bytes()) {
            Some(b"none") => NoClamp,
            Some(b"both") => Clamp,
            Some(b"near") => ClampNear,
            Some(b"far") => ClampFar,
            None => NoClamp,
            _ => lua_bail!("invalid depth clamp"),
        };
        this.params.depth = Depth {
            test,
            write,
            clamp,
            range: (near.unwrap_or(0.), far.unwrap_or(1.)),
        };
    }

    mut fn set_color_mask(lua, this, (r, g, b, a): (bool, Option<bool>, Option<bool>, Option<bool>)) {
        this.params.color_mask = match (r, g, b, a) {
            (d, None, None, None) => (d, d, d, d),
            (r, Some(g), Some(b), Some(a)) => (r, g, b, a),
            _ => lua_bail!("invalid color channel mask"),
        };
    }

    mut fn set_color_blend(lua, this, (func, src, dst): (LuaString, LuaString, LuaString)) {
        use glium::draw_parameters::{BlendingFunction::*, LinearBlendingFactor::{self, *}};
        fn map_factor(s: LuaString) -> LuaResult<LinearBlendingFactor> {
            Ok(match s.as_bytes() {
                b"zero" => Zero,
                b"one" => One,
                b"src_color" => SourceColor,
                b"one_minus_src_color" => OneMinusSourceColor,
                b"dst_color" => DestinationColor,
                b"one_minus_dst_color" => OneMinusDestinationColor,
                b"src_alpha" => SourceAlpha,
                b"src_alpha_saturate" => SourceAlphaSaturate,
                b"one_minus_src_alpha" => OneMinusSourceAlpha,
                b"dst_alpha" => DestinationAlpha,
                b"one_minus_dst_alpha" => OneMinusDestinationAlpha,
                b"constant_color" => ConstantColor,
                b"one_minus_constant_color" => OneMinusConstantColor,
                b"constant_alpha" => ConstantAlpha,
                b"one_minus_constant_alpha" => OneMinusConstantAlpha,
                _ => lua_bail!("unknown blending factor"),
            })
        }
        let source = map_factor(src)?;
        let destination = map_factor(dst)?;
        let func = match func.as_bytes() {
            b"replace" => AlwaysReplace,
            b"min" => Min,
            b"max" => Max,
            b"add" => Addition{source,destination},
            b"sub" => Subtraction{source,destination},
            b"reverse_sub" => ReverseSubtraction{source,destination},
            _ => lua_bail!("unknown depth test"),
        };
        this.params.blend.color = func;
    }

    mut fn set_alpha_blend(lua, this, (func, src, dst): (LuaString, LuaString, LuaString)) {
        use glium::draw_parameters::{BlendingFunction::*, LinearBlendingFactor::{self, *}};
        fn map_factor(s: LuaString) -> LuaResult<LinearBlendingFactor> {
            Ok(match s.as_bytes() {
                b"zero" => Zero,
                b"one" => One,
                b"src_color" => SourceColor,
                b"one_minus_src_color" => OneMinusSourceColor,
                b"dst_color" => DestinationColor,
                b"one_minus_dst_color" => OneMinusDestinationColor,
                b"src_alpha" => SourceAlpha,
                b"src_alpha_saturate" => SourceAlphaSaturate,
                b"one_minus_src_alpha" => OneMinusSourceAlpha,
                b"dst_alpha" => DestinationAlpha,
                b"one_minus_dst_alpha" => OneMinusDestinationAlpha,
                b"constant_color" => ConstantColor,
                b"one_minus_constant_color" => OneMinusConstantColor,
                b"constant_alpha" => ConstantAlpha,
                b"one_minus_constant_alpha" => OneMinusConstantAlpha,
                _ => lua_bail!("unknown blending factor"),
            })
        }
        let source = map_factor(src)?;
        let destination = map_factor(dst)?;
        let func = match func.as_bytes() {
            b"replace" => AlwaysReplace,
            b"min" => Min,
            b"max" => Max,
            b"add" => Addition{source,destination},
            b"sub" => Subtraction{source,destination},
            b"reverse_sub" => ReverseSubtraction{source,destination},
            _ => lua_bail!("unknown depth test"),
        };
        this.params.blend.alpha = func;
    }

    mut fn set_cull(lua, this, winding: LuaString) {
        use glium::draw_parameters::BackfaceCullingMode::*;
        let cull = match winding.as_bytes() {
            b"cw" => CullClockwise,
            b"none" => CullingDisabled,
            b"ccw" => CullCounterClockwise,
            _ => lua_bail!("unknown cull winding")
        };
        this.params.backface_culling = cull;
    }

    mut fn set_stencil(lua, this, (winding, test, refval, pass, fail, depthfail): (LuaString, LuaString, i32, Option<LuaString>, Option<LuaString>, Option<LuaString>)) {
        use glium::draw_parameters::{StencilTest::*, StencilOperation::*};
        let p = &mut this.params.stencil;
        let (mtest, mrefval, mpass, mfail, mdepthfail) = match winding.as_bytes() {
            b"cw" => (
                &mut p.test_clockwise,
                &mut p.reference_value_clockwise,
                &mut p.depth_pass_operation_clockwise,
                &mut p.fail_operation_clockwise,
                &mut p.pass_depth_fail_operation_clockwise,
            ),
            b"ccw" => (
                &mut p.test_counter_clockwise,
                &mut p.reference_value_counter_clockwise,
                &mut p.depth_pass_operation_counter_clockwise,
                &mut p.fail_operation_counter_clockwise,
                &mut p.pass_depth_fail_operation_counter_clockwise,
            ),
            _ => lua_bail!("unknown winding"),
        };
        let test = match test.as_bytes() {
            b"always_pass" => AlwaysPass,
            b"always_fail" => AlwaysFail,
            b"if_less" => IfLess { mask: !0 },
            b"if_less_or_equal" => IfLessOrEqual { mask: !0 },
            b"if_more" => IfMore { mask: !0 },
            b"if_more_or_equal" => IfMoreOrEqual { mask: !0 },
            b"if_equal" => IfEqual { mask: !0 },
            b"if_not_equal" => IfEqual { mask: !0 },
            _ => lua_bail!("unknown stencil test"),
        };
        let get_op = |op: Option<LuaString>, name| Ok(match op {
            Some(s) => match s.as_bytes() {
                b"keep" => Keep,
                b"zero" => Zero,
                b"replace" => Replace,
                b"increment" => Increment,
                b"increment_wrap" => IncrementWrap,
                b"decrement" => Decrement,
                b"decrement_wrap" => DecrementWrap,
                b"invert" => Invert,
                _ => lua_bail!("unknown stencil {} operation", name),
            },
            None => Keep,
        });
        let pass = get_op(pass, "pass")?;
        let fail = get_op(fail, "fail")?;
        let depthfail = get_op(depthfail, "depth fail")?;
        *mtest = test;
        *mrefval = refval;
        *mpass = pass;
        *mfail = fail;
        *mdepthfail = depthfail;
    }

    mut fn set_stencil_ref(lua, this, refval: i32) {
        this.params.stencil.reference_value_counter_clockwise = refval;
    }

    mut fn set_clip_planes(lua, this, planes: u32) {
        this.params.clip_planes_bitmask = planes;
    }

    mut fn set_multisampling(lua, this, enable: bool) {
        this.params.multisampling = enable;
    }

    mut fn set_dithering(lua, this, enable: bool) {
        this.params.dithering = enable;
    }

    mut fn set_viewport(lua, this, (left, bottom, w, h): (Option<u32>, Option<u32>, Option<u32>, Option<u32>)) {
        this.params.viewport = match (left, bottom, w, h) {
            (None, None, None, None) => None,
            (Some(left), Some(bottom), Some(width), Some(height)) => Some(glium::Rect {
                left,
                bottom,
                width,
                height,
            }),
            _ => lua_bail!("invalid viewport rectangle"),
        };
    }

    mut fn set_scissor(lua, this, (left, bottom, w, h): (Option<u32>, Option<u32>, Option<u32>, Option<u32>)) {
        this.params.scissor = match (left, bottom, w, h) {
            (None, None, None, None) => None,
            (Some(left), Some(bottom), Some(width), Some(height)) => Some(glium::Rect {
                left,
                bottom,
                width,
                height,
            }),
            _ => lua_bail!("invalid scissor rectangle"),
        };
    }

    mut fn set_draw_primitives(lua, this, enable: bool) {
        this.params.draw_primitives = enable;
    }

    mut fn set_smooth(lua, this, smooth: Option<LuaString>) {
        use glium::draw_parameters::Smooth::*;
        this.params.smooth = match smooth.as_ref().map(|s| s.as_bytes()) {
            Some(b"fastest") => Some(Fastest),
            Some(b"nicest") => Some(Nicest),
            Some(b"dont_care") => Some(DontCare),
            None => None,
            _ => lua_bail!("invalid smooth mode"),
        };
    }

    mut fn set_bounding_box(lua, this, (x0, x1, y0, y1, z0, z1, w0, w1): (f32, f32, f32, f32, f32, f32, f32, f32)) {
        this.params.primitive_bounding_box = (x0..x1, y0..y1, z0..z1, w0..w1);
    }

    mut fn set_polygon_offset(lua, this, (enable, factor, units): (bool, f32, f32)) {
        use glium::draw_parameters::PolygonOffset;
        this.params.polygon_offset = PolygonOffset {
            factor,
            units,
            fill: enable,
            point: false,
            line: false,
        };
    }
}

pub(crate) fn open_gfx_lib(state: &Rc<State>, lua: LuaContext) {
    lua.globals()
            .set(
                "gfx",
                lua_lib! {lua, state,
                    fn shader((vertex, fragment): (String, String)) {
                        let shader = program!{&state.display,
                            110 => {
                                vertex: &*vertex,
                                fragment: &*fragment,
                            }
                        }.to_lua_err()?;
                        ShaderRef{program: AssertSync(Rc::new(shader))}
                    }

                    fn buffer_empty(()) {
                        BufferRef::NoBuf
                    }

                    fn buffer_2d((pos, tex, indices): (Vec<f32>, Vec<f32>, Vec<VertIdx>)) {
                        lua_assert!(pos.len() % 2 == 0, "positions not multiple of 2");
                        lua_assert!(tex.len() % 2 == 0, "texcoords not multiple of 4");
                        lua_assert!(pos.len() == tex.len(), "not the same amount of positions as texcoords");
                        let vertices = pos.chunks_exact(2).zip(tex.chunks_exact(2)).map(|(pos, tex)| {
                            TexturedVertex {pos: [pos[0], pos[1]], tex: [tex[0], tex[1]]}
                        }).collect::<Vec<_>>();
                        BufferRef::Buf2d(Rc::new(Buffer2d {
                            vertex: VertexBuffer::new(&state.display, &vertices[..]).unwrap(),
                            index: IndexBuffer::new(&state.display, PrimitiveType::TrianglesList, &indices[..]).unwrap(),
                        }))
                    }

                    fn buffer_3d((pos, normal, color, indices): (Vec<f32>, Vec<f32>, Vec<f32>, Vec<VertIdx>)) {
                        lua_assert!(pos.len() % 3 == 0, "positions not multiple of 3");
                        lua_assert!(normal.len() % 3 == 0, "normals not multiple of 3");
                        lua_assert!(color.len() % 4 == 0, "colors not multiple of 4");
                        lua_assert!(pos.len() == normal.len(), "not the same amount of positions as normals");
                        lua_assert!(pos.len() / 3 == color.len() / 4, "not the same amount of positions as colors");
                        let vertices = pos.chunks_exact(3).zip(normal.chunks_exact(3)).zip(color.chunks_exact(4)).map(|((pos, normal), color)| {
                            let qn = |f| (f*128.) as i8;
                            let qc = |f| (f*255.) as u8;
                            SimpleVertex {
                                pos: [pos[0], pos[1], pos[2]],
                                normal: [qn(normal[0]), qn(normal[1]), qn(normal[2]), 0],
                                color: [qc(color[0]), qc(color[1]), qc(color[2]), qc(color[3])],
                            }
                        }).collect::<Vec<_>>();
                        BufferRef::Buf3d(Rc::new(Buffer3d {
                            vertex: VertexBuffer::new(&state.display, &vertices[..]).unwrap(),
                            index: IndexBuffer::new(&state.display, PrimitiveType::TrianglesList, &indices[..]).unwrap(),
                        }))
                    }

                    fn camera_stack(()) {
                        CameraStack::new()
                    }

                    fn texture(path: String) {
                        use glium::texture::RawImage2d;

                        let img = image::io::Reader::open(&path)
                            .with_context(|| format!("image file \"{}\" not found", path))
                            .to_lua_err()?
                            .decode()
                            .with_context(|| format!("image file \"{}\" invalid or corrupted", path))
                            .to_lua_err()?
                            .to_rgba8();
                        let (w, h) = img.dimensions();
                        let tex = Texture2d::new(
                            &state.display,
                            RawImage2d::from_raw_rgba(img.into_vec(), (w, h))
                        ).unwrap();
                        TextureRef::new(tex)
                    }

                    fn uniforms(()) {
                        UniformStorage{vars: AssertSync(Rc::new(RefCell::new(Vec::new())))}
                    }

                    fn draw_params(()) {
                        LuaDrawParams::default()
                    }

                    fn font((font_data, size): (LuaString, u32)) {
                        FontRef{
                            rc: AssertSync(Rc::new(Font::new(state, font_data.as_bytes(), size).to_lua_err()?)),
                        }
                    }

                    fn dimensions(()) {
                        state.frame.borrow().get_dimensions()
                    }

                    fn clear((r, g, b, a, depth, stencil): (Option<f32>, Option<f32>, Option<f32>, Option<f32>, Option<f32>, Option<i32>)) {
                        let mut frame = state.frame.borrow_mut();
                        match (r, g, b, a, depth, stencil) {
                            (Some(r), Some(g), Some(b), a, None, None) => frame.clear_color(r, g, b, a.unwrap_or(0.)),
                            (Some(r), Some(g), Some(b), a, Some(d), None) => frame.clear_color_and_depth((r, g, b, a.unwrap_or(0.)), d),
                            (Some(r), Some(g), Some(b), a, None, Some(s)) => frame.clear_color_and_stencil((r, g, b, a.unwrap_or(0.)), s),
                            (Some(r), Some(g), Some(b), a, Some(d), Some(s)) => frame.clear_all((r, g, b, a.unwrap_or(0.)), d, s),
                            (None, None, None, None, Some(d), None) => frame.clear_depth(d),
                            (None, None, None, None, None, Some(s)) => frame.clear_stencil(s),
                            (None, None, None, None, Some(d), Some(s)) => frame.clear_depth_and_stencil(d, s),
                            _ => return Err(LuaError::RuntimeError("invalid arguments".into())),
                        }
                    }

                    fn draw((buf, shader, uniforms, params): (BufferRef, ShaderRef, UniformStorage, LuaDrawParams)) {
                        let mut frame = state.frame.borrow_mut();
                        match &buf {
                            BufferRef::NoBuf => Ok(()),
                            BufferRef::Buf2d(buf) => {
                                frame.draw(&buf.vertex, &buf.index, &shader.program, &uniforms, &params.params)
                            },
                            BufferRef::Buf3d(buf) => {
                                frame.draw(&buf.vertex, &buf.index, &shader.program, &uniforms, &params.params)
                            },
                        }.unwrap();
                    }

                    fn finish(()) {
                        state.frame.borrow_mut().set_finish().unwrap();
                        *state.frame.borrow_mut() = state.display.draw();
                    }

                    fn toggle_fullscreen(exclusive: bool) {
                        use glium::glutin::window::Fullscreen;
                        let win = state.display.gl_window();
                        let win = win.window();
                        if win.fullscreen().is_some() {
                            win.set_fullscreen(None);
                        }else{
                            if exclusive {
                                if let Some(mon) = win.current_monitor() {
                                    let mode = mon.video_modes().max_by_key(|mode| {
                                        (mode.bit_depth(), mode.size().width * mode.size().height, mode.refresh_rate())
                                    });
                                    if let Some(mode) = mode {
                                        win.set_fullscreen(Some(Fullscreen::Exclusive(mode)));
                                        return Ok(());
                                    }
                                }
                            }
                            win.set_fullscreen(Some(Fullscreen::Borderless(None)));
                        }
                    }
                },
            )
            .unwrap();
}
