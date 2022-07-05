use crate::{
    chunkmesh::{Mesher2, MesherCfg, ModelMesherCfg, ModelMesherKind},
    lua::{CameraFrame, CameraStack, LuaImage, LuaMatrix, LuaStringBuf, LuaVoxelModel},
    prelude::*,
    render::{GlobalCommand, MatrixOp, RenderCommand, RenderHandle},
};
use common::{lua_assert, lua_bail, lua_func, lua_lib, lua_type};
use glium::{
    framebuffer::{
        DepthRenderBuffer, DepthStencilRenderBuffer, RenderBuffer, SimpleFrameBuffer,
        StencilRenderBuffer,
    },
    texture::{DepthStencilTexture2d, DepthTexture2d, StencilTexture2d},
};

use super::LogicState;

#[derive(Clone)]
pub(crate) enum LuaBuffer {
    NoBuf,
    Buf2d(RenderObj<GpuBuffer<TexturedVertex>>),
    Buf3d(RenderObj<GpuBuffer<SimpleVertex>>),
    VoxelBuf(RenderObj<GpuBuffer<VoxelVertex>>),
}
impl LuaUserData for LuaBuffer {}

#[derive(Clone)]
pub struct MatrixStack {
    m: RenderObj<Vec<Mat4>>,
}
lua_type! {MatrixStack, lua, this,
    // Reset the entire matrix stack to a single identity matrix.
    fn reset() {
        unsafe {
            this.m.handle().push(RenderCommand::Matrix(this.clone(), MatrixOp::Reset));
        }
    }

    // Copy the entire matrix stack from another stack.
    fn reset_from(r: LuaAnyUserData) {
        let r = r.borrow::<MatrixStack>()?;
        unsafe {
            this.m.handle().push(RenderCommand::Matrix(this.clone(), MatrixOp::Copy(r.clone())));
        }
    }

    // Set the top matrix to identity.
    fn identity() {
        unsafe {
            this.m.handle().push(RenderCommand::Matrix(this.clone(), MatrixOp::Identity));
        }
    }

    // Set the top matrix to `top * r`.
    fn mul_right(r: LuaAnyUserData) {
        let r = r.borrow::<MatrixStack>()?;
        unsafe {
            this.m.handle().push(RenderCommand::Matrix(this.clone(), MatrixOp::MulRight(r.clone())));
        }
    }

    // Set the top matrix to `r * top`.
    fn mul_left(r: LuaAnyUserData) {
        let r = r.borrow::<MatrixStack>()?;
        unsafe {
            this.m.handle().push(RenderCommand::Matrix(this.clone(), MatrixOp::MulLeft(r.clone())));
        }
    }

    fn push() {
        unsafe {
            this.m.handle().push(RenderCommand::Matrix(this.clone(), MatrixOp::Push));
        }
    }

    fn pop() {
        unsafe {
            this.m.handle().push(RenderCommand::Matrix(this.clone(), MatrixOp::Pop));
        }
    }

    fn translate((x, y, z, dx, dy, dz): (f32, f32, f32, f32, f32, f32)) {
        unsafe {
            this.m.handle().push(RenderCommand::Matrix(this.clone(), MatrixOp::Translate {
                x: Vec3::new(x, y, z),
                dx: Vec3::new(dx, dy, dz),
            }));
        }
    }

    fn scale((x, y, z, dx, dy, dz): (f32, Option<f32>, Option<f32>, Option<f32>, Option<f32>, Option<f32>)) {
        let (x, dx) = match (x, y, z, dx, dy, dz) {
            (x, Some(dx), None, None, None, None) => {
                (Vec3::broadcast(x), Vec3::broadcast(dx))
            },
            (x, Some(y), Some(z), Some(dx), Some(dy), Some(dz)) => {
                (Vec3::new(x, y, z), Vec3::new(dx, dy, dz))
            }
            _ => lua_bail!("expected 2 or 6 scale arguments")
        };
        unsafe {
            this.m.handle().push(RenderCommand::Matrix(this.clone(), MatrixOp::Scale {
                x, dx,
            }));
        }
    }

    fn rotate_x((a, da): (f32, f32)) {
        unsafe {
            this.m.handle().push(RenderCommand::Matrix(this.clone(), MatrixOp::RotX(a, da)));
        }
    }
    fn rotate_y((a, da): (f32, f32)) {
        unsafe {
            this.m.handle().push(RenderCommand::Matrix(this.clone(), MatrixOp::RotY(a, da)));
        }
    }
    fn rotate_z((a, da): (f32, f32)) {
        unsafe {
            this.m.handle().push(RenderCommand::Matrix(this.clone(), MatrixOp::RotZ(a, da)));
        }
    }
    fn rotate((a, da, x, y, z): (f32, f32, f32, f32, f32)) {
        unsafe {
            this.m.handle().push(RenderCommand::Matrix(this.clone(), MatrixOp::Rotate {
                a, da,
                axis: Vec3::new(x, y, z),
            }));
        }
    }

    fn invert() {
        unsafe {
            this.m.handle().push(RenderCommand::Matrix(this.clone(), MatrixOp::Invert));
        }
    }

    fn perspective((fov, aspect, near, far): (f32, f32, f32, f32)) {
        unsafe {
            this.m.handle().push(RenderCommand::Matrix(this.clone(), MatrixOp::Perspective {
                fov, aspect, near, far,
            }));
        }
    }
    fn orthographic((xleft, xright, ydown, yup, znear, zfar): (f32, f32, f32, f32, f32, f32)) {
        unsafe {
            this.m.handle().push(RenderCommand::Matrix(this.clone(), MatrixOp::Orthographic {
                xleft, xright, ydown, yup, znear, zfar,
            }));
        }
    }
}

#[derive(Clone)]
pub(crate) struct LuaVoxelBuf {
    size: Int3,
    buf: RenderObj<GpuBuffer<VoxelVertex>>,
    atlas: LuaTexture<SrgbTexture2d>,
}
lua_type! {LuaVoxelBuf, lua, this,
    fn size() {
        (this.size.x, this.size.y, this.size.z)
    }

    fn buffer() {
        LuaBuffer::VoxelBuf(this.buf.clone())
    }

    fn atlas_linear() {
        use glium::uniforms::{
            MagnifySamplerFilter as Magnify, MinifySamplerFilter as Minify, SamplerBehavior,
            SamplerWrapFunction as Wrap,
        };
        let mut tex = this.atlas.clone();
        tex.sampling = SamplerBehavior {
            wrap_function: (Wrap::Repeat, Wrap::Repeat, Wrap::Repeat),
            minify_filter: Minify::Linear,
            magnify_filter: Magnify::Linear,
            ..default()
        };
        tex
    }

    fn atlas_nearest() {
        use glium::uniforms::{
            MagnifySamplerFilter as Magnify, MinifySamplerFilter as Minify, SamplerBehavior,
            SamplerWrapFunction as Wrap,
        };
        let mut tex = this.atlas.clone();
        tex.sampling = SamplerBehavior {
            wrap_function: (Wrap::Repeat, Wrap::Repeat, Wrap::Repeat),
            minify_filter: Minify::Nearest,
            magnify_filter: Magnify::Nearest,
            ..default()
        };
        tex
    }
}

pub(crate) enum StaticUniform {
    Float(f32),
    Vec2([f32; 2]),
    Vec3([f32; 3]),
    Vec4([f32; 4]),
    Mat4(MatrixStack),
    Texture2d(LuaTexture<Texture2d>),
    SrgbTexture2d(LuaTexture<SrgbTexture2d>),
}
impl StaticUniform {
    /// SAFETY: Must be called from the render thread.
    /// Additionally, the returned uniform value must have a short lifespan.
    /// In particular, it must die before any referenced textures are modified.
    /// However, the returned lifetime does not reflect this due to interior mutability.
    unsafe fn as_uniform(&self) -> UniformValue {
        match self {
            &Self::Float(v) => UniformValue::Float(v),
            &Self::Vec2(v) => UniformValue::Vec2(v),
            &Self::Vec3(v) => UniformValue::Vec3(v),
            &Self::Vec4(v) => UniformValue::Vec4(v),
            &Self::Mat4(v) => {
                let mat = *v.m.inner().last().expect("empty matrix stack");
                UniformValue::Mat4(mat.into())
            }
            Self::Texture2d(tex) => {
                let tex2d = &*tex.tex.inner();
                // SAFETY: Reborrowing extends lifetime, therefore `UniformValue` must only stay
                // alive until the texture is modified again.
                let tex2d = &*(tex2d as *const _);
                UniformValue::Texture2d(tex2d, Some(tex.sampling))
            }
            Self::SrgbTexture2d(tex) => {
                let tex2d = &*tex.tex.inner();
                // SAFETY: Reborrowing extends lifetime, therefore `UniformValue` must only stay
                // alive until the texture is modified again.
                let tex2d = &*(tex2d as *const _);
                UniformValue::SrgbTexture2d(tex2d, Some(tex.sampling))
            }
        }
    }
}

#[derive(Clone)]
pub(crate) struct LuaUniforms {
    obj: RenderObj<UniformStorage>,
}
lua_type! {LuaUniforms, lua, this,
    // SAFETY: Must be called from the logic thread, as well as all other `set_*` methods.
    mut fn set_float((idx, val): (usize, f32)) {
        unsafe { this.obj.handle().push(RenderCommand::Uniform(
            this.clone(),
            idx,
            StaticUniform::Float(val)
        )); }
    }
    mut fn set_vec2((idx, x, y): (usize, f32, f32)) {
        unsafe { this.obj.handle().push(RenderCommand::Uniform(
            this.clone(),
            idx,
            StaticUniform::Vec2([x, y]),
        )); }
    }
    mut fn set_vec3((idx, x, y, z): (usize, f32, f32, f32)) {
        unsafe { this.obj.handle().push(RenderCommand::Uniform(
            this.clone(),
            idx,
            StaticUniform::Vec3([x, y, z]),
        )); }
    }
    mut fn set_vec4((idx, x, y, z, w): (usize, f32, f32, f32, f32)) {
        unsafe { this.obj.handle().push(RenderCommand::Uniform(
            this.clone(),
            idx,
            StaticUniform::Vec4([x, y, z, w]),
        )); }
    }

    mut fn set_matrix((idx, mat): (usize, MatrixStack)) {
        unsafe { this.obj.handle().push(RenderCommand::Uniform(
            this.clone(),
            idx,
            StaticUniform::Mat4(mat),
        )); }
    }

    mut fn set_texture_2d((idx, tex): (usize, LuaTexture<SrgbTexture2d>)) {
        unsafe { this.obj.handle().push(RenderCommand::Uniform(
            this.clone(),
            idx,
            StaticUniform::SrgbTexture2d(tex),
        )); }
    }

    mut fn set_linear_texture_2d((idx, tex): (usize, LuaTexture<Texture2d>)) {
        unsafe { this.obj.handle().push(RenderCommand::Uniform(
            this.clone(),
            idx,
            StaticUniform::Texture2d(tex),
        )); }
    }
}

pub(crate) struct UniformStorage {
    vars: Vec<(String, StaticUniform)>,
}
impl Uniforms for UniformStorage {
    /// SAFETY: This method is safe to comply with the `glium` API, but all of the safety
    /// guarantees of `StaticUniform::as_uniform` must be upheld for all contained uniforms.
    fn visit_values<'a, F: FnMut(&str, UniformValue<'a>)>(&'a self, mut visit: F) {
        unsafe {
            for (name, val) in self.vars.iter() {
                visit(name, val.as_uniform());
            }
        }
    }
}

pub(crate) struct UniformsOverride<'a> {
    store: &'a UniformStorage,
    over: &'a [UniformValue<'a>],
}
impl<'a> UniformsOverride<'a> {
    pub fn new(store: &'a UniformStorage, over: &'a [UniformValue]) -> Self {
        Self { store, over }
    }
}
impl<'b> Uniforms for UniformsOverride<'b> {
    /// SAFETY: This method is safe to comply with the `glium` API, but all of the safety
    /// guarantees of `StaticUniform::as_uniform` must be upheld for all contained uniforms.
    fn visit_values<'a, F: FnMut(&str, UniformValue<'a>)>(&'a self, mut visit: F) {
        for ((name, _), &val) in self.store.vars.iter().zip(self.over.iter()) {
            visit(name, val);
        }
        unsafe {
            for (name, val) in self.store.vars.iter().skip(self.over.len()) {
                visit(name, val.as_uniform());
            }
        }
    }
}

pub(crate) struct Font {
    text: TextDisplay<Rc<FontTexture>>,
}
impl Font {
    fn new(state: &Rc<State>, font_data: &[u8], size: u32) -> Result<Self> {
        let tex = Rc::new(FontTexture::new(
            &state.display,
            font_data,
            size,
            (0..0x250).filter_map(|i| std::char::from_u32(i)),
        )?);
        let text = TextDisplay::new(&state.text_sys, tex, "");
        Ok(Self { text })
    }

    fn draw(
        &mut self,
        state: &Rc<State>,
        frame: &mut Frame,
        text: &str,
        mvp: Mat4,
        color: [f32; 4],
        draw_params: &DrawParameters,
    ) {
        self.text.set_text(text);
        glium_text_rusttype::draw_with_params(
            &self.text,
            &state.text_sys,
            frame,
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
pub(crate) struct LuaFont {
    pub rc: RenderObj<Font>,
}
lua_type! {LuaFont, lua, this,
    fn draw((text, mvp, draw_params, r, g, b, a): (LuaAnyUserData, MatrixStack, LuaDrawParams, f32, f32, f32, Option<f32>)) {
        let text = text.borrow::<LuaStringBuf>()?;
        let text = std::str::from_utf8(&text.text[..]).map_err(|_| "invalid utf-8").to_lua_err()?;
        unsafe {
            this.rc.handle().push(RenderCommand::Text {
                text: text.to_string(),
                mvp,
                params: draw_params,
                color: [r, g, b, a.unwrap_or(1.)],
            });
        }
    }

    fn draw_static((text, mvp, draw_params, r, g, b, a): (LuaString, MatrixStack, LuaDrawParams, f32, f32, f32, Option<f32>)) {
        let text = text.to_str()?;
        unsafe {
            this.rc.handle().push(RenderCommand::Text {
                text: text.to_string(),
                mvp,
                params: draw_params,
                color: [r, g, b, a.unwrap_or(1.)],
            });
        }
    }
}
#[derive(Clone)]
pub(crate) struct LuaShader {
    pub obj: RenderObj<Program>,
}
impl LuaUserData for LuaShader {}

pub(crate) struct LuaTexture<T>
where
    T: 'static,
{
    pub tex: RenderObj<T>,
    pub sampling: SamplerBehavior,
}
impl<T> LuaTexture<T> {
    fn default_sampling() -> SamplerBehavior {
        use glium::uniforms::{MagnifySamplerFilter, MinifySamplerFilter, SamplerWrapFunction};
        SamplerBehavior {
            minify_filter: MinifySamplerFilter::Nearest,
            magnify_filter: MagnifySamplerFilter::Nearest,
            wrap_function: (
                SamplerWrapFunction::Repeat,
                SamplerWrapFunction::Repeat,
                SamplerWrapFunction::Repeat,
            ),
            depth_texture_comparison: None,
            max_anisotropy: 1,
        }
    }
}
impl<T> Clone for LuaTexture<T> {
    fn clone(&self) -> Self {
        Self {
            tex: self.tex.clone(),
            sampling: self.sampling.clone(),
        }
    }
}
macro_rules! lua_textures {
    ($($ty:ty),*) => {$(
        lua_type! {LuaTexture<$ty>, lua, this,
            fn dimensions() {
                (this.tex.width(), this.tex.height())
            }

            mut fn set_min(filter: LuaString) {
                use glium::uniforms::MinifySamplerFilter::*;
                this.sampling.minify_filter = match filter.as_bytes() {
                    b"linear" => Linear,
                    b"nearest" => Nearest,
                    _ => lua_bail!("unknown minify filter '{}'", filter.to_str().unwrap_or_default())
                };
            }

            mut fn set_mag(filter: LuaString) {
                use glium::uniforms::MagnifySamplerFilter::*;
                this.sampling.magnify_filter = match filter.as_bytes() {
                    b"linear" => Linear,
                    b"nearest" => Nearest,
                    _ => lua_bail!("unknown magnify filter '{}'", filter.to_str().unwrap_or_default())
                };
            }

            mut fn set_wrap(wrap: LuaString) {
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
    )*};
}
lua_textures!(SrgbTexture2d, Texture2d);

pub(crate) struct LuaMesher {
    state: Rc<State>,
    mesher: Box<Mesher2<ModelMesherKind>>,
}
impl LuaMesher {
    fn new(state: Rc<State>, cfg: &ModelMesherCfg) -> Self {
        Self {
            state,
            mesher: Box::new(Mesher2::new(&cfg.cfg, ModelMesherKind::new(cfg))),
        }
    }
}
lua_type! {LuaMesher, lua, this,
    mut fn mesh(m: LuaAnyUserData) {
        let m = m.borrow::<LuaVoxelModel>()?;
        let sz = m.0.size();
        assert_eq!(m.0.data().len(), (sz.x * sz.y * sz.z) as usize);

        this.mesher.make_model_mesh(&m.0);
        let buf = Rc::new(this.mesher.mesh.make_buffer(&this.state.display));
        let atlas = LuaTexture::new(this.mesher.atlas.make_texture(&this.state.display));
        LuaVoxelBuf {
            size: sz,
            buf,
            atlas,
        }
    }
}

#[derive(Clone)]
pub(crate) struct LuaDrawParams {
    pub depth: glium::draw_parameters::Depth,
    pub stencil: glium::draw_parameters::Stencil,
    pub blend: glium::draw_parameters::Blend,
    pub color_mask: (bool, bool, bool, bool),
    pub clip_planes: u32,
    pub backface_culling: glium::draw_parameters::BackfaceCullingMode,
    pub multisampling: bool,
    pub dithering: bool,
    pub viewport: Option<glium::Rect>,
    pub scissor: Option<glium::Rect>,
    pub draw_primitives: bool,
    pub smooth: Option<glium::draw_parameters::Smooth>,
    pub provoking_vertex: glium::draw_parameters::ProvokingVertex,
    pub primitive_bounding_box: (
        ops::Range<f32>,
        ops::Range<f32>,
        ops::Range<f32>,
        ops::Range<f32>,
    ),
    pub primitive_restart_index: bool,
    pub polygon_offset: glium::draw_parameters::PolygonOffset,
}
impl Default for LuaDrawParams {
    fn default() -> Self {
        Self {
            depth: default(),
            stencil: default(),
            blend: default(),
            color_mask: (true, true, true, true),
            backface_culling: glium::draw_parameters::BackfaceCullingMode::CullingDisabled,
            clip_planes: 0,
            multisampling: true,
            dithering: true,
            viewport: None,
            scissor: None,
            draw_primitives: true,
            smooth: None,
            provoking_vertex: glium::draw_parameters::ProvokingVertex::LastVertex,
            primitive_bounding_box: (-1.0..1.0, -1.0..1.0, -1.0..1.0, -1.0..1.0),
            primitive_restart_index: false,
            polygon_offset: Default::default(),
        }
    }
}
impl LuaDrawParams {
    pub fn to_params(&self) -> DrawParameters {
        DrawParameters {
            depth: self.depth,
            stencil: self.stencil,
            blend: self.blend,
            color_mask: self.color_mask,
            backface_culling: self.backface_culling,
            clip_planes_bitmask: self.clip_planes,
            multisampling: self.multisampling,
            dithering: self.dithering,
            viewport: self.viewport,
            scissor: self.scissor,
            draw_primitives: self.draw_primitives,
            smooth: self.smooth,
            provoking_vertex: self.provoking_vertex,
            primitive_bounding_box: self.primitive_bounding_box,
            primitive_restart_index: self.primitive_restart_index,
            polygon_offset: self.polygon_offset,
            ..default()
        }
    }
}
lua_type! {LuaDrawParams, lua, this,
    mut fn set_depth((test, write, clamp, near, far): (LuaString, bool, Option<LuaString>, Option<f32>, Option<f32>)) {
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
        this.depth = Depth {
            test,
            write,
            clamp,
            range: (near.unwrap_or(0.), far.unwrap_or(1.)),
        };
    }

    mut fn set_color_mask((r, g, b, a): (bool, Option<bool>, Option<bool>, Option<bool>)) {
        this.color_mask = match (r, g, b, a) {
            (d, None, None, None) => (d, d, d, d),
            (r, Some(g), Some(b), Some(a)) => (r, g, b, a),
            _ => lua_bail!("invalid color channel mask"),
        };
    }

    mut fn set_color_blend((func, src, dst): (LuaString, LuaString, LuaString)) {
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
        this.blend.color = func;
    }

    mut fn set_alpha_blend((func, src, dst): (LuaString, LuaString, LuaString)) {
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
        this.blend.alpha = func;
    }

    mut fn set_cull(winding: LuaString) {
        use glium::draw_parameters::BackfaceCullingMode::*;
        let cull = match winding.as_bytes() {
            b"cw" => CullClockwise,
            b"none" => CullingDisabled,
            b"ccw" => CullCounterClockwise,
            _ => lua_bail!("unknown cull winding")
        };
        this.backface_culling = cull;
    }

    mut fn set_stencil((winding, test, refval, pass, fail, depthfail): (LuaString, LuaString, i32, Option<LuaString>, Option<LuaString>, Option<LuaString>)) {
        use glium::draw_parameters::{StencilTest::*, StencilOperation::*};
        let p = &mut this.stencil;
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

    mut fn set_stencil_ref(refval: i32) {
        this.stencil.reference_value_counter_clockwise = refval;
    }

    mut fn set_clip_planes(planes: u32) {
        this.clip_planes = planes;
    }

    mut fn set_multisampling(enable: bool) {
        this.multisampling = enable;
    }

    mut fn set_dithering(enable: bool) {
        this.dithering = enable;
    }

    mut fn set_viewport((left, bottom, w, h): (Option<u32>, Option<u32>, Option<u32>, Option<u32>)) {
        this.viewport = match (left, bottom, w, h) {
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

    mut fn set_scissor((left, bottom, w, h): (Option<u32>, Option<u32>, Option<u32>, Option<u32>)) {
        this.scissor = match (left, bottom, w, h) {
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

    mut fn set_draw_primitives(enable: bool) {
        this.draw_primitives = enable;
    }

    mut fn set_smooth(smooth: Option<LuaString>) {
        use glium::draw_parameters::Smooth::*;
        this.smooth = match smooth.as_ref().map(|s| s.as_bytes()) {
            Some(b"fastest") => Some(Fastest),
            Some(b"nicest") => Some(Nicest),
            Some(b"dont_care") => Some(DontCare),
            None => None,
            _ => lua_bail!("invalid smooth mode"),
        };
    }

    mut fn set_bounding_box((x0, x1, y0, y1, z0, z1, w0, w1): (f32, f32, f32, f32, f32, f32, f32, f32)) {
        this.primitive_bounding_box = (x0..x1, y0..y1, z0..z1, w0..w1);
    }

    mut fn set_polygon_offset((enable, factor, units): (bool, f32, f32)) {
        use glium::draw_parameters::PolygonOffset;
        this.polygon_offset = PolygonOffset {
            factor,
            units,
            fill: enable,
            point: false,
            line: false,
        };
    }
}

pub(crate) fn open_gfx_lib(state: &Rc<LogicState>, lua: LuaContext) {
    let handle = &state.render;
    lua.globals()
            .set(
                "gfx",
                lua_lib! {lua, handle,
                    fn shader((vertex, fragment): (String, String)) {
                        LuaShader {
                            obj: handle.new_obj(|state| {
                                let shader = program!{&state.display,
                                    110 => {
                                        vertex: &*vertex,
                                        fragment: &*fragment,
                                    }
                                }.expect("failed to compile shader program");
                                shader
                            }),
                        }
                    }

                    fn buffer_empty(()) {
                        LuaBuffer::NoBuf
                    }

                    fn buffer_2d((pos, tex, indices): (Vec<f32>, Vec<f32>, Vec<VertIdx>)) {
                        lua_assert!(pos.len() % 2 == 0, "positions not multiple of 2");
                        lua_assert!(tex.len() % 2 == 0, "texcoords not multiple of 4");
                        lua_assert!(pos.len() == tex.len(), "not the same amount of positions as texcoords");
                        let vertices = pos.chunks_exact(2).zip(tex.chunks_exact(2)).map(|(pos, tex)| {
                            TexturedVertex {pos: [pos[0], pos[1]], tex: [tex[0], tex[1]]}
                        }).collect::<Vec<_>>();
                        LuaBuffer::Buf2d(handle.new_obj(|state| GpuBuffer {
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
                        LuaBuffer::Buf3d(handle.new_obj(|state| GpuBuffer {
                            vertex: VertexBuffer::new(&state.display, &vertices[..]).unwrap(),
                            index: IndexBuffer::new(&state.display, PrimitiveType::TrianglesList, &indices[..]).unwrap(),
                        }))
                    }

                    fn mesher(cfg: LuaValue) {
                        let cfg = rlua_serde::from_value(cfg)?;
                        //LuaMesher::new(state.clone(), &cfg)
                        todo!()
                    }

                    fn camera_stack(()) {
                        CameraStack::new()
                    }

                    fn texture(img: LuaAnyUserData) {
                        use glium::texture::RawImage2d;

                        let img = img.borrow::<LuaImage>()?;
                        let img = &img.img;
                        let (w, h) = img.dimensions();
                        LuaTexture {
                            tex: handle.new_obj(|state| SrgbTexture2d::new(
                                &state.display,
                                RawImage2d::from_raw_rgba(img.to_vec(), (w, h))
                            ).unwrap()),
                            sampling: LuaTexture::<()>::default_sampling(),
                        }
                    }

                    fn linear_texture(img: LuaAnyUserData) {
                        use glium::texture::RawImage2d;

                        let img = img.borrow::<LuaImage>()?;
                        let img = &img.img;
                        let (w, h) = img.dimensions();
                        LuaTexture {
                            tex: handle.new_obj(|state| Texture2d::new(
                                &state.display,
                                RawImage2d::from_raw_rgba(img.to_vec(), (w, h))
                            ).unwrap()),
                            sampling: LuaTexture::<()>::default_sampling(),
                        }
                    }

                    fn uniforms(vars: LuaMultiValue) {
                        let vars = vars.into_iter().map(|v| {
                            Ok((String::from_lua(v, lua)?, StaticUniform::Float(0.)))
                        }).collect::<LuaResult<Vec<_>>>()?;
                        LuaUniforms {
                            obj: handle.new_obj(|_| UniformStorage {
                                vars,
                            }),
                        }
                    }

                    fn draw_params(()) {
                        LuaDrawParams::default()
                    }

                    fn font((font_data, size): (LuaString, u32)) {
                        let font_data = font_data.as_bytes().to_vec();
                        LuaFont {
                            rc: handle.new_obj(move |state| Font::new(state, &font_data[..], size).expect("failed to create font")),
                        }
                    }

                    fn dimensions(()) {
                        handle.dimensions()
                    }

                    fn clear((r, g, b, a, depth, stencil): (Option<f32>, Option<f32>, Option<f32>, Option<f32>, Option<f32>, Option<i32>)) {
                        let color = match (r, g, b, a) {
                            (Some(r), Some(g), Some(b), a) => Some((r, g, b, a.unwrap_or(0.))),
                            (None, None, None, None) => None,
                            _ => lua_bail!("invalid clear color"),
                        };
                        handle.push(RenderCommand::Clear {
                            color,
                            depth,
                            stencil,
                        });
                    }

                    fn draw((buf, shader, uniforms, params): (LuaBuffer, LuaShader, LuaUniforms, LuaDrawParams)) {
                        handle.push(RenderCommand::Draw {
                            buf,
                            shader,
                            uniforms,
                            params,
                        });
                    }

                    fn finish(()) {
                        handle.finish();
                    }

                    fn toggle_fullscreen(exclusive: bool) {
                        handle.global_cmd(GlobalCommand::ToggleFullscreen{exclusive});
                    }
                },
            )
            .unwrap();
}
