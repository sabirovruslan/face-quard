use anyhow::{Context, Ok, Result, bail};
use image::{DynamicImage, GenericImageView};
use ndarray::Array4;
use ort::{
    inputs,
    session::{self, Session},
    value::TensorRef,
};

#[derive(Debug, Clone, Copy)]
pub struct FaceBox {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    confidence: f32,
}

#[derive(Debug)]
pub struct ScrfdFaceDetector {
    session: Session,
    input_name: String,
    output_names: Vec<String>,
    input_size: u32,
    confidence_threshold: f32,
    nms_threshold: f32,
}

#[derive(Debug, Clone, Copy)]
struct LetterboxInfo {
    scale: f32,
    pad_x: f32,
    pad_y: f32,
}

impl ScrfdFaceDetector {
    pub fn new(
        model_path: &str,
        input_size: u32,
        confidence_threshold: f32,
        nms_threshold: f32,
    ) -> Result<Self> {
        let session = Session::builder()
            .context("failed to create SCRFD session builder")?
            .with_intra_threads(2)
            .map_err(|err| -> ort::Error { err.into() })
            .context("failed to configure SCRFD intra threads")?
            .commit_from_file(model_path)
            .with_context(|| format!("failed to load SCRFD model from {model_path}"))?;

        let input_name = session
            .inputs()
            .first()
            .context("SCRFD model has no inputs")?
            .name()
            .to_string();

        let output_names = session
            .outputs()
            .iter()
            .map(|output| output.name().to_string())
            .collect::<Vec<String>>();

        if output_names.len() < 6 {
            bail!("SCRFD model must have at least 6 outputs");
        }

        Ok(Self {
            session,
            input_name,
            output_names,
            input_size,
            confidence_threshold,
            nms_threshold,
        })
    }

    fn detect_faces(&mut self, image: &DynamicImage) -> Result<Vec<FaceBox>> {
        let (input, letterbox) = preprocess_detector_image(image, self.input_size)?;

        let input_tensor =
            TensorRef::from_array_view(&input).context("failed to create SCRFD input tensor")?;

        let outputs = self
            .session
            .run(inputs![self.input_name.as_str() => input_tensor])
            .context("failed to run SCRFD model")?;

        let faces = Vec::new();
        let strides = [8_u32, 16, 32];

        for (level, stride) in strides.iter().enumerate() {
            let score_output = outputs
                .get(self.output_names[level].as_str())
                .context("missing SCRFD score output")?;

            let bbox_output = outputs
                .get(self.output_names[level + 3].as_str())
                .context("missing SCRFD bbox output")?;

            let (_, scores) = score_output
                .try_extract_tensor::<f32>()
                .context("failed to extract SCRFD scores")?;

            let (_, bboxes) = bbox_output
                .try_extract_tensor::<f32>()
                .context("failed to extract SCRFD bboxes")?;
        }

        todo!()
    }
}

fn preprocess_detector_image(
    image: &DynamicImage,
    input_size: u32,
) -> Result<(Array4<f32>, LetterboxInfo)> {
    let (src_width, src_height) = image.dimensions();
    let scale = (input_size as f32 / src_width as f32).min((input_size as f32 / src_height as f32));

    let resized_width = (src_width as f32 * scale).round() as u32;
    let resized_height = (src_height as f32 * scale).round() as u32;

    let rgb = image.to_rgb8();
    let resized = image::imageops::resize(
        &rgb,
        resized_width,
        resized_height,
        image::imageops::FilterType::Triangle,
    );

    let pad_x = ((input_size - resized_width) / 2) as f32;
    let pad_y = ((input_size - resized_height) / 2) as f32;

    let mut input = Array4::<f32>::zeros((1, 3, input_size as usize, input_size as usize));

    for y in 0..resized_height {
        for x in 0..resized_width {
            let pixel = resized.get_pixel(x, y);

            let dst_x = x + pad_x as u32;
            let dst_y = y + pad_y as u32;

            input[[0, 0, dst_y as usize, dst_x as usize]] = (pixel[0] as f32 - 127.5) / 128.0;
            input[[0, 1, dst_y as usize, dst_x as usize]] = (pixel[1] as f32 - 127.5) / 128.0;
            input[[0, 2, dst_y as usize, dst_x as usize]] = (pixel[2] as f32 - 127.5) / 128.0;
        }
    }

    Ok((
        input,
        LetterboxInfo {
            scale,
            pad_x,
            pad_y,
        },
    ))
}
