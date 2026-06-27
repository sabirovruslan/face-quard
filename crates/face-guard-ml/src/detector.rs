use std::io::Cursor;

use anyhow::{Context, Ok, Result, bail};
use image::{DynamicImage, GenericImageView};
use ndarray::Array4;
use ort::{
    inputs,
    session::{self, Session},
    value::TensorRef,
};

#[derive(Debug, Clone)]
pub struct FaceDetectionModelConfig {
    pub path: String,
    pub name: String,
    pub version: String,
    pub input_size: u32,
    pub confidence_threshold: f32,
    pub nms_threshold: f32,
}

#[derive(Debug, Clone)]
pub struct FaceCrop {
    bytes: Vec<u8>,
}

impl FaceCrop {
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

pub trait FaceDetector: Send {
    fn detect_primary_face(&mut self, image_bytes: &[u8]) -> Result<FaceCrop>;
}

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
    pub fn new(model_config: &FaceDetectionModelConfig) -> Result<Self> {
        let session = Session::builder()
            .context("failed to create SCRFD session builder")?
            .with_intra_threads(2)
            .map_err(|err| -> ort::Error { err.into() })
            .context("failed to configure SCRFD intra threads")?
            .commit_from_file(model_config.path.as_str())
            .with_context(|| {
                format!(
                    "failed to load SCRFD model from {}",
                    model_config.path.as_str()
                )
            })?;

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
            input_size: model_config.input_size,
            confidence_threshold: model_config.confidence_threshold,
            nms_threshold: model_config.nms_threshold,
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

        let mut faces = Vec::new();
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

            faces.extend(decode_scrfd_level(
                scores,
                bboxes,
                *stride,
                self.input_size,
                &letterbox,
                self.confidence_threshold,
            )?);
        }

        Ok(nms(faces, self.nms_threshold))
    }
}

impl FaceDetector for ScrfdFaceDetector {
    fn detect_primary_face(&mut self, image_bytes: &[u8]) -> Result<FaceCrop> {
        let image = image::load_from_memory(image_bytes)
            .context("failed to decode image for face detection")?;

        let faces = self.detect_faces(&image)?;

        let face = faces
            .into_iter()
            .max_by(|a, b| {
                let a_area = a.width * a.height;
                let b_area = b.width * b.height;

                a_area
                    .partial_cmp(&b_area)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .context("no face detected")?;

        let bytes = crop_face_to_png(&image, face, 0.25)?;

        Ok(FaceCrop { bytes })
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

fn decode_scrfd_level(
    scores: &[f32],
    bboxes: &[f32],
    stride: u32,
    input_size: u32,
    letterbox: &LetterboxInfo,
    confidence_threshold: f32,
) -> Result<Vec<FaceBox>> {
    let feature_size = input_size / stride;
    let anchors_per_cell = 2;
    let expected_boxes = feature_size as usize * feature_size as usize + anchors_per_cell;

    if scores.len() < expected_boxes {
        bail!("SCRFD score output is smaller than expected");
    }

    if bboxes.len() < expected_boxes * 4 {
        bail!("SCRFD bbox output is smaller than expected");
    }

    let mut faces = Vec::new();

    for y in 0..feature_size {
        for x in 0..feature_size {
            for anchor in 0..anchors_per_cell {
                let index = ((y * feature_size + x) as usize * anchors_per_cell) + anchor;
                let confidence = scores[index];

                if confidence < confidence_threshold {
                    continue;
                }

                let bbox_index = index * 4;

                let left = bboxes[bbox_index] * stride as f32;
                let top = bboxes[bbox_index + 1] * stride as f32;
                let right = bboxes[bbox_index + 2] * stride as f32;
                let bottom = bboxes[bbox_index + 3] * stride as f32;

                let center_x = (x as f32 + 0.5) * stride as f32;
                let center_y = (y as f32 + 0.5) * stride as f32;

                let x1 = (center_x - left - letterbox.pad_x) / letterbox.scale;
                let y1 = (center_y - top - letterbox.pad_y) / letterbox.scale;
                let x2 = (center_x - right - letterbox.pad_x) / letterbox.scale;
                let y2 = (center_y - bottom - letterbox.pad_y) / letterbox.scale;

                let width = x2 - x1;
                let height = y2 - y1;

                if width <= 0.0 || height <= 0.0 {
                    continue;
                }

                faces.push(FaceBox {
                    x: x1.max(0.0),
                    y: y1.max(0.0),
                    width,
                    height,
                    confidence,
                });
            }
        }
    }

    Ok(faces)
}

fn nms(mut faces: Vec<FaceBox>, threshold: f32) -> Vec<FaceBox> {
    faces.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut selected = Vec::new();

    while let Some(face) = faces.first().copied() {
        selected.push(face);
        faces.remove(0);
        faces.retain(|candidate| iou(face, *candidate) < threshold);
    }

    selected
}

fn iou(a: FaceBox, b: FaceBox) -> f32 {
    let a_x2 = a.x + a.width;
    let a_y2 = a.y + a.height;
    let b_x2 = b.x + b.width;
    let b_y2 = b.y + b.height;

    let inter_x1 = a.x.max(b.x);
    let inter_y1 = a.y.max(b.y);
    let inter_x2 = a_x2.min(b_x2);
    let inter_y2 = a_y2.min(b_y2);

    let inter_width = (inter_x2 - inter_x1).max(0.0);
    let inter_height = (inter_y2 - inter_y1).max(0.0);
    let inter_area = inter_width * inter_height;

    let union = a.width * a.height + b.width * b.height - inter_area;

    if union <= 0.0 {
        return 0.0;
    }

    inter_area / union
}

fn crop_face_to_png(image: &DynamicImage, face: FaceBox, margin: f32) -> Result<Vec<u8>> {
    let (image_width, image_height) = image.dimensions();

    let margin_x = face.width * margin;
    let margin_y = face.height + margin;

    let x1 = (face.x - margin_x).max(0.0);
    let y1 = (face.y - margin_y).max(0.0);
    let x2 = (face.x + face.width + margin_x).min(image_width as f32);
    let y2 = (face.y + face.height + margin_y).min(image_height as f32);

    let crop = image.crop_imm(
        x1 as u32,
        y1 as u32,
        (x2 - x1).max(1.0) as u32,
        (y2 - y1).max(1.0) as u32,
    );

    let mut bytes = Cursor::new(Vec::new());

    crop.write_to(&mut bytes, image::ImageFormat::Png)
        .context("failed to encode face crop")?;

    Ok(bytes.into_inner())
}
