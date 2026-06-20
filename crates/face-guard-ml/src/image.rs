use anyhow::{Context, Result, bail};
use image::imageops::FilterType;
use ndarray::Array4;

pub(crate) fn preprocess_face_image(image_bytes: &[u8]) -> Result<Array4<f32>> {
    let image = image::load_from_memory(image_bytes).context("failed to decode image")?;

    let rgb = image.to_rgba8();

    let resized = image::imageops::resize(&rgb, 112, 112, FilterType::Triangle);

    let mut input = Array4::<f32>::zeros((1, 3, 112, 112));

    for y in 0..112 {
        for x in 0..112 {
            let pixel = resized.get_pixel(x, y);

            let r = pixel[0] as f32;
            let g = pixel[1] as f32;
            let b = pixel[2] as f32;

            input[[0, 0, y as usize, x as usize]] = normalize_pixel(r);
            input[[0, 1, y as usize, x as usize]] = normalize_pixel(g);
            input[[0, 2, y as usize, x as usize]] = normalize_pixel(b);
        }
    }

    Ok(input)
}

fn normalize_pixel(value: f32) -> f32 {
    (value - 127.5) / 127.5
}

pub(crate) fn normalize_l2(values: &mut [f32]) -> Result<()> {
    let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();

    if norm == 0.0 || !norm.is_finite() {
        bail!("invalid embedding norm: {norm}")
    }

    for value in values {
        *value /= norm;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    use ::image::{DynamicImage, ImageBuffer, ImageFormat, Rgba};

    const EPSILON: f32 = 1e-6;

    fn encode_png(pixel: Rgba<u8>) -> Vec<u8> {
        let image = ImageBuffer::from_pixel(1, 1, pixel);
        let mut bytes = Cursor::new(Vec::new());

        DynamicImage::ImageRgba8(image)
            .write_to(&mut bytes, ImageFormat::Png)
            .unwrap();

        bytes.into_inner()
    }

    fn assert_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() <= EPSILON,
            "expected {actual} to be close to {expected}"
        );
    }

    #[test]
    fn preprocess_face_image_decodes_resizes_and_normalizes_image() {
        let image_bytes = encode_png(Rgba([255, 0, 127, 64]));

        let tensor = preprocess_face_image(&image_bytes).unwrap();

        assert_eq!(tensor.shape(), &[1, 3, 112, 112]);
        assert_close(tensor[[0, 0, 0, 0]], normalize_pixel(255.0));
        assert_close(tensor[[0, 1, 56, 56]], normalize_pixel(0.0));
        assert_close(tensor[[0, 2, 111, 111]], normalize_pixel(127.0));
    }

    #[test]
    fn preprocess_face_image_rejects_invalid_image_bytes() {
        let error = preprocess_face_image(b"not an image").unwrap_err();

        assert_eq!(error.to_string(), "failed to decode image");
    }

    #[test]
    fn normalize_l2_scales_values_to_unit_norm() {
        let mut values = vec![3.0, 4.0];

        normalize_l2(&mut values).unwrap();

        assert_close(values[0], 0.6);
        assert_close(values[1], 0.8);

        let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
        assert_close(norm, 1.0);
    }

    #[test]
    fn normalize_l2_rejects_zero_norm() {
        let mut values = vec![0.0, 0.0];

        let error = normalize_l2(&mut values).unwrap_err();

        assert_eq!(error.to_string(), "invalid embedding norm: 0");
    }

    #[test]
    fn normalize_l2_rejects_non_finite_norm() {
        let mut values = vec![f32::INFINITY];

        let error = normalize_l2(&mut values).unwrap_err();

        assert!(error.to_string().starts_with("invalid embedding norm:"));
    }
}
