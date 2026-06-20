use anyhow::{Context, Result, bail};

const MAX_IMAGE_SIZE_BYTES: usize = 10 * 1024 * 1024;
const MIN_IMAGE_WIDTH: u32 = 64;
const MIN_IMAGE_HEIGHT: u32 = 64;
const MAX_IMAGE_WIDTH: u32 = 6000;
const MAX_IMAGE_HEIGHT: u32 = 6000;

pub fn validate_upload_input(bytes: &[u8]) -> Result<()> {
    if bytes.is_empty() {
        bail!("image file cannot be empty");
    }

    if bytes.len() > MAX_IMAGE_SIZE_BYTES {
        bail!(
            "image file is too large: max {} bytes, got {} bytes",
            MAX_IMAGE_SIZE_BYTES,
            bytes.len()
        );
    }

    Ok(())
}

pub fn validate_image_format(format: image::ImageFormat) -> Result<()> {
    match format {
        image::ImageFormat::Jpeg | image::ImageFormat::WebP | image::ImageFormat::Png => {}
        _ => bail!("unsupported image format: {:?}", format),
    }

    Ok(())
}

pub fn validate_image_bytes(bytes: &[u8]) -> Result<()> {
    let image = image::load_from_memory(bytes).context("failed to decode image")?;

    let (width, height) = (image.width(), image.height());
    if width < MIN_IMAGE_WIDTH || height < MIN_IMAGE_HEIGHT {
        bail!(
            "image is too small: min {}x{}, got {}x{}",
            MIN_IMAGE_WIDTH,
            MIN_IMAGE_HEIGHT,
            width,
            height
        )
    }

    if width > MAX_IMAGE_WIDTH || height > MAX_IMAGE_HEIGHT {
        bail!(
            "image is too large: max {}x{}, got {}x{}",
            MAX_IMAGE_WIDTH,
            MAX_IMAGE_HEIGHT,
            width,
            height
        )
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    use ::image::{DynamicImage, ImageBuffer, ImageFormat, Rgba};

    fn encode_png(width: u32, height: u32) -> Vec<u8> {
        let image = ImageBuffer::from_pixel(width, height, Rgba([10, 20, 30, 255]));
        let mut bytes = Cursor::new(Vec::new());

        DynamicImage::ImageRgba8(image)
            .write_to(&mut bytes, ImageFormat::Png)
            .unwrap();

        bytes.into_inner()
    }

    #[test]
    fn validate_upload_input_accepts_non_empty_image_with_allowed_size() {
        validate_upload_input(&[1, 2, 3]).unwrap();
    }

    #[test]
    fn validate_upload_input_rejects_empty_bytes() {
        let error = validate_upload_input(&[]).unwrap_err();

        assert_eq!(error.to_string(), "image file cannot be empty");
    }

    #[test]
    fn validate_upload_input_rejects_too_large_bytes() {
        let bytes = vec![0; MAX_IMAGE_SIZE_BYTES + 1];

        let error = validate_upload_input(&bytes).unwrap_err();

        assert_eq!(
            error.to_string(),
            format!(
                "image file is too large: max {} bytes, got {} bytes",
                MAX_IMAGE_SIZE_BYTES,
                MAX_IMAGE_SIZE_BYTES + 1
            )
        );
    }

    #[test]
    fn validate_image_format_accepts_supported_formats() {
        for format in [
            image::ImageFormat::Jpeg,
            image::ImageFormat::Png,
            image::ImageFormat::WebP,
        ] {
            validate_image_format(format).unwrap();
        }
    }

    #[test]
    fn validate_image_format_rejects_unsupported_format() {
        let error = validate_image_format(image::ImageFormat::Gif).unwrap_err();

        assert_eq!(error.to_string(), "unsupported image format: Gif");
    }

    #[test]
    fn validate_image_bytes_accepts_valid_dimensions() {
        let bytes = encode_png(MIN_IMAGE_WIDTH, MIN_IMAGE_HEIGHT);

        validate_image_bytes(&bytes).unwrap();
    }

    #[test]
    fn validate_image_bytes_rejects_invalid_image_bytes() {
        let error = validate_image_bytes(b"not an image").unwrap_err();

        assert_eq!(error.to_string(), "failed to decode image");
    }

    #[test]
    fn validate_image_bytes_rejects_too_small_image() {
        let bytes = encode_png(MIN_IMAGE_WIDTH - 1, MIN_IMAGE_HEIGHT);

        let error = validate_image_bytes(&bytes).unwrap_err();

        assert_eq!(
            error.to_string(),
            format!(
                "image is too small: min {}x{}, got {}x{}",
                MIN_IMAGE_WIDTH,
                MIN_IMAGE_HEIGHT,
                MIN_IMAGE_WIDTH - 1,
                MIN_IMAGE_HEIGHT
            )
        );
    }

    #[test]
    fn validate_image_bytes_rejects_too_large_image() {
        let bytes = encode_png(MAX_IMAGE_WIDTH + 1, MIN_IMAGE_HEIGHT);

        let error = validate_image_bytes(&bytes).unwrap_err();

        assert_eq!(
            error.to_string(),
            format!(
                "image is too large: max {}x{}, got {}x{}",
                MAX_IMAGE_WIDTH,
                MAX_IMAGE_HEIGHT,
                MAX_IMAGE_WIDTH + 1,
                MIN_IMAGE_HEIGHT
            )
        );
    }
}
