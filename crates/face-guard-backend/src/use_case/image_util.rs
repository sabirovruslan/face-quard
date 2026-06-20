use anyhow::{Context, Result};

use crate::use_case::image_validation::validate_image_format;

pub fn extract_image_format(bytes: &[u8]) -> Result<image::ImageFormat> {
    let format = image::guess_format(bytes).context("failed to detect image format")?;

    validate_image_format(format)?;

    Ok(format)
}

pub fn image_content_type(format: image::ImageFormat) -> &'static str {
    match format {
        image::ImageFormat::Jpeg => "image/jpeg",
        image::ImageFormat::WebP => "image/webp",
        image::ImageFormat::Png => "image/png",
        _ => "application/octet-stream",
    }
}

pub fn image_extension(format: image::ImageFormat) -> &'static str {
    match format {
        image::ImageFormat::Jpeg => "jpg",
        image::ImageFormat::WebP => "webp",
        image::ImageFormat::Png => "png",
        _ => "bin",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    use ::image::{DynamicImage, ImageBuffer, ImageFormat, Rgba};

    fn encode_image(format: ImageFormat) -> Vec<u8> {
        let image = ImageBuffer::from_pixel(64, 64, Rgba([10, 20, 30, 255]));
        let mut bytes = Cursor::new(Vec::new());

        DynamicImage::ImageRgba8(image)
            .write_to(&mut bytes, format)
            .unwrap();

        bytes.into_inner()
    }

    #[test]
    fn extract_image_format_returns_supported_format() {
        let png = encode_image(ImageFormat::Png);
        let jpeg = encode_image(ImageFormat::Jpeg);

        assert_eq!(extract_image_format(&png).unwrap(), ImageFormat::Png);
        assert_eq!(extract_image_format(&jpeg).unwrap(), ImageFormat::Jpeg);
    }

    #[test]
    fn extract_image_format_rejects_unknown_bytes() {
        let error = extract_image_format(b"not an image").unwrap_err();

        assert_eq!(error.to_string(), "failed to detect image format");
    }

    #[test]
    fn extract_image_format_rejects_unsupported_format() {
        let gif_header = b"GIF89a\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\xff\xff\xff";

        let error = extract_image_format(gif_header).unwrap_err();

        assert_eq!(error.to_string(), "unsupported image format: Gif");
    }

    #[test]
    fn image_content_type_returns_known_mime_types() {
        assert_eq!(image_content_type(ImageFormat::Jpeg), "image/jpeg");
        assert_eq!(image_content_type(ImageFormat::WebP), "image/webp");
        assert_eq!(image_content_type(ImageFormat::Png), "image/png");
    }

    #[test]
    fn image_content_type_falls_back_for_unknown_format() {
        assert_eq!(
            image_content_type(ImageFormat::Gif),
            "application/octet-stream"
        );
    }

    #[test]
    fn image_extension_returns_known_extensions() {
        assert_eq!(image_extension(ImageFormat::Jpeg), "jpg");
        assert_eq!(image_extension(ImageFormat::WebP), "webp");
        assert_eq!(image_extension(ImageFormat::Png), "png");
    }

    #[test]
    fn image_extension_falls_back_for_unknown_format() {
        assert_eq!(image_extension(ImageFormat::Gif), "bin");
    }
}
