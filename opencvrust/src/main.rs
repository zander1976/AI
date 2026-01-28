use opencv::{
    Result,
    core::{self as cv_core, Mat, Point, Rect, Scalar, Size, Vector},
    dnn, highgui, imgproc,
    prelude::*,
    videoio,
};

fn main() -> Result<()> {
    let config = "deploy.prototxt";
    let model = "res10_300x300_ssd_iter_140000_fp16.caffemodel"; // or full version

    let mut net = dnn::read_net_from_caffe(config, model)?;
    net.set_prefer_target(dnn::DNN_TARGET_CPU)?; // or DNN_TARGET_CUDA if you have it

    let mut cap = videoio::VideoCapture::new(0, videoio::CAP_ANY)?; // Try CAP_AVFOUNDATION on macOS if issues
    if !cap.is_opened()? {
        return Err("Cannot open camera".into());
    }

    let mut frame = Mat::default();
    loop {
        cap.read(&mut frame)?;
        if frame.size()?.width <= 0 {
            break;
        }

        let blob = dnn::blob_from_image_to(
            &frame,
            1.0,
            Size::new(300, 300),
            Scalar::new(104.0, 177.0, 123.0, 0.0),
            false,
            false,
            cv_core::CV_32F,
        )?;

        net.set_input(&blob, "", 1.0, Scalar::default())?;
        let mut detections = net.forward("")?;

        let (h, w) = (frame.rows(), frame.cols());
        for i in 0..detections.rows() {
            let confidence = detections.at_2d::<f32>(i, 2)?.clone();
            if *confidence > 0.5 {
                let box_vals = detections.row(i).at_row::<f32>(3)?;
                let x1 = (box_vals[0] * w as f32) as i32;
                let y1 = (box_vals[1] * h as f32) as i32;
                let x2 = (box_vals[2] * w as f32) as i32;
                let y2 = (box_vals[3] * h as f32) as i32;

                // Padding (optional)
                let pad_x = ((x2 - x1) as f32 * 0.2) as i32;
                let pad_y = ((y2 - y1) as f32 * 0.3) as i32;
                let x1 = (x1 - pad_x).max(0);
                let y1 = (y1 - pad_y).max(0);
                let x2 = (x2 + pad_x).min(w);
                let y2 = (y2 + pad_y).min(h);

                imgproc::rectangle(
                    &mut frame,
                    Rect::new(x1, y1, x2 - x1, y2 - y1),
                    Scalar::new(0.0, 255.0, 0.0, 0.0),
                    2,
                    imgproc::LINE_8,
                    0,
                )?;

                let text = format!("{:.1}%", *confidence * 100.0);
                imgproc::put_text(
                    &mut frame,
                    &text,
                    Point::new(x1, y1 - 10),
                    imgproc::FONT_HERSHEY_SIMPLEX,
                    0.5,
                    Scalar::new(0.0, 255.0, 0.0, 0.0),
                    1,
                    imgproc::LINE_8,
                    false,
                )?;
            }
        }

        highgui::imshow("Face Detection - Rust", &frame)?;
        if highgui::wait_key(10)? == 'q' as i32 {
            break;
        }
    }

    Ok(())
}
