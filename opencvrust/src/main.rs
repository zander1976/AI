use opencv::{
    Result,
    core::{self as cv_core, Mat, Point, Rect, Scalar, Size},
    dnn, highgui, imgproc,
    prelude::*,
    videoio,
};

fn main() -> Result<()> {
    let config_file = "deploy.prototxt";
    let model_file = "res10_300x300_ssd_iter_140000_fp16.caffemodel";

    let mut net = dnn::read_net_from_caffe(config_file, model_file)?;
    net.set_preferable_target(dnn::DNN_TARGET_CPU)?;

    let mut cap = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;
    // let mut cap = videoio::VideoCapture::new(0, videoio::CAP_AVFOUNDATION)?; // macOS fallback

    if !cap.is_opened()? {
        return Err(opencv::Error::new(
            cv_core::StsError,
            "Cannot open camera".to_string(),
        ));
    }

    println!("Press 'q' to quit");

    let mut frame = Mat::default();
    loop {
        cap.read(&mut frame)?;
        if frame.size()?.width <= 0 {
            break;
        }

        let mut blob = Mat::default();
        dnn::blob_from_image_to(
            &frame,
            &mut blob,
            1.0,
            Size::new(300, 300),
            Scalar::new(104.0, 177.0, 123.0, 0.0),
            false,
            false,
            cv_core::CV_32F,
        )?;

        net.set_input(&blob, "", 1.0, Scalar::default())?;

        let mut output_names = cv_core::Vector::<String>::new();
        output_names.push("detection_out");

        let mut detections = Mat::default();
        net.forward(&mut detections, &output_names)?;

        let h = frame.rows();
        let w = frame.cols();

        for i in 0..detections.rows() {
            let confidence = *detections.at_2d::<f32>(i, 2)?;

            if confidence > 0.5 {
                // FIXED: bind row Mat first
                let row_mat = detections.row(i)?;
                let box_row = row_mat.at_row::<f32>(3)?;

                let x1 = (box_row[0] * w as f32) as i32;
                let y1 = (box_row[1] * h as f32) as i32;
                let x2 = (box_row[2] * w as f32) as i32;
                let y2 = (box_row[3] * h as f32) as i32;

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

                let text = format!("{:.1}%", confidence * 100.0);
                imgproc::put_text(
                    &mut frame,
                    &text,
                    Point::new(x1, (y1 - 10).max(0)),
                    imgproc::FONT_HERSHEY_SIMPLEX,
                    0.5,
                    Scalar::new(0.0, 255.0, 0.0, 0.0),
                    1,
                    imgproc::LINE_8,
                    false,
                )?;
            }
        }

        highgui::imshow("DNN Face Detection - Rust", &frame)?;

        if highgui::wait_key(10)? == 'q' as i32 {
            break;
        }
    }

    Ok(())
}
