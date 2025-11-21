use mnist::*;
use ndarray::prelude::*;
use ndarray_npy::read_npy;

fn activation(x: &Array2<f32>) -> Array2<f32> {
    x.map(|v| v.max(0.0))
}

fn softmax(logits: &Array2<f32>) -> Array2<f32> {
    let max = logits.map_axis(Axis(1), |row| row.fold(f32::NEG_INFINITY, |a, &b| a.max(b)));
    let shifted = logits - &max.insert_axis(Axis(1));
    let exp = shifted.map(|v| v.exp());
    let sum = exp.sum_axis(Axis(1));
    &exp / &sum.insert_axis(Axis(1))
}

fn argmax(row: &[f32]) -> usize {
    let mut max_i = 0;
    let mut max_v = f32::MIN;
    for (i, &v) in row.iter().enumerate() {
        if v > max_v {
            max_v = v;
            max_i = i;
        }
    }
    max_i
}

fn main() {
    let l1_loaded: Array2<f32> = read_npy("l1.npy").unwrap();
    let l2_loaded: Array2<f32> = read_npy("l2.npy").unwrap();

    let Mnist {
        tst_img, tst_lbl, ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(60_000)
        .validation_set_length(0)
        .test_set_length(10_000)
        .finalize();

    let test_x = Array2::from_shape_vec((10_000, 784), tst_img)
        .unwrap()
        .map(|v| *v as f32 / 255.0);

    let test_y: Vec<usize> = tst_lbl.iter().map(|v| *v as usize).collect();

    let h = activation(&test_x.dot(&l1_loaded));
    let logits = h.dot(&l2_loaded);
    let probs = softmax(&logits);

    let mut correct = 0;

    for i in 0..10_000 {
        let predicted = argmax(probs.slice(s![i, ..]).as_slice().unwrap());
        if predicted == test_y[i] {
            correct += 1;
        }
    }

    let accuracy = correct as f32 / 10_000.0;

    println!("Accuracy: {:?}%", accuracy * 100.0);
}
