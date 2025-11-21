use mnist::*;
use ndarray::prelude::*;
use ndarray_npy::write_npy;
use rand::Rng;
use std::env;
use std::time::Instant;

fn random_array(r: usize, c: usize) -> Array2<f32> {
    let mut rng = rand::rng();
    Array2::from_shape_fn((r, c), |_| (rng.random::<f32>() - 0.5) * 0.05)
}

fn activation(x: &Array2<f32>) -> Array2<f32> {
    x.map(|v| v.max(0.0))
}

fn activation_derivative(x: &Array2<f32>) -> Array2<f32> {
    x.map(|v| if *v > 0.0 { 1.0 } else { 0.0 })
}

fn softmax(logits: &Array2<f32>) -> Array2<f32> {
    let max = logits.map_axis(Axis(1), |row| row.fold(f32::NEG_INFINITY, |a, &b| a.max(b)));
    let shifted = logits - &max.insert_axis(Axis(1));
    let exp = shifted.map(|v| v.exp());
    let sum = exp.sum_axis(Axis(1));
    &exp / &sum.insert_axis(Axis(1))
}

fn cross_entropy(pred: &Array2<f32>, labels: &[usize]) -> f32 {
    let batch = labels.len();
    let mut loss = 0.0;
    for (i, &label) in labels.iter().enumerate() {
        loss -= pred[[i, label]].ln();
    }
    loss / batch as f32
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
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(60_000)
        .validation_set_length(0)
        .test_set_length(10_000)
        .finalize();

    let train_x = Array2::from_shape_vec((60_000, 784), trn_img)
        .unwrap()
        .map(|v| *v as f32 / 255.0);

    let train_y: Vec<usize> = trn_lbl.iter().map(|v| *v as usize).collect();

    let test_x = Array2::from_shape_vec((10_000, 784), tst_img)
        .unwrap()
        .map(|v| *v as f32 / 255.0);

    let test_y: Vec<usize> = tst_lbl.iter().map(|v| *v as usize).collect();

    let input_size = 784;
    let hidden_size = 128;
    let output_size = 10;

    let lr = 0.01;
    let batch_size = 64;
    let args: Vec<String> = env::args().collect();
    let epochs = &args[1];
    let epochs: i32 = epochs.trim().parse().unwrap();

    let mut l1 = random_array(input_size, hidden_size);
    let mut l2 = random_array(hidden_size, output_size);

    println!("Training...");
    let total_time = Instant::now();

    for epoch in 1..=epochs {
        let mut rng = rand::rng();
        let epoch_time = Instant::now();

        for k in 0..(60_000 / batch_size) {
            let idxs: Vec<usize> = (0..batch_size)
                .map(|_| rng.random_range(0..60_000))
                .collect();

            let x_batch = Array2::from_shape_fn((batch_size, 784), |(i, j)| train_x[(idxs[i], j)]);

            let y_batch: Vec<usize> = idxs.iter().map(|&i| train_y[i]).collect();

            let h_pre = x_batch.dot(&l1);
            let h = activation(&h_pre);
            let logits = h.dot(&l2);
            let probs = softmax(&logits);

            let _loss = cross_entropy(&probs, &y_batch);

            let mut dlogits = probs.clone();
            for (i, &label) in y_batch.iter().enumerate() {
                dlogits[[i, label]] -= 1.0;
            }
            dlogits /= batch_size as f32;

            let dl2 = h.t().dot(&dlogits);

            let dh = dlogits.dot(&l2.t());
            let dh_relu = dh * activation_derivative(&h_pre);
            let dl1 = x_batch.t().dot(&dh_relu);

            l1 -= &(lr * dl1);
            l2 -= &(lr * dl2);
            if k % 100 == 0 {
                println!("Current epoch: {:?}%", 100 * k / (60_000 / batch_size));
            }
        }

        println!("Epoch {epoch} complete in {:#?}", epoch_time.elapsed());
    }
    write_npy("l1.npy", &l1).unwrap();
    write_npy("l2.npy", &l2).unwrap();

    // ---------- Test Accuracy ----------
    let h = activation(&test_x.dot(&l1));
    let logits = h.dot(&l2);
    let probs = softmax(&logits);

    let mut correct = 0;
    for i in 0..10_000 {
        let predicted = argmax(probs.slice(s![i, ..]).as_slice().unwrap());
        if predicted == test_y[i] {
            correct += 1;
        }
    }

    let accuracy = correct as f32 / 10_000.0;

    println!("Test accuracy: {:.2}%", accuracy * 100.0);
    println!("Total time: {:#?}", total_time.elapsed());
}
