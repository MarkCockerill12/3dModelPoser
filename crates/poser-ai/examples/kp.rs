fn main() {
    let p = std::env::args().nth(1).unwrap();
    let img = image::open(&p).unwrap().to_rgb8();
    let t = std::time::Instant::now();
    let k = poser_ai::estimate(&img).unwrap();
    println!("{}x{} in {:.0} ms", img.width(), img.height(), t.elapsed().as_secs_f32() * 1000.0);
    for (i, kp) in k.iter().enumerate() {
        println!("{i:2} {:7.1} {:7.1}  {:.2}", kp.pos.x, kp.pos.y, kp.score);
    }
}
