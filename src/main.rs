fn main() {
    let input = vec![1, 2, 3];
    let mut output: Vec<i32> = Vec::new();

    for value in input {
        output.push(-value);
    }

    println!("{:?}", output);
}
