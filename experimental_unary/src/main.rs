use std::io;

fn main() {
    let mut input_text = String::new();

    io::stdin().read_line(&mut input_text);

    let angka_input: i32 = input_text.trim().parse();

    let mut output: Vec<i32> = Vec::new();

    for value in angka_input {
        output.push(-value);
    }

    println!("{:?}", output);
}
