use pzeudo::PzeudoCore;

fn main() {
    let mut pzeudo = PzeudoCore::<4, 4096>::default();

    let tensor_a = pzeudo.create_tensor_from_value(10.);
    let tensor_b = pzeudo.create_tensor_from_value(20.);

    let tensor_c = pzeudo.add(&tensor_a, &tensor_b);

    let tensor_d = pzeudo.add(&tensor_c, &tensor_b);

    let tensor_e = pzeudo.add(&tensor_c, &tensor_d);

    pzeudo.execution();
    pzeudo.join();
}
