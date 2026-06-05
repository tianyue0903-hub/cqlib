// This code is part of Cqlib.
//
// (C) Copyright China Telecom Quantum Group 2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use super::*;
use crate::circuit::circuit_param::ParameterValue;
use crate::circuit::{
    Circuit, ConditionView, Instruction, Operation, Parameter, Qubit, StandardGate, UnitaryGate,
};
use crate::visualization::circuit::{ParameterDisplayMode, ParameterFormatOptions};
use smallvec::smallvec;
use std::env;
use std::f64::consts::PI;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
struct RgbImage {
    width: u32,
    height: u32,
    data: Vec<u8>,
}

#[derive(Debug)]
struct VisualCasePaths {
    actual_svg: PathBuf,
    actual_png: PathBuf,
    reference_png: PathBuf,
    diff_png: PathBuf,
}

fn q(index: usize) -> Qubit {
    let id = u32::try_from(index).expect("qubit index should fit in u32");
    Qubit::new(id)
}

fn measure_all(circuit: &mut Circuit) {
    for idx in 0..circuit.width() {
        circuit.measure(q(idx)).unwrap();
    }
}

fn visual_threshold() -> f64 {
    env::var("CQLIB_VISUAL_THRESHOLD")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.995)
}

fn ensure_dir(path: &Path) {
    fs::create_dir_all(path).expect("failed to create test directory");
}

fn visual_case_paths(filename: &str) -> VisualCasePaths {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let figure_root = manifest_dir
        .join("src")
        .join("visualization")
        .join("circuit")
        .join("figure");
    let references_dir = figure_root.join("references");
    let diffs_dir = figure_root.join("diffs");
    ensure_dir(&figure_root);
    ensure_dir(&references_dir);
    ensure_dir(&diffs_dir);
    VisualCasePaths {
        actual_svg: figure_root.join(filename.replace(".png", ".svg")),
        actual_png: figure_root.join(filename),
        reference_png: references_dir.join(filename),
        diff_png: diffs_dir.join(format!("diff_{filename}")),
    }
}

fn load_png_rgb(path: &Path) -> RgbImage {
    let pixmap = resvg::tiny_skia::Pixmap::load_png(path)
        .unwrap_or_else(|e| panic!("failed to load png `{}`: {e}", path.display()));
    let width = pixmap.width();
    let height = pixmap.height();
    let src = pixmap.data();
    let mut data = vec![255u8; (width as usize) * (height as usize) * 3];

    for idx in 0..(width as usize * height as usize) {
        let s = idx * 4;
        let d = idx * 3;
        let r = u32::from(src[s]);
        let g = u32::from(src[s + 1]);
        let b = u32::from(src[s + 2]);
        let a = u32::from(src[s + 3]);

        // tiny-skia stores premultiplied rgba, so composite over white here.
        let out_r = (r + ((255 * (255 - a) + 127) / 255)).min(255);
        let out_g = (g + ((255 * (255 - a) + 127) / 255)).min(255);
        let out_b = (b + ((255 * (255 - a) + 127) / 255)).min(255);

        data[d] = out_r as u8;
        data[d + 1] = out_g as u8;
        data[d + 2] = out_b as u8;
    }

    RgbImage {
        width,
        height,
        data,
    }
}

fn pad_rgb_to_canvas(img: &RgbImage, width: u32, height: u32) -> Vec<u8> {
    let mut out = vec![255u8; (width as usize) * (height as usize) * 3];
    for y in 0..img.height {
        let src_offset = (y as usize) * (img.width as usize) * 3;
        let dst_offset = (y as usize) * (width as usize) * 3;
        let row_bytes = (img.width as usize) * 3;
        out[dst_offset..dst_offset + row_bytes]
            .copy_from_slice(&img.data[src_offset..src_offset + row_bytes]);
    }
    out
}

fn similarity_ratio(a: &[u8], b: &[u8]) -> f64 {
    if a.len() != b.len() {
        return 0.0;
    }
    let mse = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = f64::from(*x) - f64::from(*y);
            d * d
        })
        .sum::<f64>()
        / (a.len() as f64);
    if mse <= 1e-12 {
        return 1.0;
    }
    (1.0 - mse / (255.0 * 255.0)).max(0.0)
}

fn save_diff_png(
    a: &[u8],
    b: &[u8],
    width: u32,
    height: u32,
    output_path: &Path,
) -> Result<(), String> {
    let mut pixmap = resvg::tiny_skia::Pixmap::new(width, height)
        .ok_or_else(|| "failed to allocate diff pixmap".to_string())?;
    let dst = pixmap.data_mut();
    const AMP: u16 = 4;

    for idx in 0..(width as usize * height as usize) {
        let i3 = idx * 3;
        let i4 = idx * 4;
        let dr = (i16::from(a[i3]) - i16::from(b[i3])).unsigned_abs() as u16;
        let dg = (i16::from(a[i3 + 1]) - i16::from(b[i3 + 1])).unsigned_abs() as u16;
        let db = (i16::from(a[i3 + 2]) - i16::from(b[i3 + 2])).unsigned_abs() as u16;
        dst[i4] = (dr.saturating_mul(AMP).min(255)) as u8;
        dst[i4 + 1] = (dg.saturating_mul(AMP).min(255)) as u8;
        dst[i4 + 2] = (db.saturating_mul(AMP).min(255)) as u8;
        dst[i4 + 3] = 255;
    }

    pixmap
        .save_png(output_path)
        .map_err(|e| format!("failed to save diff png `{}`: {e}", output_path.display()))
}

fn save_diff_and_similarity(actual_png: &Path, reference_png: &Path, diff_png: &Path) -> f64 {
    let actual = load_png_rgb(actual_png);
    if !reference_png.exists() {
        fs::copy(actual_png, reference_png).unwrap_or_else(|e| {
            panic!(
                "failed to bootstrap reference `{}` from `{}`: {e}",
                reference_png.display(),
                actual_png.display()
            )
        });
        return 1.0;
    }

    let reference = load_png_rgb(reference_png);
    let width = actual.width.max(reference.width);
    let height = actual.height.max(reference.height);
    let actual_padded = pad_rgb_to_canvas(&actual, width, height);
    let reference_padded = pad_rgb_to_canvas(&reference, width, height);
    let ratio = similarity_ratio(&actual_padded, &reference_padded);
    save_diff_png(&actual_padded, &reference_padded, width, height, diff_png)
        .expect("failed to write diff png");
    ratio
}

fn assert_visual_match(circuit: &Circuit, options: FigureDrawerOptions, filename: &str) {
    let paths = visual_case_paths(filename);

    render_figure_to_file(circuit, &paths.actual_svg.to_string_lossy(), &options)
        .expect("failed to render svg");
    render_figure_to_file(circuit, &paths.actual_png.to_string_lossy(), &options)
        .expect("failed to render png");

    let ratio = save_diff_and_similarity(&paths.actual_png, &paths.reference_png, &paths.diff_png);
    let threshold = visual_threshold();
    assert!(
        ratio >= threshold,
        "Similarity ratio {ratio:.4} < {threshold:.4} for {filename}"
    );
}

fn make_bell() -> Circuit {
    let mut circuit = Circuit::new(2);
    circuit.h(q(0)).unwrap();
    circuit.cx(q(0), q(1)).unwrap();
    measure_all(&mut circuit);
    circuit
}

fn make_all_gate() -> Circuit {
    let mut c = Circuit::new(6);
    let q0 = q(0);
    let q1 = q(1);
    let q2 = q(2);
    let q3 = q(3);
    let q4 = q(4);
    let q5 = q(5);

    c.h(q0).unwrap();
    c.h(q1).unwrap();
    c.x(q0).unwrap();
    c.x(q2).unwrap();
    c.y(q1).unwrap();
    c.y(q3).unwrap();
    c.z(q2).unwrap();
    c.z(q4).unwrap();

    c.rx(q0, PI / 3.0).unwrap();
    c.rx(q1, PI / 4.0).unwrap();
    c.ry(q2, PI / 2.0).unwrap();
    c.ry(q3, PI / 5.0).unwrap();
    c.rz(q4, PI / 3.0).unwrap();
    c.rz(q5, PI / 4.0).unwrap();
    c.rxy(q0, PI / 6.0, PI / 3.0).unwrap();
    c.rxx(q1, q2, PI / 7.0).unwrap();
    c.ryy(q0, q3, PI / 6.0).unwrap();
    c.rzx(q3, q1, PI / 7.0).unwrap();
    c.rzz(q1, q2, PI / 6.0).unwrap();

    c.crx(q0, q4, PI / 3.0).unwrap();
    c.crx(q1, q5, PI / 4.0).unwrap();
    c.cry(q2, q3, 0.12 * PI).unwrap();
    c.cry(q0, q2, PI / 5.0).unwrap();
    c.crz(q3, q5, PI / 3.0).unwrap();
    c.crz(q1, q4, PI / 4.0).unwrap();

    c.x2p(q0).unwrap();
    c.x2p(q1).unwrap();
    c.x2m(q2).unwrap();
    c.x2m(q3).unwrap();
    c.y2p(q4).unwrap();
    c.y2p(q5).unwrap();
    c.y2m(q0).unwrap();
    c.y2m(q1).unwrap();
    c.xy(q2, PI / 8.0).unwrap();
    c.xy(q3, PI / 9.0).unwrap();
    c.xy2p(q4, PI / 10.0).unwrap();
    c.xy2m(q5, PI / 11.0).unwrap();

    c.s(q0).unwrap();
    c.s(q1).unwrap();
    c.sdg(q2).unwrap();
    c.sdg(q3).unwrap();
    c.t(q4).unwrap();
    c.t(q5).unwrap();
    c.tdg(q0).unwrap();
    c.tdg(q1).unwrap();

    c.cx(q0, q1).unwrap();
    c.cx(q2, q3).unwrap();
    c.cz(q1, q4).unwrap();
    c.cz(q3, q5).unwrap();
    c.cy(q0, q5).unwrap();
    c.cy(q2, q4).unwrap();
    c.swap(q1, q4).unwrap();

    c.ccx(q0, q1, q2).unwrap();
    c.u(q0, PI / 3.0, PI / 4.0, PI / 5.0).unwrap();
    c.u(q1, PI / 2.0, PI / 3.0, PI / 4.0).unwrap();
    c.u(q2, 0.34, 0.13, 0.56).unwrap();
    measure_all(&mut c);
    c
}

fn make_directive_and_fsim() -> Circuit {
    let mut circuit = Circuit::new(4);
    circuit.h(q(0)).unwrap();
    circuit.fsim(q(1), q(2), 0.21, -0.44).unwrap();
    circuit.barrier(vec![]).unwrap();
    circuit.delay(q(0), ParameterValue::from(40.0)).unwrap();
    circuit.reset(q(3)).unwrap();
    measure_all(&mut circuit);
    circuit
}

fn make_module_unitary_fallback() -> Circuit {
    let mut circuit = Circuit::new(4);

    let mut sub_label = Circuit::new(2);
    sub_label.h(q(0)).unwrap();
    sub_label.cx(q(0), q(1)).unwrap();
    let labeled_gate = sub_label.to_gate("SUB_DEMO_LABEL").unwrap();
    circuit
        .append(
            labeled_gate,
            vec![q(0), q(1)],
            Vec::<ParameterValue>::new(),
            None,
        )
        .unwrap();

    let mut sub_empty = Circuit::new(2);
    sub_empty.x(q(0)).unwrap();
    sub_empty.y(q(1)).unwrap();
    let fallback_gate = sub_empty.to_gate("").unwrap();
    circuit
        .append(
            fallback_gate,
            vec![q(2), q(3)],
            Vec::<ParameterValue>::new(),
            None,
        )
        .unwrap();

    let labeled_unitary = UnitaryGate::new("U_DEMO_LABEL", 2, 0);
    circuit.unitary(labeled_unitary, vec![q(0), q(2)]).unwrap();

    let fallback_unitary = UnitaryGate::new("", 2, 0);
    circuit.unitary(fallback_unitary, vec![q(1), q(3)]).unwrap();
    circuit
}

fn make_module_for_decompose() -> Circuit {
    let mut sub = Circuit::new(2);
    sub.h(q(0)).unwrap();
    sub.cx(q(0), q(1)).unwrap();
    sub.ry(q(1), 0.45).unwrap();
    let sub_gate = sub.to_gate("SUB_BELL").unwrap();

    let mut circuit = Circuit::new(2);
    circuit
        .append(
            sub_gate,
            vec![q(0), q(1)],
            Vec::<ParameterValue>::new(),
            None,
        )
        .unwrap();
    circuit.rz(q(0), -0.32).unwrap();
    circuit
}

fn make_while_control_flow() -> Circuit {
    let mut circuit = Circuit::new(3);
    circuit.measure(q(1)).unwrap();
    let condition = ConditionView::new(q(1), 1);
    let body = vec![
        Operation {
            instruction: Instruction::Standard(StandardGate::H),
            qubits: smallvec![q(0)],
            params: smallvec![],
            label: None,
        },
        Operation {
            instruction: Instruction::Standard(StandardGate::CX),
            qubits: smallvec![q(0), q(2)],
            params: smallvec![],
            label: None,
        },
        Operation {
            instruction: Instruction::Directive(crate::circuit::Directive::Measure),
            qubits: smallvec![q(0)],
            params: smallvec![],
            label: None,
        },
    ];
    circuit.while_loop(condition, body).unwrap();
    circuit.reset(q(2)).unwrap();
    circuit
}

fn make_if_no_else_control_flow() -> Circuit {
    let mut circuit = Circuit::new(3);
    circuit.measure(q(0)).unwrap();
    let condition = ConditionView::new(q(0), 0);
    let true_body = vec![
        Operation {
            instruction: Instruction::Standard(StandardGate::X),
            qubits: smallvec![q(1)],
            params: smallvec![],
            label: None,
        },
        Operation {
            instruction: Instruction::Standard(StandardGate::RZ),
            qubits: smallvec![q(2)],
            params: smallvec![0.32.into()],
            label: None,
        },
    ];
    circuit.if_else(condition, true_body, None).unwrap();
    circuit
}

fn make_mcgate_and_phase() -> Circuit {
    let mut circuit = Circuit::new(5);
    circuit.phase(q(4), 0.22).unwrap();
    circuit
        .multi_control(
            StandardGate::RY,
            [q(0), q(1), q(2)],
            vec![q(3)],
            vec![ParameterValue::from(0.31)],
        )
        .unwrap();
    circuit
}

fn make_fold_stress() -> Circuit {
    let mut circuit = Circuit::new(3);
    for i in 0..14 {
        circuit.h(q(0)).unwrap();
        circuit.cx(q(0), q(1)).unwrap();
        circuit.ry(q(2), (i as f64) * 0.07 - 0.3).unwrap();
    }
    measure_all(&mut circuit);
    circuit
}

#[test]
fn test_bell_default_style() {
    assert_visual_match(
        &make_bell(),
        FigureDrawerOptions::default(),
        "bell_default.png",
    );
}

#[test]
fn test_reverse_bits() {
    assert_visual_match(
        &make_bell(),
        FigureDrawerOptions {
            reverse_bits: true,
            ..FigureDrawerOptions::default()
        },
        "bell_reverse_bits.png",
    );
}

#[test]
fn test_initial_state() {
    assert_visual_match(
        &make_bell(),
        FigureDrawerOptions {
            initial_state: true,
            ..FigureDrawerOptions::default()
        },
        "show_initial_state.png",
    );
}

#[test]
fn test_all_gate() {
    assert_visual_match(
        &make_all_gate(),
        FigureDrawerOptions::default(),
        "all_gate.png",
    );
}

#[test]
fn test_directive_and_fsim() {
    assert_visual_match(
        &make_directive_and_fsim(),
        FigureDrawerOptions::default(),
        "directive_and_fsim.png",
    );
}

#[test]
fn test_module_unitary_label_and_fallback() {
    assert_visual_match(
        &make_module_unitary_fallback(),
        FigureDrawerOptions::default(),
        "module_unitary_label_fallback.png",
    );
}

#[test]
fn test_decompose_circuit_gates() {
    assert_visual_match(
        &make_module_for_decompose(),
        FigureDrawerOptions {
            decompose_circuit_gates: true,
            ..FigureDrawerOptions::default()
        },
        "module_decompose.png",
    );
}

#[test]
fn test_barrier() {
    let mut circuit = Circuit::new(2);
    circuit.h(q(0)).unwrap();
    circuit.cx(q(0), q(1)).unwrap();
    circuit.barrier(vec![q(0)]).unwrap();
    circuit.barrier(vec![q(1)]).unwrap();
    assert_visual_match(&circuit, FigureDrawerOptions::default(), "barrier.png");
}

#[test]
fn test_swap() {
    let mut circuit = Circuit::new(2);
    circuit.x(q(0)).unwrap();
    circuit.cz(q(0), q(1)).unwrap();
    circuit.h(q(1)).unwrap();
    circuit.swap(q(0), q(1)).unwrap();
    assert_visual_match(&circuit, FigureDrawerOptions::default(), "swap.png");
}

#[test]
fn test_long_theta() {
    let mut circuit = Circuit::new(3);
    circuit.h(q(0)).unwrap();
    circuit.rx(q(1), PI).unwrap();
    circuit.rx(q(1), PI / 3.0).unwrap();
    circuit.rx(q(0), 1.0 / 3.0).unwrap();
    circuit.rx(q(1), PI * 13.0 / 3.0).unwrap();
    measure_all(&mut circuit);
    assert_visual_match(&circuit, FigureDrawerOptions::default(), "long_theta.png");
}

#[test]
fn test_moment() {
    let mut circuit = Circuit::new(3);
    circuit.h(q(1)).unwrap();
    circuit.cx(q(0), q(2)).unwrap();
    measure_all(&mut circuit);
    assert_visual_match(&circuit, FigureDrawerOptions::default(), "moment.png");
}

#[test]
fn test_parameter_numeric() {
    let theta = 0.35;
    let mut circuit = Circuit::new(2);
    circuit.h(q(0)).unwrap();
    circuit.rx(q(1), theta).unwrap();
    circuit.cry(q(0), q(1), theta).unwrap();
    measure_all(&mut circuit);
    assert_visual_match(
        &circuit,
        FigureDrawerOptions::default(),
        "parameter_numeric.png",
    );
}

#[test]
fn test_parameter_small_non_zero_uses_scientific_notation() {
    let mut circuit = Circuit::new(1);
    circuit.rx(q(0), 0.0004).unwrap();
    assert_visual_match(
        &circuit,
        FigureDrawerOptions::default(),
        "parameter_small_non_zero.png",
    );
}

#[test]
fn test_parameter_pi_fraction_preferred() {
    let mut circuit = Circuit::new(1);
    circuit.rx(q(0), PI / 2.0).unwrap();

    assert_visual_match(
        &circuit,
        FigureDrawerOptions {
            parameter_format: ParameterFormatOptions {
                mode: ParameterDisplayMode::PiFractionPreferred,
                ..ParameterFormatOptions::default()
            },
            ..FigureDrawerOptions::default()
        },
        "parameter_pi_fraction_preferred.png",
    );
}

#[test]
fn test_parameter_symbolic_with_value_for_symbolic_expr() {
    let mut circuit = Circuit::new(1);
    let theta = Parameter::symbol("theta");
    circuit.rx(q(0), theta + 1.0).unwrap();

    assert_visual_match(
        &circuit,
        FigureDrawerOptions {
            parameter_format: ParameterFormatOptions {
                mode: ParameterDisplayMode::SymbolicWithValue,
                ..ParameterFormatOptions::default()
            },
            ..FigureDrawerOptions::default()
        },
        "parameter_symbolic_with_value.png",
    );
}

#[test]
fn test_two_qubit_rotation() {
    let mut circuit = Circuit::new(4);
    circuit.rxx(q(0), q(1), PI / 3.0).unwrap();
    circuit.ryy(q(1), q(2), PI / 4.0).unwrap();
    circuit.rzz(q(2), q(3), PI / 5.0).unwrap();
    circuit.rzx(q(0), q(3), PI / 6.0).unwrap();
    assert_visual_match(
        &circuit,
        FigureDrawerOptions::default(),
        "two_qubit_rotation.png",
    );
}

#[test]
fn test_unitary() {
    let mut circuit = Circuit::new(4);
    let unitary = UnitaryGate::new("UNITARY", 3, 0);
    circuit.unitary(unitary, vec![q(0), q(1), q(3)]).unwrap();
    assert_visual_match(&circuit, FigureDrawerOptions::default(), "unitary.png");
}

#[test]
fn test_control_flow_expansion() {
    let mut circuit = Circuit::new(2);
    let condition = ConditionView::new(q(0), 1);
    let true_body = vec![Operation {
        instruction: Instruction::Standard(StandardGate::X),
        qubits: smallvec![q(1)],
        params: smallvec![],
        label: None,
    }];
    let false_body = vec![Operation {
        instruction: Instruction::Standard(StandardGate::Z),
        qubits: smallvec![q(1)],
        params: smallvec![],
        label: None,
    }];
    circuit
        .if_else(condition, true_body, Some(false_body))
        .unwrap();

    assert_visual_match(
        &circuit,
        FigureDrawerOptions::default(),
        "if_else_control_flow.png",
    );
}

#[test]
fn test_control_flow_while() {
    assert_visual_match(
        &make_while_control_flow(),
        FigureDrawerOptions::default(),
        "while_control_flow.png",
    );
}

#[test]
fn test_control_flow_if_without_else() {
    assert_visual_match(
        &make_if_no_else_control_flow(),
        FigureDrawerOptions::default(),
        "if_no_else_control_flow.png",
    );
}

#[test]
fn test_default_style_is_applied() {
    let mut circuit = Circuit::new(1);
    circuit.x(q(0)).unwrap();
    assert_visual_match(
        &circuit,
        FigureDrawerOptions::default(),
        "default_style_applied.png",
    );
}

#[test]
fn test_show_params_false() {
    let mut circuit = Circuit::new(2);
    circuit.rx(q(0), 0.66).unwrap();
    circuit.crz(q(0), q(1), -0.44).unwrap();
    circuit.u(q(1), 0.4, -0.2, 0.1).unwrap();
    assert_visual_match(
        &circuit,
        FigureDrawerOptions {
            show_params: false,
            ..FigureDrawerOptions::default()
        },
        "show_params_false.png",
    );
}

#[test]
fn test_multicontrol_and_phase() {
    assert_visual_match(
        &make_mcgate_and_phase(),
        FigureDrawerOptions::default(),
        "multicontrol_and_phase.png",
    );
}

#[test]
fn test_fold_layout() {
    assert_visual_match(
        &make_fold_stress(),
        FigureDrawerOptions {
            fold: 8,
            ..FigureDrawerOptions::default()
        },
        "fold_layout.png",
    );
}
