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

//! Internal helpers for text-circuit rendering.

use super::*;

pub(super) fn y_wire(lane: usize) -> usize {
    lane * 2 + 1
}

pub(super) fn total_rows(num_qubits: usize) -> usize {
    num_qubits * 2 + 1
}

pub(super) fn effective_line_width(options: &TextDrawerOptions) -> usize {
    let w = options.line_width;
    if w > MIN_LINE_WIDTH {
        return w as usize;
    }
    if w < 0 {
        return usize::MAX;
    }
    DEFAULT_LINE_WIDTH as usize
}

pub(super) fn make_lines(
    visual: &VisualCircuit,
    show_params: bool,
    initial_state: bool,
) -> Vec<Vec<Vec<String>>> {
    let mut lines = Vec::new();
    let lines_count = total_rows(visual.num_qubits());
    let qubit_len = visual
        .qubits
        .iter()
        .map(|q| q.id().to_string().len())
        .max()
        .unwrap_or(1);

    let initial_prefix = if initial_state { "|0>" } else { "" };
    let empty_line = " ".repeat(qubit_len + if initial_state { 9 } else { 6 });
    let mut start_lines: Vec<Vec<String>> = Vec::with_capacity(lines_count);
    for qubit in &visual.qubits {
        start_lines.push(vec![empty_line.clone()]);
        let q_label = format!("Q{}", qubit.id());
        let wire_head = BoxChar::LEFT_RIGHT.repeat(2);
        start_lines.push(vec![format!(
            " {:>width$}: {}{}",
            q_label,
            initial_prefix,
            wire_head,
            width = qubit_len + 1
        )]);
    }
    start_lines.push(vec![empty_line]);
    lines.push(start_lines);

    for moment in generate_moments(visual) {
        let mut moment_lines: Vec<Vec<String>> = (0..lines_count)
            .map(|i| {
                if i % 2 == 1 {
                    vec![BoxChar::LEFT_RIGHT.to_string()]
                } else {
                    vec![" ".to_string()]
                }
            })
            .collect();

        let columns = moment_to_columns(visual, &moment);
        for column in columns.clone() {
            let column_lines = draw_column(visual, &column, show_params);
            for (line_i, line) in column_lines.into_iter().enumerate() {
                moment_lines[line_i].extend(line);
            }
        }

        for (i, row) in moment_lines.iter_mut().enumerate() {
            row.push(if i % 2 == 1 {
                BoxChar::LEFT_RIGHT.to_string()
            } else {
                " ".to_string()
            });
        }

        let col_len = columns.len();
        if col_len > 1 {
            let s =
                BoxChar::LEFT_RIGHT.repeat(str_len(&moment_lines[0].concat()).saturating_sub(2));
            moment_lines[0] = vec![
                BoxChar::BOTTOM_RIGHT.to_string(),
                s.clone(),
                BoxChar::BOTTOM_LEFT.to_string(),
            ];
            moment_lines[lines_count - 1] = vec![
                BoxChar::TOP_RIGHT.to_string(),
                s,
                BoxChar::TOP_LEFT.to_string(),
            ];
        }
        lines.push(moment_lines);
    }

    lines
}

pub(super) fn generate_moments(visual: &VisualCircuit) -> Vec<Vec<usize>> {
    let mut layers = vec![Vec::new(); visual.num_columns.max(1)];
    for (op_idx, op) in visual.operations.iter().enumerate() {
        if op.column >= layers.len() {
            layers.resize(op.column + 1, Vec::new());
        }
        layers[op.column].push(op_idx);
    }
    layers
}

pub(super) fn moment_to_columns(visual: &VisualCircuit, moment: &[usize]) -> Vec<Vec<usize>> {
    let mut ranges: Vec<(usize, usize, usize)> = Vec::new();
    for op_idx in moment {
        let op = &visual.operations[*op_idx];
        if op.lanes.is_empty() {
            continue;
        }
        let mut ys = op.lanes.iter().map(|lane| y_wire(*lane));
        if let Some(first) = ys.next() {
            let mut min_y = first;
            let mut max_y = first;
            for y in ys {
                min_y = min_y.min(y);
                max_y = max_y.max(y);
            }
            ranges.push((min_y, max_y, *op_idx));
        }
    }

    let mut columns: Vec<Vec<(usize, usize, usize)>> = Vec::new();
    for range in ranges {
        let (min_y, max_y, _) = range;
        let mut placed = false;
        for column in &mut columns {
            let overlaps = column
                .iter()
                .any(|(cmin, cmax, _)| max_y >= *cmin && min_y <= *cmax);
            if !overlaps {
                column.push(range);
                placed = true;
                break;
            }
        }
        if !placed {
            columns.push(vec![range]);
        }
    }

    columns
        .into_iter()
        .map(|col| col.into_iter().map(|(_, _, op_idx)| op_idx).collect())
        .collect()
}

pub(super) fn draw_column(
    visual: &VisualCircuit,
    column: &[usize],
    show_params: bool,
) -> Vec<Vec<String>> {
    let lines_count = total_rows(visual.num_qubits());
    let mut lines: Vec<Vec<String>> = vec![Vec::new(); lines_count];

    for op_idx in column {
        let op = &visual.operations[*op_idx];
        draw_operation(visual, op, &mut lines, show_params);
    }

    let mut max_width = 1usize;
    for line in &lines {
        for s in line {
            max_width = max_width.max(str_len(s));
        }
    }

    let empty_line = " ".repeat(max_width);
    let wire_line = BoxChar::LEFT_RIGHT.repeat(max_width);
    for (i, line) in lines.iter_mut().enumerate() {
        if i % 2 == 0 {
            if let Some(token) = line.first() {
                *line = vec![center_fill(token, max_width, ' ')];
            } else {
                *line = vec![empty_line.clone()];
            }
        } else if let Some(token) = line.first() {
            *line = vec![center_fill(
                token,
                max_width,
                BoxChar::LEFT_RIGHT.chars().next().unwrap_or('─'),
            )];
        } else {
            *line = vec![wire_line.clone()];
        }
    }

    lines
}

pub(super) fn draw_operation(
    visual: &VisualCircuit,
    op: &VisualOperation,
    lines: &mut [Vec<String>],
    show_params: bool,
) {
    let label = compose_label(op, show_params);
    match op.style {
        VisualOpStyle::Gate => {
            if op.lanes.len() <= 1 {
                if let Some(lane) = op.lanes.first().copied() {
                    let y = y_wire(lane);
                    lines[y].push(label);
                }
                return;
            }

            if op.label == "FSIM" {
                draw_fsim(lines, op);
                return;
            }

            let mark_targets =
                op.label == "UNITARY" || matches!(op.label.as_str(), "RXX" | "RYY" | "RZX" | "RZZ");
            if op.span_box || mark_targets {
                draw_span_box(lines, &op.lanes, &label, mark_targets);
                return;
            }

            for lane in &op.lanes {
                lines[y_wire(*lane)].push(label.clone());
            }
            draw_connect_vertical(lines, &op.covered_lanes);
        }
        VisualOpStyle::Controlled { num_controls } => {
            for lane in op.lanes.iter().take(num_controls) {
                lines[y_wire(*lane)].push(BoxChar::DOT.to_string());
            }
            for lane in op.lanes.iter().skip(num_controls) {
                lines[y_wire(*lane)].push(label.clone());
            }
            draw_connect_vertical(lines, &op.covered_lanes);
        }
        VisualOpStyle::Cz => {
            for lane in &op.lanes {
                lines[y_wire(*lane)].push(BoxChar::DOT.to_string());
            }
            draw_connect_vertical(lines, &op.covered_lanes);
        }
        VisualOpStyle::Swap => {
            for lane in &op.lanes {
                lines[y_wire(*lane)].push(BoxChar::CONNECT.to_string());
            }
            draw_connect_vertical(lines, &op.covered_lanes);
        }
        VisualOpStyle::Barrier => {
            let lanes: Vec<usize> = if op.lanes.is_empty() {
                (0..visual.num_qubits()).collect()
            } else {
                op.lanes.clone()
            };
            for lane in lanes {
                let y = y_wire(lane);
                if y > 1 && y - 1 < lines.len() {
                    lines[y - 1].push(BoxChar::TOP_BOTTOM.to_string());
                }
                if y < lines.len() {
                    lines[y].push(BoxChar::TOP_BOTTOM.to_string());
                }
            }
        }
        VisualOpStyle::Measure | VisualOpStyle::Reset | VisualOpStyle::Delay => {
            for lane in &op.lanes {
                lines[y_wire(*lane)].push(label.clone());
            }
        }
        VisualOpStyle::ControlFlow { .. } => {
            if is_control_flow_marker(op) {
                let span_lanes: &[usize] = if op.covered_lanes.is_empty() {
                    &op.lanes
                } else {
                    &op.covered_lanes
                };
                draw_control_flow_marker(lines, span_lanes, &op.label);
            }
        }
    }
}

pub(super) fn draw_control_flow_marker(lines: &mut [Vec<String>], lanes: &[usize], label: &str) {
    if lanes.is_empty() {
        return;
    }
    let q_indices: Vec<usize> = lanes.iter().map(|lane| y_wire(*lane)).collect();
    if q_indices.is_empty() {
        return;
    }
    let min_idx = q_indices.iter().copied().min().unwrap_or(0);
    let max_idx = q_indices.iter().copied().max().unwrap_or(min_idx);
    let top_line = min_idx.saturating_sub(1);
    let bottom_line = (max_idx + 1).min(lines.len().saturating_sub(1));
    if bottom_line <= top_line {
        lines[min_idx].push(label.to_string());
        return;
    }

    let inner_width = (str_len(label) + 2).max(6);
    lines[top_line].push(format!(
        "{}{}{}",
        BoxChar::BOTTOM_RIGHT,
        BoxChar::LEFT_RIGHT.repeat(inner_width),
        BoxChar::BOTTOM_LEFT
    ));
    lines[bottom_line].push(format!(
        "{}{}{}",
        BoxChar::TOP_RIGHT,
        BoxChar::LEFT_RIGHT.repeat(inner_width),
        BoxChar::TOP_LEFT
    ));

    let label_y = select_label_row(top_line, bottom_line);

    #[allow(clippy::needless_range_loop)]
    for y in (top_line + 1)..bottom_line {
        let is_wire = y % 2 == 1;
        let fill = ' ';
        let mut body = fill.to_string().repeat(inner_width);
        if y == label_y {
            body = overlay_text_center(&body, label);
        }
        let left_side = if is_wire {
            BoxChar::TOP_BOTTOM_LEFT
        } else {
            BoxChar::TOP_BOTTOM
        };
        let right_side = if is_wire {
            BoxChar::TOP_BOTTOM_RIGHT
        } else {
            BoxChar::TOP_BOTTOM
        };
        lines[y].push(format!("{left_side}{body}{right_side}"));
    }
}

pub(super) fn is_control_flow_marker(op: &VisualOperation) -> bool {
    matches!(op.style, VisualOpStyle::ControlFlow { .. })
        && op.children.is_none()
        && matches!(
            op.style,
            VisualOpStyle::ControlFlow {
                kind: VisualControlFlowKind::IfStart
                    | VisualControlFlowKind::ElseStart
                    | VisualControlFlowKind::WhileStart
                    | VisualControlFlowKind::End
            }
        )
}

pub(super) fn draw_connect_vertical(lines: &mut [Vec<String>], lanes: &[usize]) {
    if lanes.is_empty() {
        return;
    }
    let mut ys = lanes.iter().map(|lane| y_wire(*lane));
    let Some(first) = ys.next() else {
        return;
    };
    let mut min_y = first;
    let mut max_y = first;
    for y in ys {
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }

    #[allow(clippy::needless_range_loop)]
    for y in (min_y + 1)..max_y {
        if y % 2 == 1 {
            lines[y].push(BoxChar::TOP_BOTTOM_LEFT_RIGHT.to_string());
        } else {
            lines[y].push(BoxChar::TOP_BOTTOM.to_string());
        }
    }
}

pub(super) fn draw_fsim(lines: &mut [Vec<String>], op: &VisualOperation) {
    let span_lanes: &[usize] = if op.covered_lanes.is_empty() {
        &op.lanes
    } else {
        &op.covered_lanes
    };
    if span_lanes.is_empty() {
        return;
    }

    let mut ys = span_lanes.iter().map(|lane| y_wire(*lane));
    let Some(first) = ys.next() else {
        return;
    };
    let mut min_y = first;
    let mut max_y = first;
    for y in ys {
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }

    let block_width = 7usize;
    let mid = block_width / 2;

    // End qubit lines show gate name only.
    let end_token = center_fill("FSIM", block_width, '─');
    for lane in &op.lanes {
        lines[y_wire(*lane)].push(end_token.clone());
    }

    #[allow(clippy::needless_range_loop)]
    for y in (min_y + 1)..max_y {
        let base_char = if y % 2 == 1 {
            BoxChar::TOP_BOTTOM_LEFT_RIGHT.chars().next().unwrap_or('┼')
        } else {
            BoxChar::TOP_BOTTOM.chars().next().unwrap_or('│')
        };

        let mut token_chars: Vec<char> = if y % 2 == 1 {
            vec!['─'; block_width]
        } else {
            vec![' '; block_width]
        };
        token_chars[mid] = base_char;
        lines[y].push(token_chars.into_iter().collect());
    }
}

pub(super) fn draw_span_box(
    lines: &mut [Vec<String>],
    lanes: &[usize],
    label: &str,
    mark_targets: bool,
) {
    let q_indices: Vec<usize> = lanes.iter().map(|lane| y_wire(*lane)).collect();
    if q_indices.is_empty() {
        return;
    }
    let min_idx = q_indices.iter().copied().min().unwrap_or(0);
    let max_idx = q_indices.iter().copied().max().unwrap_or(min_idx);
    let width = (str_len(label) + 4).max(7);

    let label_bar = center_fill(label, width - 2, ' ');
    let top_border = format!(
        "{}{}{}",
        BoxChar::BOTTOM_RIGHT,
        BoxChar::LEFT_RIGHT.repeat(width - 2),
        BoxChar::BOTTOM_LEFT
    );
    let bottom_border = format!(
        "{}{}{}",
        BoxChar::TOP_RIGHT,
        BoxChar::LEFT_RIGHT.repeat(width - 2),
        BoxChar::TOP_LEFT
    );
    let mid_blank = format!(
        "{}{}{}",
        BoxChar::TOP_BOTTOM,
        " ".repeat(width - 2),
        BoxChar::TOP_BOTTOM
    );
    let mid_label = format!(
        "{}{}{}",
        BoxChar::TOP_BOTTOM,
        label_bar,
        BoxChar::TOP_BOTTOM
    );

    let top_line = min_idx.saturating_sub(1);
    let bottom_line = (max_idx + 1).min(lines.len().saturating_sub(1));
    let center_line = (top_line + bottom_line) / 2;

    lines[top_line].push(top_border);
    lines[bottom_line].push(bottom_border);

    let mut markers: Vec<(usize, String)> = Vec::new();
    if mark_targets {
        let mut uniq = Vec::new();
        for q in q_indices {
            if !uniq.contains(&q) {
                uniq.push(q);
            }
        }
        for (idx, qline) in uniq.into_iter().enumerate() {
            markers.push((qline, idx.to_string()));
        }
    }

    for (idx, line) in lines
        .iter_mut()
        .enumerate()
        .take(bottom_line)
        .skip(top_line + 1)
    {
        let mut content = if idx == center_line {
            mid_label.clone()
        } else {
            mid_blank.clone()
        };
        if idx != center_line
            && let Some((_, marker)) = markers.iter().find(|(q, _)| *q == idx)
        {
            content = with_marker(&content, marker);
        }
        line.push(content);
    }
}

pub(super) fn select_label_row(top_line: usize, bottom_line: usize) -> usize {
    let mid = (top_line + bottom_line) / 2;
    if mid % 2 == 0 {
        return mid;
    }
    if mid > top_line {
        return mid - 1;
    }
    if mid + 1 < bottom_line {
        return mid + 1;
    }
    mid
}

pub(super) fn overlay_text_center(base: &str, text: &str) -> String {
    let mut chars: Vec<char> = base.chars().collect();
    if chars.is_empty() {
        return text.to_string();
    }
    let text_chars: Vec<char> = text.chars().collect();
    if text_chars.len() > chars.len() {
        return text.to_string();
    }
    let text_len = text_chars.len();
    let start = (chars.len() - text_len) / 2;
    for (i, ch) in text_chars.into_iter().enumerate() {
        chars[start + i] = ch;
    }
    chars.into_iter().collect()
}

pub(super) fn with_marker(base: &str, marker: &str) -> String {
    let chars: Vec<char> = base.chars().collect();
    if chars.len() < 3 {
        return base.to_string();
    }
    let marker_chars: Vec<char> = marker.chars().collect();
    let mut out = Vec::with_capacity(chars.len());
    out.push(chars[0]);
    for ch in marker_chars {
        out.push(ch);
    }
    let start = marker.len() + 1;
    if start < chars.len() {
        out.extend_from_slice(&chars[start..]);
    }
    out.into_iter().collect()
}

pub(super) fn compose_label(op: &VisualOperation, show_params: bool) -> String {
    if matches!(op.style, VisualOpStyle::Reset) {
        return "|0>".to_string();
    }
    if show_params && !op.params.is_empty() {
        format!("{}({})", op.label, op.params.join(","))
    } else {
        op.label.clone()
    }
}

pub(super) fn str_len(s: &str) -> usize {
    s.chars().count()
}

pub(super) fn center_fill(text: &str, width: usize, fill: char) -> String {
    let len = str_len(text);
    if len >= width {
        return text.to_string();
    }
    let pad = width - len;
    let left = pad / 2;
    let right = pad - left;
    format!(
        "{}{}{}",
        fill.to_string().repeat(left),
        text,
        fill.to_string().repeat(right)
    )
}
