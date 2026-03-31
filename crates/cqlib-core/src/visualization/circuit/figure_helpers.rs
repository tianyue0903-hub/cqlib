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

//! Internal helpers for SVG figure rendering.

use super::*;

pub(super) fn svg_line(
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
    stroke: &str,
    stroke_width: f64,
    dash: Option<&str>,
) -> String {
    let dash_attr = dash
        .map(|d| format!(" stroke-dasharray=\"{}\"", d))
        .unwrap_or_default();
    format!(
        "<line x1=\"{:.3}\" y1=\"{:.3}\" x2=\"{:.3}\" y2=\"{:.3}\" stroke=\"{}\" stroke-width=\"{:.3}\"{} />",
        x1, y1, x2, y2, stroke, stroke_width, dash_attr
    )
}

pub(super) fn svg_rect(
    x: f64,
    y: f64,
    w: f64,
    h: f64,
    fill: &str,
    stroke: &str,
    lw: f64,
    dash: Option<&str>,
) -> String {
    let dash_attr = dash
        .map(|d| format!(" stroke-dasharray=\"{}\"", d))
        .unwrap_or_default();
    format!(
        "<rect x=\"{:.3}\" y=\"{:.3}\" width=\"{:.3}\" height=\"{:.3}\" fill=\"{}\" stroke=\"{}\" stroke-width=\"{:.3}\"{} />",
        x, y, w, h, fill, stroke, lw, dash_attr
    )
}

pub(super) fn svg_circle(
    cx: f64,
    cy: f64,
    r: f64,
    fill: Option<&str>,
    stroke: Option<&str>,
    lw: f64,
) -> String {
    format!(
        "<circle cx=\"{:.3}\" cy=\"{:.3}\" r=\"{:.3}\" fill=\"{}\" stroke=\"{}\" stroke-width=\"{:.3}\" />",
        cx,
        cy,
        r,
        fill.unwrap_or("none"),
        stroke.unwrap_or("none"),
        lw
    )
}

pub(super) fn svg_text(
    x: f64,
    y: f64,
    text: &str,
    font_size: f64,
    color: &str,
    anchor: &str,
) -> String {
    format!(
        "<text x=\"{:.3}\" y=\"{:.3}\" fill=\"{}\" font-size=\"{:.3}\" font-family=\"DejaVu Sans, Arial, sans-serif\" text-anchor=\"{}\" dominant-baseline=\"middle\">{}</text>",
        x,
        y,
        color,
        font_size,
        anchor,
        escape_xml(text)
    )
}

pub(super) fn draw_gate_label_svg(
    x_px: f64,
    y_px: f64,
    label: &str,
    text_color: &str,
    box_w_px: f64,
    box_h_px: f64,
    base_name_fs: f64,
    allow_shrink: bool,
) -> Vec<String> {
    let mut parts = label.splitn(2, '\n');
    let name = parts.next().unwrap_or_default();
    let param = parts.next().filter(|s| !s.is_empty());
    let avail_w = (box_w_px - LABEL_INNER_PADDING_PX).max(LABEL_MIN_INNER_PX);
    let avail_h = (box_h_px - LABEL_INNER_PADDING_PX).max(LABEL_MIN_INNER_PX);
    // Keep a unified global base size (from style default), and only shrink if this gate overflows.
    let mut nfs = base_name_fs.max(1.0);
    let mut pfs = (nfs * PARAM_FONT_SCALE).max(1.0);
    if allow_shrink {
        for _ in 0..LABEL_FIT_MAX_ITERS {
            // Use conservative width factors so measured text is less likely to overflow the box.
            let name_w = name.chars().count() as f64 * nfs * NAME_WIDTH_FACTOR;
            let (need_w, need_h) = if let Some(p) = param {
                let param_w = p.chars().count() as f64 * pfs * PARAM_WIDTH_FACTOR;
                let gap = nfs * LABEL_LINE_GAP_SCALE;
                (name_w.max(param_w), nfs + gap + pfs)
            } else {
                (name_w, nfs)
            };
            if need_w <= avail_w && need_h <= avail_h {
                break;
            }
            let s = (avail_w / need_w)
                .min(avail_h / need_h)
                .min(LABEL_FIT_MAX_STEP)
                .max(LABEL_FIT_MIN_STEP);
            nfs = (nfs * s).max(1.0);
            pfs = (nfs * PARAM_FONT_SCALE).max(1.0);
        }
    }
    let mut out = Vec::new();
    if let Some(p) = param {
        let gap = nfs * LABEL_LINE_GAP_SCALE;
        let total_h = nfs + gap + pfs;
        let top = y_px - total_h / 2.0;
        let name_y = top + nfs / 2.0;
        let param_y = name_y + (nfs / 2.0 + gap + pfs / 2.0);
        out.push(svg_text(x_px, name_y, name, nfs, text_color, "middle"));
        out.push(svg_text(x_px, param_y, p, pfs, text_color, "middle"));
    } else {
        out.push(svg_text(x_px, y_px, name, nfs, text_color, "middle"));
    }
    out
}

pub(super) fn draw_box_svg(
    x: f64,
    y: f64,
    label: &str,
    palette: &FigurePalette,
    style: &GateStyle,
    base_font_size: f64,
    width: f64,
    height: f64,
    sx: f64,
    sy: f64,
    px: &impl Fn(f64) -> f64,
    py: &impl Fn(f64) -> f64,
) -> Vec<String> {
    draw_labeled_rect_svg(
        x,
        y,
        width,
        height,
        label,
        palette,
        style,
        base_font_size,
        sx,
        sy,
        px,
        py,
        None,
        None,
        true,
    )
}

pub(super) fn draw_span_box_svg(
    x: f64,
    y_min: f64,
    height: f64,
    label: &str,
    palette: &FigurePalette,
    style: &GateStyle,
    base_font_size: f64,
    width: f64,
    sx: f64,
    sy: f64,
    px: &impl Fn(f64) -> f64,
    py: &impl Fn(f64) -> f64,
) -> Vec<String> {
    draw_labeled_rect_svg(
        x,
        y_min + height / 2.0,
        width,
        height,
        label,
        palette,
        style,
        base_font_size,
        sx,
        sy,
        px,
        py,
        None,
        None,
        true,
    )
}

pub(super) fn draw_flow_box_svg(
    x: f64,
    y_min: f64,
    height: f64,
    label: &str,
    palette: &FigurePalette,
    style: &GateStyle,
    base_font_size: f64,
    width: f64,
    sx: f64,
    sy: f64,
    px: &impl Fn(f64) -> f64,
    py: &impl Fn(f64) -> f64,
) -> Vec<String> {
    draw_labeled_rect_svg(
        x,
        y_min + height / 2.0,
        width,
        height,
        label,
        palette,
        style,
        base_font_size,
        sx,
        sy,
        px,
        py,
        Some("6,4"),
        Some(palette.gate_edge_color),
        false,
    )
}

pub(super) fn draw_labeled_rect_svg(
    x: f64,
    y: f64,
    width: f64,
    height: f64,
    label: &str,
    palette: &FigurePalette,
    style: &GateStyle,
    base_font_size: f64,
    sx: f64,
    sy: f64,
    px: &impl Fn(f64) -> f64,
    py: &impl Fn(f64) -> f64,
    dash: Option<&str>,
    edge_fallback: Option<&str>,
    allow_shrink: bool,
) -> Vec<String> {
    let fill = normalized_fill_color(style, palette).unwrap_or_else(|| "none".to_string());
    // Border color defaults to the fill color so the gate box outline matches its background.
    let edge = normalized_edge_color(style, palette)
        .or_else(|| edge_fallback.map(str::to_string))
        .unwrap_or_else(|| fill.clone());
    let lw = style_line_width(style, palette);
    let text_color = str_color(style.text_color.as_deref(), palette.text_color);
    let rx = px(x - width / 2.0);
    let ry = py(y - height / 2.0);
    let rw = width * sx;
    let rh = height * sy;
    let mut out = vec![svg_rect(rx, ry, rw, rh, &fill, &edge, lw, dash)];
    out.extend(draw_gate_label_svg(
        px(x),
        py(y),
        label,
        &text_color,
        rw,
        rh,
        base_font_size,
        allow_shrink,
    ));
    out
}

pub(super) fn draw_measure_svg(
    x: f64,
    y: f64,
    palette: &FigurePalette,
    style: &GateStyle,
    base_font_size: f64,
    width: f64,
    height: f64,
    sx: f64,
    sy: f64,
    px: &impl Fn(f64) -> f64,
    py: &impl Fn(f64) -> f64,
) -> Vec<String> {
    let mut out = draw_box_svg(
        x,
        y,
        "",
        palette,
        style,
        base_font_size,
        width,
        height,
        sx,
        sy,
        px,
        py,
    );
    let lw = style_line_width(style, palette);
    let line_color = str_color(style.line_color.as_deref(), palette.text_color);

    // Gauge arc: larger semicircle in the lower half of the box
    let cx = px(x);
    let cy = py(y + 0.18 * height);
    let rx = 0.35 * width * sx;
    let ry = 0.35 * height * sy;
    let arc_lw = (lw * 3.0).max(3.0);
    let arc_sx = cx - rx;
    let arc_ex = cx + rx;
    out.push(format!(
        "<path d=\"M {:.3} {:.3} A {:.3} {:.3} 0 0 1 {:.3} {:.3}\" fill=\"none\" stroke=\"{}\" stroke-width=\"{:.3}\" stroke-linecap=\"round\" />",
        arc_sx, cy, rx, ry, arc_ex, cy, line_color, arc_lw
    ));

    // Small circle at the bottom center (pivot point)
    let dot_r = (lw * 3.0).max(3.0);
    out.push(svg_circle(cx, cy, dot_r, Some(&line_color), None, 0.0));

    // Arrow from pivot toward upper-right (45° direction)
    let arrow_len_x = rx * 0.9;
    let arrow_len_y = ry * 0.9;
    let ax1 = cx + arrow_len_x;
    let ay1 = cy - arrow_len_y;
    out.push(svg_line(cx, cy, ax1, ay1, &line_color, lw, None));

    // Arrowhead
    let head = (lw * 3.5).max(6.0);
    let angle: f64 = std::f64::consts::FRAC_PI_4; // 45°
    let perp = angle + std::f64::consts::FRAC_PI_2;
    let p1x = ax1 - head * angle.cos() + (head * 0.4) * perp.cos();
    let p1y = ay1 + head * angle.sin() - (head * 0.4) * perp.sin();
    let p2x = ax1 - head * angle.cos() - (head * 0.4) * perp.cos();
    let p2y = ay1 + head * angle.sin() + (head * 0.4) * perp.sin();
    out.push(format!(
        "<polygon points=\"{:.3},{:.3} {:.3},{:.3} {:.3},{:.3}\" fill=\"{}\"/>",
        ax1, ay1, p1x, p1y, p2x, p2y, line_color
    ));

    // "0" label at upper-left, "1" label at upper-right
    let label_fs = (base_font_size * 0.85).max(6.0);
    let label_y = py(y - 0.30 * height);
    let label_x0 = cx - rx * 0.75;
    let label_x1 = cx + rx * 0.75;
    out.push(svg_text(label_x0, label_y, "0", label_fs, &line_color, "middle"));
    out.push(svg_text(label_x1, label_y, "1", label_fs, &line_color, "middle"));

    out
}

pub(super) fn escape_xml(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('\"', "&quot;")
}

/// Generate SVG markup from pre-built visual IR.
///
/// This method is useful when you want to cache or transform visualization IR once and render
pub(super) fn compose_label(label: &str, params: &[String], show_params: bool) -> String {
    if show_params && !params.is_empty() {
        format!("{label}\n{}", params.join(","))
    } else {
        label.to_string()
    }
}

pub(super) fn show_span_lane_markers(
    op: &crate::visualization::circuit::model::VisualOperation,
) -> bool {
    matches!(op.label.as_str(), "RXX" | "RYY" | "RZX" | "RZZ" | "UNITARY")
}

pub(super) fn is_module_span_gate(
    op: &crate::visualization::circuit::model::VisualOperation,
) -> bool {
    matches!(op.style, VisualOpStyle::Gate) && op.span_box
}

pub(super) fn is_control_flow_box(
    op: &crate::visualization::circuit::model::VisualOperation,
) -> bool {
    matches!(op.style, VisualOpStyle::ControlFlow { .. })
}

pub(super) fn module_span_column_width(
    op: &crate::visualization::circuit::model::VisualOperation,
    show_params: bool,
    gate_width: f64,
) -> f64 {
    let label = compose_label(&op.label, &op.params, show_params);
    let text_len = label
        .lines()
        .map(|line| line.chars().count())
        .max()
        .unwrap_or(0) as f64;
    if text_len <= 0.0 {
        return gate_width;
    }
    let padded_len = if text_len > MODULE_LABEL_PADDING_THRESHOLD {
        text_len + MODULE_LABEL_PADDING_CHARS
    } else {
        text_len
    };
    // Reserve extra horizontal space for module/unitary labels to prevent overflow.
    gate_width.max((padded_len / MODULE_LABEL_WIDTH_DIVISOR) * gate_width)
}

pub(super) fn control_flow_column_width(
    op: &crate::visualization::circuit::model::VisualOperation,
    gate_width: f64,
) -> f64 {
    let label_len = op.label.chars().count() as f64;
    if label_len <= 0.0 {
        return gate_width;
    }
    // Keep control-flow labels at base font-size and expand box width instead of shrinking text.
    const CONTROL_FLOW_LABEL_DIVISOR: f64 = 2.9;
    const CONTROL_FLOW_LABEL_PADDING_CHARS: f64 = 0.0;
    gate_width.max(
        ((label_len + CONTROL_FLOW_LABEL_PADDING_CHARS) / CONTROL_FLOW_LABEL_DIVISOR) * gate_width,
    )
}

/// Split columns into folded rows using an order-preserving greedy strategy.
///
/// The algorithm keeps each row as full as possible under the computed width budget.
pub(super) fn split_columns_by_fold(
    col_widths: &[f64],
    fold: i32,
    moment_spacing: f64,
    gate_width: f64,
) -> Vec<Vec<usize>> {
    if col_widths.is_empty() {
        return vec![Vec::new()];
    }
    if fold < 0 {
        return vec![(0..col_widths.len()).collect()];
    }
    let target_cols = usize::try_from(fold)
        .ok()
        .filter(|value| *value > 0)
        .unwrap_or(col_widths.len());

    // Width budget derived from actual average column width (not only gate_width),
    // with a small slack to keep rows as long/compact as possible.
    let avg_col_width =
        col_widths.iter().copied().sum::<f64>() / (col_widths.len() as f64).max(1.0);
    let effective_col_width = avg_col_width.max(gate_width);
    let target_width = (2.0 * moment_spacing
        + target_cols as f64 * effective_col_width
        + target_cols.saturating_sub(1) as f64 * moment_spacing)
        * FOLD_TARGET_SLACK;

    let mut rows = Vec::new();
    let mut start = 0usize;
    while start < col_widths.len() {
        let mut row = vec![start];
        let mut width = 2.0 * moment_spacing + col_widths[start];
        let mut next = start + 1;
        while next < col_widths.len() {
            let candidate = width + moment_spacing + col_widths[next];
            if candidate <= target_width {
                row.push(next);
                width = candidate;
                next += 1;
            } else {
                break;
            }
        }
        rows.push(row);
        start = next;
    }
    rows
}

pub(super) fn lane_to_y(lane: usize, y_base: f64) -> f64 {
    y_base + lane as f64 * WIRE_PITCH
}

pub(super) fn str_color(candidate: Option<&str>, fallback: &str) -> String {
    candidate.unwrap_or(fallback).to_string()
}

pub(super) fn normalize_style_color(candidate: Option<&str>) -> Option<String> {
    match candidate {
        Some(value)
            if value.eq_ignore_ascii_case("none") || value.eq_ignore_ascii_case("transparent") =>
        {
            None
        }
        Some(value) => Some(value.to_string()),
        None => None,
    }
}

pub(super) fn normalized_fill_color(style: &GateStyle, palette: &FigurePalette) -> Option<String> {
    normalize_style_color(style.background_color.as_deref())
        .or_else(|| normalize_style_color(palette.gate_fill_color))
}

pub(super) fn normalized_edge_color(style: &GateStyle, palette: &FigurePalette) -> Option<String> {
    normalize_style_color(style.border_color.as_deref())
        .or_else(|| normalize_style_color(Some(palette.gate_edge_color)))
}

pub(super) fn style_line_width(style: &GateStyle, palette: &FigurePalette) -> f64 {
    style.line_width.unwrap_or(palette.gate_linewidth)
}

/// Resolve style key used to query `StyleBook`.
///
/// Priority:
/// 1. Primitive style categories (`M/R/D/B/CZ/SWAP`);
/// 2. Control-flow families (`IF/ELSE/END/WHILE`);
/// 3. Span-box module category (`MODULE`);
/// 4. Gate label fallback.
pub(super) fn op_style_key(op: &crate::visualization::circuit::model::VisualOperation) -> &str {
    match &op.style {
        VisualOpStyle::Measure => "M",
        VisualOpStyle::Reset => "R",
        VisualOpStyle::Delay => "D",
        VisualOpStyle::Barrier => "B",
        VisualOpStyle::Cz => "CZ",
        VisualOpStyle::Swap => "SWAP",
        VisualOpStyle::ControlFlow { kind } => control_flow_style_key(*kind),
        VisualOpStyle::Gate if op.span_box => "MODULE",
        _ => op.label.as_str(),
    }
}

pub(super) fn control_flow_style_key(kind: VisualControlFlowKind) -> &'static str {
    match kind {
        VisualControlFlowKind::IfElseBlock { .. } | VisualControlFlowKind::IfStart => "IF",
        VisualControlFlowKind::ElseStart => "ELSE",
        VisualControlFlowKind::WhileBlock { .. } | VisualControlFlowKind::WhileStart => "WHILE",
        VisualControlFlowKind::End => "END",
    }
}
