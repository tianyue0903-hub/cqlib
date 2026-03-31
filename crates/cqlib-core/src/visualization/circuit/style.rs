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

use serde::Deserialize;
use std::collections::HashMap;

/// Per-gate visual style settings (compatible with Python style JSON schema).
#[derive(Debug, Clone, Deserialize, Default)]
pub struct GateStyle {
    #[serde(default)]
    pub border_color: Option<String>,
    #[serde(default)]
    pub background_color: Option<String>,
    #[serde(default)]
    pub font_size: Option<f64>,
    #[serde(default)]
    pub text_color: Option<String>,
    #[serde(default)]
    pub line_color: Option<String>,
    #[serde(default)]
    pub line_width: Option<f64>,
}

/// Style dictionary keyed by gate name (with mandatory `default` fallback).
#[derive(Debug, Clone)]
pub struct StyleBook {
    styles: HashMap<String, GateStyle>,
    default_style: GateStyle,
}

impl StyleBook {
    /// Load style map by name and apply optional runtime overrides.
    pub fn new(style_name: &str, overrides: &HashMap<String, GateStyle>) -> Self {
        let mut styles = load_style(style_name);
        for (k, v) in overrides {
            styles.insert(k.clone(), v.clone());
        }
        if !styles.contains_key("default") {
            styles.insert("default".to_string(), GateStyle::default());
        }
        let default_style = styles.get("default").cloned().unwrap_or_default();
        Self {
            styles,
            default_style,
        }
    }

    /// Return style for gate name, falling back to `default`.
    pub fn get(&self, gate_name: &str) -> &GateStyle {
        self.styles.get(gate_name).unwrap_or(&self.default_style)
    }
}

/// Load built-in style JSON by name.
///
/// Supported names:
/// - `default`
/// - `gray`
///
/// Unknown names fall back to `default`.
fn load_style(style_name: &str) -> HashMap<String, GateStyle> {
    let style_key = style_name.trim().to_ascii_lowercase();
    let json = match style_key.as_str() {
        "gray" => include_str!("styles/gray.json"),
        _ => include_str!("styles/default.json"),
    };
    serde_json::from_str(json).unwrap_or_else(|_| {
        let mut fallback = HashMap::new();
        fallback.insert("default".to_string(), GateStyle::default());
        fallback
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_style_book_falls_back_to_default_for_unknown_gate() {
        let overrides = HashMap::new();
        let book = StyleBook::new("default", &overrides);
        let style = book.get("UNKNOWN_GATE");
        assert_eq!(style.text_color.as_deref(), Some("black"));
    }

    #[test]
    fn test_style_book_loads_gray_style() {
        let overrides = HashMap::new();
        let book = StyleBook::new("gray", &overrides);
        let style = book.get("default");
        assert_eq!(style.background_color.as_deref(), Some("white"));
    }
}