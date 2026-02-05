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

use crate::circuit::parameter::PyParameter;
use cqlib_core::circuit::Parameter;
use cqlib_core::circuit::gate::{Instruction, StandardGate};
use num_complex::Complex64;
use numpy::{PyArray2, ToPyArray};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::RwLock;

#[pyclass(name = "StandardGate", module = "cqlib.circuit.gates")]
#[derive(Debug)]
pub struct PyStandardGate {
    pub inner: StandardGate,
    pub params: Vec<Parameter>,
    hash: RwLock<Option<u64>>,
}

impl Clone for PyStandardGate {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner,
            params: self.params.clone(),
            hash: RwLock::new(None),
        }
    }
}

#[pymethods]
impl PyStandardGate {
    #[new]
    fn new() -> PyResult<Self> {
        // 禁止用户直接实例化 StandardGate()
        Err(PyValueError::new_err(
            "StandardGate cannot be instantiated directly. Use static attributes like StandardGate.H or StandardGate.RX",
        ))
    }

    /// Magic method to make the gate callable for parameter binding.
    /// Example: gate = StandardGate.RX(3.14)
    #[pyo3(signature = (*args))]
    fn __call__(&self, args: &Bound<'_, PyTuple>) -> PyResult<Self> {
        let expected_params = self.inner.num_params();

        // 如果门不需要参数（如 H, X），调用它通常意味着克隆一个新实例
        if expected_params == 0 {
            if !args.is_empty() {
                return Err(PyValueError::new_err(format!(
                    "Gate {} expects 0 parameters, got {}",
                    self.inner,
                    args.len()
                )));
            }
            return Ok(self.clone());
        }

        if args.len() != expected_params {
            return Err(PyValueError::new_err(format!(
                "Gate {} expects {} parameters, got {}",
                self.inner,
                expected_params,
                args.len()
            )));
        }

        let mut new_params = Vec::with_capacity(expected_params);
        for arg in args {
            if let Ok(py_param) = arg.extract::<PyParameter>() {
                new_params.push(py_param.inner);
            } else if let Ok(val) = arg.extract::<f64>() {
                new_params.push(Parameter::from(val));
            } else {
                return Err(PyTypeError::new_err(format!(
                    "Parameter argument must be a float or Parameter, got {:?}",
                    arg
                )));
            }
        }

        Ok(PyStandardGate {
            inner: self.inner,
            params: new_params,
            hash: RwLock::new(None),
        })
    }

    fn __repr__(&self) -> String {
        if self.params.is_empty() {
            format!("{:?}", self.inner)
        } else {
            let params_str: Vec<String> = self.params.iter().map(|p| p.to_string()).collect();
            format!("{:?}({})", self.inner, params_str.join(", "))
        }
    }

    fn __eq__(&self, other: &PyStandardGate) -> bool {
        self.inner == other.inner && self.params == other.params
    }

    fn __hash__(&self) -> u64 {
        match self.hash.read() {
            Ok(guard) if guard.is_some() => return guard.unwrap(),
            _ => {}
        }
        let mut guard = self.hash.write().expect("Hash cache lock poisoned");
        if let Some(hash) = *guard {
            return hash;
        }
        let mut hasher = DefaultHasher::new();
        self.inner.hash(&mut hasher);
        self.params.hash(&mut hasher);
        let hash = hasher.finish();

        *guard = Some(hash);
        hash
    }

    #[getter]
    fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    #[getter]
    fn num_ctrl_qubits(&self) -> usize {
        self.inner.num_ctrl_qubits()
    }

    #[getter]
    fn num_params(&self) -> usize {
        self.inner.num_params()
    }

    #[getter]
    fn params(&self) -> Vec<PyParameter> {
        self.params
            .iter()
            .map(|p| PyParameter { inner: p.clone() })
            .collect()
    }

    // --- Methods ---

    /// Returns the unitary matrix of the gate as a NumPy array.
    #[pyo3(signature = (params=None))]
    fn matrix<'py>(
        &self,
        py: Python<'py>,
        params: Option<Vec<f64>>,
    ) -> PyResult<Bound<'py, PyArray2<Complex64>>> {
        let eval_params: Vec<f64>;

        // 逻辑：优先使用传入的 params，其次尝试计算内部存储的 params
        if let Some(p) = params {
            if p.len() != self.inner.num_params() {
                return Err(PyValueError::new_err(format!(
                    "Gate {:?} expects {} parameters, got {}",
                    self.inner,
                    self.inner.num_params(),
                    p.len()
                )));
            }
            eval_params = p;
        } else if !self.params.is_empty() {
            // 尝试评估内部参数（必须是常数）
            let mut calculated = Vec::with_capacity(self.params.len());
            for p in &self.params {
                match p.evaluate(&None) {
                    Ok(val) => calculated.push(val),
                    Err(_) => {
                        return Err(PyValueError::new_err(
                            "Cannot compute matrix: gate has symbolic parameters.\
                            Please provide concrete values via the 'params' argument.",
                        ));
                    }
                }
            }
            eval_params = calculated;
        } else if self.inner.num_params() == 0 {
            eval_params = vec![];
        } else {
            return Err(PyValueError::new_err(format!(
                "Gate {:?} expects {} parameters, but none were provided.",
                self.inner,
                self.inner.num_params()
            )));
        }

        // StandardGate::matrix 返回 Cow<Array2>
        // 使用 rust-numpy 的 to_pyarray 进行高效转换
        let mat_cow = self.inner.matrix(&eval_params);
        Ok(mat_cow.to_pyarray(py))
    }

    fn control(&self, num_ctrls: usize) -> PyResult<Self> {
        let inst: Instruction = self.inner.into();

        // 尝试生成控制门
        match inst.control(num_ctrls) {
            Some(Instruction::Standard(std_gate)) => {
                // 如果结果仍然是 StandardGate (例如 X.control(1) -> CX)，我们很高兴
                Ok(PyStandardGate {
                    inner: std_gate,
                    params: self.params.clone(),
                    hash: RwLock::new(None),
                })
            }
            Some(_) => {
                // 如果结果变成了 ExtendedGate (例如 H.control(1))，目前我们的简化架构不支持
                Err(PyValueError::new_err(
                    "Controlled version of this gate results in a non-standard gate, which is not supported in this simplified version.",
                ))
            }
            None => Err(PyValueError::new_err(format!(
                "Cannot control gate {:?}",
                self.inner
            ))),
        }
    }

    fn inverse(&self) -> PyResult<Self> {
        // 使用存储的参数，如果没有则使用默认值（通常 0.0 对于确定类型足够）
        let params_to_use = if !self.params.is_empty() {
            self.params.clone()
        } else {
            vec![Parameter::from(0.0); 3]
        };

        match self.inner.inverse(&params_to_use) {
            Some((inv_gate, inv_params)) => Ok(PyStandardGate {
                inner: inv_gate,
                params: inv_params.into_vec(),
                hash: RwLock::new(None),
            }),
            None => Err(PyValueError::new_err(format!(
                "Gate {:?} is not invertible",
                self.inner
            ))),
        }
    }
}

// 注册静态属性 (H, X, RX...)
pub fn register_gates(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let cls = m.getattr("StandardGate")?;
    let py = m.py();

    // 辅助宏或函数来创建实例
    // 由于我们没有公开 new，我们需要想办法构造。
    // 实际上我们可以通过 Py::new 来构造，因为我们在 Rust 侧。

    let add_gate = |name: &str, gate: StandardGate| -> PyResult<()> {
        let instance = Py::new(
            py,
            PyStandardGate {
                inner: gate,
                params: Vec::new(),
                hash: RwLock::new(None),
            },
        )?;
        cls.setattr(name, instance)?;
        Ok(())
    };

    // Single Qubit
    add_gate("I", StandardGate::I)?;
    add_gate("H", StandardGate::H)?;
    add_gate("X", StandardGate::X)?;
    add_gate("Y", StandardGate::Y)?;
    add_gate("Z", StandardGate::Z)?;
    add_gate("S", StandardGate::S)?;
    add_gate("SDG", StandardGate::SDG)?;
    add_gate("T", StandardGate::T)?;
    add_gate("TDG", StandardGate::TDG)?;

    // Parametric
    add_gate("RX", StandardGate::RX)?;
    add_gate("RY", StandardGate::RY)?;
    add_gate("RZ", StandardGate::RZ)?;
    add_gate("U", StandardGate::U)?;
    add_gate("Phase", StandardGate::Phase)?;
    add_gate("GPhase", StandardGate::GPhase)?;

    // Two Qubit
    add_gate("CX", StandardGate::CX)?;
    add_gate("CY", StandardGate::CY)?;
    add_gate("CZ", StandardGate::CZ)?;
    add_gate("CCX", StandardGate::CCX)?; // 3-qubit
    add_gate("SWAP", StandardGate::SWAP)?;

    add_gate("RXX", StandardGate::RXX)?;
    add_gate("RXY", StandardGate::RXY)?;
    add_gate("RYY", StandardGate::RYY)?;
    add_gate("RZX", StandardGate::RZX)?;
    add_gate("RZZ", StandardGate::RZZ)?;

    add_gate("CRX", StandardGate::CRX)?;
    add_gate("CRY", StandardGate::CRY)?;
    add_gate("CRZ", StandardGate::CRZ)?;

    add_gate("XY", StandardGate::XY)?;
    add_gate("X2P", StandardGate::X2P)?;
    add_gate("X2M", StandardGate::X2M)?;
    add_gate("XY2P", StandardGate::XY2P)?;
    add_gate("XY2M", StandardGate::XY2M)?;
    add_gate("Y2P", StandardGate::Y2P)?;
    add_gate("Y2M", StandardGate::Y2M)?;

    add_gate("FSIM", StandardGate::FSIM)?;

    Ok(())
}

impl PyStandardGate {
    pub fn from(gate: StandardGate, params: Vec<Parameter>) -> Self {
        Self {
            inner: gate,
            params,
            hash: RwLock::new(None),
        }
    }
}
