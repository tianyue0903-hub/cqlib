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

#[derive(Debug, Clone, PartialEq)]
pub enum Expression {
    Real(f64),
    Integer(i64),
    Id(String),
    Pi,
    BinaryOp(Box<Expression>, OpCode, Box<Expression>),
    UnaryOp(UnaryOpCode, Box<Expression>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum OpCode {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}

#[derive(Debug, Clone, PartialEq)]
pub enum UnaryOpCode {
    Sin,
    Cos,
    Tan,
    Exp,
    Ln,
    Sqrt,
    Asin,
    Acos,
    Atan,
    Neg,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Argument {
    Id(String),
    IndexedId(String, i64),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    QReg(String, i64),
    CReg(String, i64),
    Include(String),
    Barrier(Vec<Argument>),
    Reset(Argument),
    Measure(Argument, Argument),
    CustomGate(String, Vec<Expression>, Vec<Argument>),
    Opaque(String, Vec<String>, Vec<String>),
    GateDecl(Box<GateDeclData>),
    If(String, i64, Box<Statement>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct GateDeclData {
    pub name: String,
    pub params: Vec<String>,
    pub qubits: Vec<String>,
    pub body: Vec<Statement>,
}

#[derive(Debug, Clone)]
pub struct OpenQASMProgram {
    pub version: f64,
    pub statements: Vec<Statement>,
}
