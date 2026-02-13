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

//! # Parameter Expression Parser
//!
//! This module provides a robust recursive descent parser for mathematical expressions
//! that can be converted into [`Parameter`] objects.
//!
//! Supported syntax:
//! - Numbers: `1`, `3.14`, `-2.5`, `1e-5` (scientific notation)
//! - Constants: `pi`, `e`
//! - Operators: `+`, `-`, `*`, `/`, `%` (mod), `^` (pow)
//! - Functions: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sqrt`, `exp`, `ln`, `log`, `abs`, `sign`
//! - Parentheses: `(`, `)`
//!
//! ## Examples
//!
//! ```rust
//! use cqlib_core::circuit::parameter::parse::parse_parameter;
//!
//! let p1 = parse_parameter("1.0").unwrap();
//! let p2 = parse_parameter("sin(pi/2)").unwrap();
//! let p3 = parse_parameter("x^2 + y^2").unwrap();
//! let p4 = parse_parameter("log(100, 10)").unwrap();
//! ```

use crate::circuit::parameter::expr_node::ExprNode;
use crate::circuit::parameter::impls::Parameter;
use std::error::Error;
use std::fmt;
use std::iter::Peekable;
use std::str::Chars;

/// Errors that can occur during expression parsing.
#[derive(Debug, Clone, PartialEq)]
pub enum ParseError {
    /// Unexpected end of input
    UnexpectedEndOfInput,
    /// Unexpected token encountered
    UnexpectedToken(String),
    /// Invalid number format
    InvalidNumber(String),
    /// Mismatched parentheses
    MismatchedParentheses,
    /// Empty expression
    EmptyExpression,
    /// Unknown function name
    UnknownFunction(String),
    /// Incorrect number of arguments for function
    InvalidArgumentCount {
        func: String,
        expected: String,
        found: usize,
    },
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseError::UnexpectedEndOfInput => write!(f, "Unexpected end of input"),
            ParseError::UnexpectedToken(s) => write!(f, "Unexpected token: {}", s),
            ParseError::InvalidNumber(s) => write!(f, "Invalid number literal: {}", s),
            ParseError::MismatchedParentheses => write!(f, "Mismatched parentheses"),
            ParseError::EmptyExpression => write!(f, "Empty expression"),
            ParseError::UnknownFunction(s) => write!(f, "Unknown function: {}", s),
            ParseError::InvalidArgumentCount {
                func,
                expected,
                found,
            } => write!(
                f,
                "Function '{}' expects {} arguments, found {}",
                func, expected, found
            ),
        }
    }
}

impl Error for ParseError {}

/// Token types for the expression lexer.
///
/// This enum represents all lexical tokens that can appear in a mathematical
/// expression. The lexer converts raw input strings into a sequence of these
/// tokens for the parser to consume.
#[derive(Debug, Clone, PartialEq)]
enum Token {
    Number(f64),
    Identifier(String),
    Plus,     // +
    Minus,    // -
    Multiply, // *
    Divide,   // /
    Modulo,   // %
    Power,    // ^
    LParen,   // (
    RParen,   // )
    Comma,    // ,
    Eof,
}

/// A peekable lexer for mathematical expressions.
///
/// The lexer performs lexical analysis, converting raw input strings into
/// a stream of tokens. It uses a peekable iterator to allow one-character
/// lookahead during tokenization.
struct Lexer<'a> {
    /// Iterator over the input characters with peek capability
    chars: Peekable<Chars<'a>>,
}

impl<'a> Lexer<'a> {
    /// Creates a new lexer for the given input string.
    ///
    /// # Arguments
    ///
    /// * `input` - The mathematical expression to tokenize
    fn new(input: &'a str) -> Self {
        Lexer {
            chars: input.chars().peekable(),
        }
    }

    /// Retrieves the next token from the input.
    ///
    /// This method skips whitespace, then attempts to match the next
    /// character(s) to a valid token type.
    ///
    /// # Returns
    ///
    /// * `Ok(Token)` - The next token in the stream
    /// * `Err(ParseError)` - If an invalid character is encountered
    fn next_token(&mut self) -> Result<Token, ParseError> {
        self.skip_whitespace();

        match self.chars.peek() {
            None => Ok(Token::Eof),
            Some(&c) => match c {
                '+' => {
                    self.chars.next();
                    Ok(Token::Plus)
                }
                '-' => {
                    self.chars.next();
                    Ok(Token::Minus)
                }
                '*' => {
                    self.chars.next();
                    Ok(Token::Multiply)
                }
                '/' => {
                    self.chars.next();
                    Ok(Token::Divide)
                }
                '%' => {
                    self.chars.next();
                    Ok(Token::Modulo)
                }
                '^' => {
                    self.chars.next();
                    Ok(Token::Power)
                }
                '(' => {
                    self.chars.next();
                    Ok(Token::LParen)
                }
                ')' => {
                    self.chars.next();
                    Ok(Token::RParen)
                }
                ',' => {
                    self.chars.next();
                    Ok(Token::Comma)
                }

                // Numbers (including starting with dot like .5)
                '0'..='9' | '.' => self.read_number(),

                // Identifiers
                'a'..='z' | 'A'..='Z' | '_' => self.read_identifier(),

                _ => {
                    let bad_char = self.chars.next().unwrap();
                    Err(ParseError::UnexpectedToken(bad_char.to_string()))
                }
            },
        }
    }

    /// Skips over whitespace characters in the input.
    fn skip_whitespace(&mut self) {
        while let Some(&c) = self.chars.peek() {
            if c.is_whitespace() {
                self.chars.next();
            } else {
                break;
            }
        }
    }

    /// Reads a numeric literal from the input.
    ///
    /// Handles integers, floating-point numbers, and scientific notation
    /// (e.g., `1e-5`, `3.14`, `.5`).
    ///
    /// # Returns
    ///
    /// * `Ok(Token::Number(f64))` - The parsed number
    /// * `Err(ParseError::InvalidNumber)` - If the number format is invalid
    fn read_number(&mut self) -> Result<Token, ParseError> {
        let mut s = String::new();
        let mut has_dot = false;
        let mut has_exponent = false;

        while let Some(&c) = self.chars.peek() {
            if c.is_ascii_digit() {
                s.push(self.chars.next().unwrap());
            } else if c == '.' {
                if has_dot || has_exponent {
                    break;
                }
                has_dot = true;
                s.push(self.chars.next().unwrap());
            } else if c == 'e' || c == 'E' {
                if has_exponent {
                    break;
                }
                has_exponent = true;
                s.push(self.chars.next().unwrap());

                // Consume optional + or - after e
                if let Some(&next_c) = self.chars.peek() {
                    if next_c == '+' || next_c == '-' {
                        s.push(self.chars.next().unwrap());
                    }
                }
            } else {
                break;
            }
        }

        s.parse::<f64>()
            .map(Token::Number)
            .map_err(|_| ParseError::InvalidNumber(s))
    }

    /// Reads an identifier (variable name or function name) from the input.
    ///
    /// Identifiers must start with a letter or underscore, followed by
    /// alphanumeric characters or underscores.
    ///
    /// # Returns
    ///
    /// * `Ok(Token::Identifier(String))` - The parsed identifier
    fn read_identifier(&mut self) -> Result<Token, ParseError> {
        let mut s = String::new();

        while let Some(&c) = self.chars.peek() {
            if c.is_alphabetic() || c == '_' || c.is_ascii_digit() {
                s.push(self.chars.next().unwrap());
            } else {
                break;
            }
        }

        Ok(Token::Identifier(s))
    }
}

/// Recursive descent parser for mathematical expressions.
///
/// Implements a top-down parser using recursive descent with operator
/// precedence climbing. The parser converts tokens into a `Parameter`
/// expression tree.
///
/// # Operator Precedence (Low to High)
///
/// 1. Additive: `+`, `-`
/// 2. Multiplicative: `*`, `/`, `%`
/// 3. Power: `^` (right-associative)
/// 4. Unary: `-`, `+`
/// 5. Primary: Number, Identifier, Function(), `(...)`
struct Parser<'a> {
    /// The lexer providing the token stream
    lexer: Lexer<'a>,
    /// The current lookahead token
    current_token: Token,
}

impl<'a> Parser<'a> {
    /// Creates a new parser for the given input string.
    ///
    /// # Arguments
    ///
    /// * `input` - The mathematical expression to parse
    ///
    /// # Returns
    ///
    /// * `Ok(Parser)` - A ready-to-use parser
    /// * `Err(ParseError)` - If the first token cannot be read
    fn new(input: &'a str) -> Result<Self, ParseError> {
        let mut lexer = Lexer::new(input);
        let current_token = lexer.next_token()?;
        Ok(Parser {
            lexer,
            current_token,
        })
    }

    /// Consumes the current token if it matches the expected type.
    ///
    /// # Arguments
    ///
    /// * `token_discriminant` - The expected token type
    ///
    /// # Returns
    ///
    /// * `Ok(())` - Token was consumed successfully
    /// * `Err(ParseError::UnexpectedToken)` - If token doesn't match
    fn eat(&mut self, token_discriminant: &Token) -> Result<(), ParseError> {
        if std::mem::discriminant(&self.current_token) == std::mem::discriminant(token_discriminant)
        {
            self.current_token = self.lexer.next_token()?;
            Ok(())
        } else {
            Err(ParseError::UnexpectedToken(format!(
                "{:?}",
                self.current_token
            )))
        }
    }

    /// Parses the complete expression and returns a Parameter.
    ///
    /// This is the main entry point for parsing. It ensures the entire
    /// input is consumed (no trailing tokens).
    ///
    /// # Returns
    ///
    /// * `Ok(Parameter)` - The parsed expression tree
    /// * `Err(ParseError)` - If parsing fails or there are trailing tokens
    fn parse(&mut self) -> Result<Parameter, ParseError> {
        if self.current_token == Token::Eof {
            return Err(ParseError::EmptyExpression);
        }
        let result = self.expr()?;
        if self.current_token != Token::Eof {
            return Err(ParseError::UnexpectedToken(format!(
                "{:?}",
                self.current_token
            )));
        }
        Ok(result)
    }

    /// Parses an additive expression.
    ///
    /// Grammar: `expr ::= term (('+' | '-') term)*`
    fn expr(&mut self) -> Result<Parameter, ParseError> {
        let mut left = self.term()?;

        loop {
            match self.current_token {
                Token::Plus => {
                    self.eat(&Token::Plus)?;
                    let right = self.term()?;
                    left = left + right;
                }
                Token::Minus => {
                    self.eat(&Token::Minus)?;
                    let right = self.term()?;
                    left = left - right;
                }
                _ => break,
            }
        }
        Ok(left)
    }

    /// Parses a multiplicative expression.
    ///
    /// Grammar: `term ::= power (('*' | '/' | '%') power)*`
    fn term(&mut self) -> Result<Parameter, ParseError> {
        let mut left = self.power()?;

        loop {
            match self.current_token {
                Token::Multiply => {
                    self.eat(&Token::Multiply)?;
                    let right = self.power()?;
                    left = left * right;
                }
                Token::Divide => {
                    self.eat(&Token::Divide)?;
                    let right = self.power()?;
                    left = left / right;
                }
                Token::Modulo => {
                    self.eat(&Token::Modulo)?;
                    let right = self.power()?;
                    left = Parameter::new(ExprNode::Mod(left.node, right.node));
                }
                _ => break,
            }
        }
        Ok(left)
    }

    /// Parses a power expression (right-associative).
    ///
    /// Grammar: `power ::= unary ('^' power)?`
    ///
    /// Note: Right-associative means `2^3^2` parses as `2^(3^2)`
    fn power(&mut self) -> Result<Parameter, ParseError> {
        let left = self.unary()?;

        if let Token::Power = self.current_token {
            self.eat(&Token::Power)?;
            let right = self.power()?;
            Ok(left.pow(&right))
        } else {
            Ok(left)
        }
    }

    /// Parses a unary expression.
    ///
    /// Grammar: `unary ::= ('+' | '-') unary | primary`
    fn unary(&mut self) -> Result<Parameter, ParseError> {
        match self.current_token {
            Token::Plus => {
                self.eat(&Token::Plus)?;
                self.unary()
            }
            Token::Minus => {
                self.eat(&Token::Minus)?;
                let val = self.unary()?;
                Ok(Parameter::from(0.0) - val)
            }
            _ => self.primary(),
        }
    }

    /// Parses a primary expression (atom).
    ///
    /// Grammar: `primary ::= Number | Identifier | Function | '(' expr ')'`
    fn primary(&mut self) -> Result<Parameter, ParseError> {
        match self.current_token.clone() {
            Token::Number(n) => {
                self.eat(&Token::Number(0.0))?;
                Ok(Parameter::from(n))
            }
            Token::Identifier(name) => {
                // Check if it's a function call lookahead
                // We consume the ID first
                self.eat(&Token::Identifier(String::new()))?;

                if let Token::LParen = self.current_token {
                    self.parse_function_call(&name)
                } else {
                    // Constant or Variable
                    match name.as_str() {
                        "pi" | "PI" => Ok(Parameter::pi()),
                        "e" | "E" => Ok(Parameter::e()),
                        _ => Ok(Parameter::symbol(name)),
                    }
                }
            }
            Token::LParen => {
                self.eat(&Token::LParen)?;
                let expr = self.expr()?;
                if let Token::RParen = self.current_token {
                    self.eat(&Token::RParen)?;
                    Ok(expr)
                } else {
                    Err(ParseError::MismatchedParentheses)
                }
            }
            _ => Err(ParseError::UnexpectedToken(format!(
                "{:?}",
                self.current_token
            ))),
        }
    }

    /// Parses a function call expression.
    ///
    /// # Arguments
    ///
    /// * `func_name` - The name of the function being called
    ///
    /// # Returns
    ///
    /// * `Ok(Parameter)` - The parsed function call as a parameter
    /// * `Err(ParseError)` - If function name is unknown or argument count is wrong
    fn parse_function_call(&mut self, func_name: &str) -> Result<Parameter, ParseError> {
        self.eat(&Token::LParen)?;

        let mut args = Vec::new();
        if self.current_token != Token::RParen {
            args.push(self.expr()?);
            while let Token::Comma = self.current_token {
                self.eat(&Token::Comma)?;
                args.push(self.expr()?);
            }
        }

        self.eat(&Token::RParen)?;

        match func_name {
            "sin" => Self::check_unary(func_name, args, |p| p.sin()),
            "cos" => Self::check_unary(func_name, args, |p| p.cos()),
            "tan" => Self::check_unary(func_name, args, |p| p.tan()),
            "asin" => Self::check_unary(func_name, args, |p| p.asin()),
            "acos" => Self::check_unary(func_name, args, |p| p.acos()),
            "atan" => Self::check_unary(func_name, args, |p| p.atan()),
            "sqrt" => Self::check_unary(func_name, args, |p| p.sqrt()),
            "exp" => Self::check_unary(func_name, args, |p| p.exp()),
            "ln" => Self::check_unary(func_name, args, |p| p.ln()),
            "abs" => Self::check_unary(func_name, args, |p| p.abs()),
            "sign" => {
                Self::check_unary(func_name, args, |p| Parameter::new(ExprNode::Sign(p.node)))
            }
            "log" => match args.len() {
                1 => Ok(args[0].log(None)),
                2 => Ok(args[0].log(Some(args[1].clone()))),
                n => Err(ParseError::InvalidArgumentCount {
                    func: "log".to_string(),
                    expected: "1 or 2".to_string(),
                    found: n,
                }),
            },
            _ => Err(ParseError::UnknownFunction(func_name.to_string())),
        }
    }

    /// Helper to validate and apply a unary function.
    ///
    /// # Arguments
    ///
    /// * `name` - The function name (for error messages)
    /// * `args` - The parsed arguments
    /// * `op` - The operation to apply
    ///
    /// # Returns
    ///
    /// * `Ok(Parameter)` - The result of applying the operation
    /// * `Err(ParseError::InvalidArgumentCount)` - If argument count is not 1
    fn check_unary<F>(name: &str, args: Vec<Parameter>, op: F) -> Result<Parameter, ParseError>
    where
        F: FnOnce(Parameter) -> Parameter,
    {
        if args.len() == 1 {
            Ok(op(args[0].clone()))
        } else {
            Err(ParseError::InvalidArgumentCount {
                func: name.to_string(),
                expected: "1".to_string(),
                found: args.len(),
            })
        }
    }
}

/// Parse a mathematical expression string into a [`Parameter`].
///
/// # Arguments
///
/// * `expr` - The expression string to parse
///
/// # Returns
///
/// * `Ok(Parameter)` - The parsed parameter
/// * `Err(ParseError)` - If the expression is invalid
pub fn parse_parameter(expr: &str) -> Result<Parameter, ParseError> {
    let mut parser = Parser::new(expr)?;
    parser.parse()
}

#[cfg(test)]
#[path = "./parse_test.rs"]
mod parse_test;
