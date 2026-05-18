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

//! Lexer for the rule DSL.
//!
//! The lexer produces tokens that borrow from the original input string,
//! making it cheap to extract expression spans for delegation to
//! [`Parameter::try_from`](crate::circuit::Parameter).

/// Classification of a token in the rule DSL.
#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    /// Keyword `rule`.
    Rule,
    /// Keyword `match`.
    Match,
    /// Keyword `require`.
    Require,
    /// Keyword `rewrite`.
    Rewrite,
    /// Keyword `mod` (used in `require { a == b mod c }`).
    Mod,
    /// An identifier, e.g. `merge_rz`, `a`, `π`.
    Ident,
    /// A numeric literal, e.g. `0`, `2.5`, `3.14`.
    Number,
    /// Left brace `{`.
    LBrace,
    /// Right brace `}`.
    RBrace,
    /// Left parenthesis `(`.
    LParen,
    /// Right parenthesis `)`.
    RParen,
    /// Left bracket `[`.
    LBracket,
    /// Right bracket `]`.
    RBracket,
    /// Comma `,`.
    Comma,
    /// A line break.
    Newline,
    /// Equality operator `==`.
    EqEq,
    /// Plus operator `+`.
    Plus,
    /// Minus operator `-`.
    Minus,
    /// Star operator `*`.
    Star,
    /// Slash operator `/`.
    Slash,
    /// End of input.
    Eof,
}

/// A single lexical token.
///
/// Tokens are zero-copy: `text` borrows directly from the original input
/// string, so extracting expression spans is allocation-free.
#[derive(Debug, Clone, PartialEq)]
pub struct Token<'a> {
    /// The kind of token.
    pub kind: TokenKind,
    /// The raw text slice from the input string.
    pub text: &'a str,
    /// Byte offset of the start of this token in the input string.
    pub pos: usize,
}

/// Error produced by the lexer when an unexpected character is encountered.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[error("lexer error at byte {pos}: {msg}")]
pub struct LexError {
    /// Human-readable error message.
    pub msg: String,
    /// Byte offset in the input where the error occurred.
    pub pos: usize,
}

/// Hand-written character scanner for the rule DSL.
///
/// # Example
///
/// ```rust
/// use cqlib_core::compiler::knowledge::rule_dsl::lexer::{Lexer, TokenKind};
///
/// let mut lexer = Lexer::new("rule foo { match {} rewrite {} }");
/// assert_eq!(lexer.next_token().unwrap().kind, TokenKind::Rule);
/// assert_eq!(lexer.next_token().unwrap().kind, TokenKind::Ident);
/// ```
pub struct Lexer<'a> {
    /// The full input string being tokenized.
    input: &'a str,
    /// Current byte offset into `input`.
    pos: usize,
}

impl<'a> Lexer<'a> {
    /// Creates a new lexer positioned at the start of `input`.
    pub fn new(input: &'a str) -> Self {
        Self { input, pos: 0 }
    }

    /// Returns the next token from the input.
    ///
    /// Horizontal whitespace and `//` line comments are skipped automatically.
    /// Line breaks are emitted as [`TokenKind::Newline`].
    /// Returns [`TokenKind::Eof`] when the end of the input is reached.
    pub fn next_token(&mut self) -> Result<Token<'a>, LexError> {
        self.skip_horizontal_whitespace_and_comments();

        if self.pos >= self.input.len() {
            return Ok(Token {
                kind: TokenKind::Eof,
                text: "",
                pos: self.pos,
            });
        }

        let start = self.pos;
        let ch = self.peek_char().unwrap();

        match ch {
            '\n' | '\r' => {
                self.advance();
                if ch == '\r' && self.peek_char() == Some('\n') {
                    self.advance();
                }
                // 合并连续换行（包括 \r\n 和 \n 的混合）
                loop {
                    self.skip_horizontal_whitespace_and_comments();
                    match self.peek_char() {
                        Some('\n') => {
                            self.advance();
                        }
                        Some('\r') => {
                            self.advance();
                            if self.peek_char() == Some('\n') {
                                self.advance();
                            }
                        }
                        _ => break,
                    }
                }
                self.make_token(TokenKind::Newline, start)
            }
            '{' => {
                self.advance();
                self.make_token(TokenKind::LBrace, start)
            }
            '}' => {
                self.advance();
                self.make_token(TokenKind::RBrace, start)
            }
            '(' => {
                self.advance();
                self.make_token(TokenKind::LParen, start)
            }
            ')' => {
                self.advance();
                self.make_token(TokenKind::RParen, start)
            }
            '[' => {
                self.advance();
                self.make_token(TokenKind::LBracket, start)
            }
            ']' => {
                self.advance();
                self.make_token(TokenKind::RBracket, start)
            }
            ',' => {
                self.advance();
                self.make_token(TokenKind::Comma, start)
            }
            '+' => {
                self.advance();
                self.make_token(TokenKind::Plus, start)
            }
            '-' => {
                self.advance();
                self.make_token(TokenKind::Minus, start)
            }
            '*' => {
                self.advance();
                self.make_token(TokenKind::Star, start)
            }
            '/' => {
                self.advance();
                if self.peek_char() == Some('/') {
                    self.skip_line_comment();
                    self.next_token()
                } else {
                    self.make_token(TokenKind::Slash, start)
                }
            }
            '=' => {
                self.advance();
                if self.peek_char() == Some('=') {
                    self.advance();
                    self.make_token(TokenKind::EqEq, start)
                } else {
                    Err(LexError {
                        msg: format!("unexpected character '{}'", ch),
                        pos: start,
                    })
                }
            }
            _ if ch.is_alphabetic() || ch == '_' => self.read_ident(start),
            _ if ch.is_ascii_digit() || ch == '.' => self.read_number(start),
            _ => Err(LexError {
                msg: format!("unexpected character '{}'", ch),
                pos: start,
            }),
        }
    }

    /// Reads an identifier or keyword starting at `start`.
    fn read_ident(&mut self, start: usize) -> Result<Token<'a>, LexError> {
        while let Some(ch) = self.peek_char() {
            if ch.is_alphanumeric() || ch == '_' {
                self.advance();
            } else {
                break;
            }
        }
        let text = &self.input[start..self.pos];
        let kind = match text {
            "rule" => TokenKind::Rule,
            "match" => TokenKind::Match,
            "require" => TokenKind::Require,
            "rewrite" => TokenKind::Rewrite,
            "mod" => TokenKind::Mod,
            _ => TokenKind::Ident,
        };
        Ok(Token {
            kind,
            text,
            pos: start,
        })
    }

    /// Reads a numeric literal starting at `start`.
    fn read_number(&mut self, start: usize) -> Result<Token<'a>, LexError> {
        while let Some(ch) = self.peek_char() {
            if ch.is_ascii_digit() || ch == '.' {
                self.advance();
            } else {
                break;
            }
        }
        let text = &self.input[start..self.pos];
        Ok(Token {
            kind: TokenKind::Number,
            text,
            pos: start,
        })
    }

    /// Skips whitespace and line comments until the next meaningful character.
    fn skip_horizontal_whitespace_and_comments(&mut self) {
        loop {
            match self.peek_char() {
                Some(ch) if ch.is_whitespace() && ch != '\n' && ch != '\r' => self.advance(),
                Some('/') if self.peek_next() == Some('/') => self.skip_line_comment(),
                _ => break,
            }
        }
    }

    /// Skips from `//` to the end of the current line.
    fn skip_line_comment(&mut self) {
        while let Some(ch) = self.peek_char() {
            if ch == '\n' {
                break;
            }
            self.advance();
        }
    }

    /// Builds a [`Token`] of the given kind spanning from `start` to the
    /// current cursor position.
    fn make_token(&self, kind: TokenKind, start: usize) -> Result<Token<'a>, LexError> {
        Ok(Token {
            kind,
            text: &self.input[start..self.pos],
            pos: start,
        })
    }

    /// Returns the next character without consuming it.
    fn peek_char(&self) -> Option<char> {
        self.input[self.pos..].chars().next()
    }

    /// Returns the character after the next one without consuming anything.
    fn peek_next(&self) -> Option<char> {
        self.input[self.pos..].chars().nth(1)
    }

    /// Consumes one UTF-8 character and advances the cursor.
    fn advance(&mut self) {
        if let Some(ch) = self.peek_char() {
            self.pos += ch.len_utf8();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lex_all(input: &str) -> Vec<TokenKind> {
        let mut lexer = Lexer::new(input);
        let mut kinds = Vec::new();
        loop {
            let token = lexer.next_token().unwrap();
            kinds.push(token.kind.clone());
            if token.kind == TokenKind::Eof {
                break;
            }
        }
        kinds
    }

    #[test]
    fn lex_keywords_and_ident() {
        let kinds = lex_all("rule match require rewrite mod foo");
        assert_eq!(
            kinds,
            vec![
                TokenKind::Rule,
                TokenKind::Match,
                TokenKind::Require,
                TokenKind::Rewrite,
                TokenKind::Mod,
                TokenKind::Ident,
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn lex_punctuation() {
        let kinds = lex_all("{ } ( ) [ ] , \n == + - * /");
        assert_eq!(
            kinds,
            vec![
                TokenKind::LBrace,
                TokenKind::RBrace,
                TokenKind::LParen,
                TokenKind::RParen,
                TokenKind::LBracket,
                TokenKind::RBracket,
                TokenKind::Comma,
                TokenKind::Newline,
                TokenKind::EqEq,
                TokenKind::Plus,
                TokenKind::Minus,
                TokenKind::Star,
                TokenKind::Slash,
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn lex_numbers() {
        let kinds = lex_all("0 1 2.5 3.14");
        assert_eq!(
            kinds,
            vec![
                TokenKind::Number,
                TokenKind::Number,
                TokenKind::Number,
                TokenKind::Number,
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn lex_comment() {
        let kinds = lex_all("rule // this is a comment\nmatch");
        assert_eq!(
            kinds,
            vec![
                TokenKind::Rule,
                TokenKind::Newline,
                TokenKind::Match,
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn lex_consecutive_newlines() {
        let kinds = lex_all("rule\n\n\r\nmatch");
        assert_eq!(
            kinds,
            vec![
                TokenKind::Rule,
                TokenKind::Newline,
                TokenKind::Match,
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn lex_unicode_identifier_accepted() {
        // Greek and CJK characters are alphabetic per Unicode → accepted as Ident
        let kinds = lex_all("rule α");
        assert_eq!(
            kinds,
            vec![
                TokenKind::Rule,
                TokenKind::Ident, // α (U+03B1)
                TokenKind::Eof
            ]
        );
    }
}
