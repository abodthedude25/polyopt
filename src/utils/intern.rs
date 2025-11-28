//! Symbol interning for efficient identifier storage.

use string_interner::{StringInterner, DefaultSymbol, backend::StringBackend, Symbol as SymbolTrait};
use std::fmt;
use std::sync::RwLock;
use serde::{Serialize, Deserialize};
use once_cell::sync::Lazy;

/// Type alias for our interner backend
type Backend = StringBackend<DefaultSymbol>;

/// A symbol representing an interned string.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Symbol(u32);

impl Symbol {
    pub(crate) fn from_raw(index: u32) -> Self { Symbol(index) }
    pub fn as_raw(&self) -> u32 { self.0 }
}

impl fmt::Debug for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Symbol({})", self.0)
    }
}

/// Global symbol interner (thread-safe).
static GLOBAL_INTERNER: Lazy<RwLock<StringInterner<Backend>>> = 
    Lazy::new(|| RwLock::new(StringInterner::new()));

/// A symbol interner for efficient string storage.
#[derive(Debug)]
pub struct SymbolInterner {
    interner: StringInterner<Backend>,
}

impl Default for SymbolInterner {
    fn default() -> Self { Self::new() }
}

impl SymbolInterner {
    pub fn new() -> Self {
        Self { interner: StringInterner::new() }
    }

    pub fn intern(&mut self, s: &str) -> Symbol {
        let sym = self.interner.get_or_intern(s);
        Symbol(sym.to_usize() as u32)
    }

    pub fn resolve(&self, sym: Symbol) -> Option<&str> {
        let internal_sym = DefaultSymbol::try_from_usize(sym.0 as usize)?;
        self.interner.resolve(internal_sym)
    }

    pub fn get(&self, s: &str) -> Option<Symbol> {
        self.interner.get(s).map(|sym| Symbol(sym.to_usize() as u32))
    }

    pub fn len(&self) -> usize { self.interner.len() }
    pub fn is_empty(&self) -> bool { self.interner.is_empty() }
}

/// Intern a string in the global interner.
pub fn intern(s: &str) -> Symbol {
    let mut interner = GLOBAL_INTERNER.write().unwrap();
    let sym = interner.get_or_intern(s);
    Symbol(sym.to_usize() as u32)
}

/// Resolve a symbol from the global interner.
pub fn resolve(sym: Symbol) -> Option<String> {
    let interner = GLOBAL_INTERNER.read().unwrap();
    let internal_sym = DefaultSymbol::try_from_usize(sym.0 as usize)?;
    interner.resolve(internal_sym).map(|s| s.to_string())
}

/// Well-known symbols that are pre-interned.
pub mod keywords {
    use super::Symbol;
    use once_cell::sync::Lazy;

    pub static FUNC: Lazy<Symbol> = Lazy::new(|| super::intern("func"));
    pub static FOR: Lazy<Symbol> = Lazy::new(|| super::intern("for"));
    pub static IF: Lazy<Symbol> = Lazy::new(|| super::intern("if"));
    pub static ELSE: Lazy<Symbol> = Lazy::new(|| super::intern("else"));
    pub static RETURN: Lazy<Symbol> = Lazy::new(|| super::intern("return"));
    pub static TO: Lazy<Symbol> = Lazy::new(|| super::intern("to"));
    pub static STEP: Lazy<Symbol> = Lazy::new(|| super::intern("step"));
    pub static LET: Lazy<Symbol> = Lazy::new(|| super::intern("let"));
    pub static VAR: Lazy<Symbol> = Lazy::new(|| super::intern("var"));
    pub static CONST: Lazy<Symbol> = Lazy::new(|| super::intern("const"));
    pub static PARALLEL: Lazy<Symbol> = Lazy::new(|| super::intern("parallel"));
    pub static REDUCE: Lazy<Symbol> = Lazy::new(|| super::intern("reduce"));
    pub static MIN: Lazy<Symbol> = Lazy::new(|| super::intern("min"));
    pub static MAX: Lazy<Symbol> = Lazy::new(|| super::intern("max"));
    pub static INT: Lazy<Symbol> = Lazy::new(|| super::intern("int"));
    pub static FLOAT: Lazy<Symbol> = Lazy::new(|| super::intern("float"));
    pub static DOUBLE: Lazy<Symbol> = Lazy::new(|| super::intern("double"));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interner() {
        let mut interner = SymbolInterner::new();
        let sym1 = interner.intern("hello");
        let sym2 = interner.intern("world");
        let sym3 = interner.intern("hello");
        assert_eq!(sym1, sym3);
        assert_ne!(sym1, sym2);
        assert_eq!(interner.resolve(sym1), Some("hello"));
    }

    #[test]
    fn test_global_interner() {
        let sym1 = intern("test_string");
        let sym2 = intern("test_string");
        assert_eq!(sym1, sym2);
        assert_eq!(resolve(sym1), Some("test_string".to_string()));
    }
}
