#[derive(Debug, Clone, PartialEq, Eq, Ord, PartialOrd, Hash)]
pub enum Dim {
    Value(usize),
    // Neg(Box<Dim>),
    Add(Vec<Dim>),
    Inv(Box<Dim>),
    Mul(Vec<Dim>),
    Symbol(String),
}

impl Dim {
    pub fn from_num(n: usize) -> Self {
        Dim::Value(n)
    }

    pub fn from_symbol(symbol: String) -> Self {
        Dim::Symbol(symbol)
    }

    pub fn from_add(dims: Vec<Dim>) -> Self {
        Dim::Add(dims)
    }
}

impl std::ops::Mul for Dim {
    type Output = Dim;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut dim = Dim::Mul(vec![self.clone(), rhs]);
        dim.simplify();
        dim
    }
}

pub trait Mul<Rhs = Self> {
    type Output;

    fn mul(self, rhs: Rhs) -> Self::Output;
}

impl Dim {
    pub fn divide_by(&mut self, expr: Dim) {
        *self = Dim::Mul(vec![self.clone(), Dim::Inv(Box::new(expr))])
    }

    pub fn flatten(&mut self) {
        match self {
            Dim::Mul(muls) => {
                let mut newmuls: Vec<Dim> = vec![];
                let mut value: Option<usize> = None;
                for m in muls {
                    m.flatten();
                    match m {
                        Dim::Mul(mm) => {
                            mm.iter_mut().for_each(|m| match m {
                                Dim::Value(v) => {
                                    if let Some(vv) = &mut value {
                                        *vv *= *v;
                                    } else {
                                        value = Some(*v);
                                    }
                                }
                                m => m.flatten(),
                            });
                            newmuls.extend(mm.clone());
                        }
                        Dim::Value(v) => {
                            if let Some(vv) = &mut value {
                                *vv *= *v;
                            } else {
                                value = Some(*v);
                            }
                        }
                        m => {
                            newmuls.push(m.clone());
                        }
                    }
                }
                if newmuls.is_empty() {
                    if let Some(vv) = value {
                        *self = Dim::Value(vv);
                    } else {
                        unreachable!("At least one operator must stay in flattening")
                    }
                } else {
                    if let Some(vv) = value {
                        newmuls.push(Dim::Value(vv));
                    }
                    newmuls.sort();
                    *self = Dim::Mul(newmuls)
                }
            }
            Dim::Add(adds) => {
                let mut newadds: Vec<Dim> = vec![];
                let mut value: Option<usize> = None;
                for m in adds {
                    m.flatten();
                    match m {
                        Dim::Add(mm) => {
                            mm.iter_mut().for_each(|m| match m {
                                Dim::Value(v) => {
                                    if let Some(vv) = &mut value {
                                        *vv += *v;
                                    } else {
                                        value = Some(*v);
                                    }
                                }
                                m => m.flatten(),
                            });
                            newadds.extend(mm.clone());
                        }
                        Dim::Value(v) => {
                            if let Some(vv) = &mut value {
                                *vv += *v;
                            } else {
                                value = Some(*v);
                            }
                        }
                        m => {
                            newadds.push(m.clone());
                        }
                    }
                }
                if newadds.is_empty() {
                    if let Some(vv) = value {
                        *self = Dim::Value(vv);
                    } else {
                        unreachable!("At least one operator must stay in flattening")
                    }
                } else {
                    if let Some(vv) = value {
                        newadds.push(Dim::Value(vv));
                    }
                    newadds.sort();
                    *self = Dim::Add(newadds)
                }
            }
            _ => (),
        };
    }

    pub fn simplify(&mut self) {
        self.flatten();
        if let Dim::Mul(muls) = self {
            let mut newmuls: Vec<Dim> = vec![];
            let mut skip: Vec<usize> = vec![];
            for (i, m) in muls.iter().enumerate() {
                if skip.contains(&i) {
                    continue;
                }
                if let Dim::Inv(inv_m) = m {
                    let mut do_skip = false;
                    for (j, mm) in muls.iter().enumerate().skip(i + 1) {
                        if mm == &**inv_m {
                            do_skip = true;
                            skip.push(j);
                        }
                    }
                    if !do_skip {
                        newmuls.push(m.clone())
                    }
                } else {
                    newmuls.push(m.clone())
                }
            }
            if newmuls.is_empty() {
                *self = Dim::Value(1);
            } else {
                *self = Dim::Mul(newmuls);
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Kind {
    Float,
    Symbol(String),
    Implicit,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Device {
    Cpu,
    Symbol(String),
    Implicit,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorType {
    pub shape: Vec<Dim>,
    pub kind: Kind,
    pub device: Device,
}

impl TensorType {
    pub fn new(shape: Vec<Dim>, kind: Kind, device: Device) -> Self {
        Self {
            shape,
            kind,
            device,
        }
    }

    pub fn is_arg_compatible(&self, other: &TensorType) -> bool {
        if self.shape.len() != other.shape.len() {
            false
        } else {
            // TODO handle implicit and actual tensor shapes
            true
        }
    }

    pub fn is_view_compatible(&self, other: &TensorType) -> bool {
        let mut fulldim = Dim::Mul(self.shape.clone());
        fulldim.simplify();

        let mut otherdim = Dim::Mul(other.shape.clone());
        otherdim.simplify();

        fulldim == otherdim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flatten_symbol() {
        let d1 = Dim::Symbol("b".to_string());
        let d2 = Dim::Symbol("c".to_string());
        let d3 = Dim::Symbol("a".to_string());

        let mut expr = Dim::Mul(vec![d1.clone(), Dim::Mul(vec![d2.clone(), d3.clone()])]);
        expr.flatten();
        assert_eq!(expr, Dim::Mul(vec![d3, d1, d2]));
    }

    #[test]
    fn flatten_inv() {
        let d1 = Dim::Symbol("b".to_string());
        let d2 = Dim::Inv(Box::new(d1.clone()));

        let mut expr = Dim::Mul(vec![d1.clone(), d2.clone()]);
        expr.flatten();
        assert_eq!(expr, Dim::Mul(vec![d2, d1]));
    }

    #[test]
    fn flatten_value() {
        let d1 = Dim::Value(2);
        let d2 = Dim::Value(3);
        let d3 = Dim::Value(5);

        let mut expr = Dim::Mul(vec![d1.clone(), Dim::Mul(vec![d2.clone(), d3.clone()])]);
        expr.flatten();
        assert_eq!(expr, Dim::Value(30));

        let mut expr = Dim::Add(vec![d1.clone(), Dim::Add(vec![d2.clone(), d3.clone()])]);
        expr.flatten();
        assert_eq!(expr, Dim::Value(10));

        let mut expr = Dim::Add(vec![d1.clone(), Dim::Mul(vec![d2.clone(), d3.clone()])]);
        expr.flatten();
        assert_eq!(expr, Dim::Value(17));

        let mut expr = Dim::Mul(vec![d1.clone(), Dim::Add(vec![d2.clone(), d3.clone()])]);
        expr.flatten();
        assert_eq!(expr, Dim::Value(16));
    }

    #[test]
    fn simplify_inv() {
        let d1 = Dim::Symbol("b".to_string());
        let d2 = Dim::Inv(Box::new(d1.clone()));

        let mut expr = Dim::Mul(vec![d1.clone(), d2.clone()]);
        expr.simplify();
        assert_eq!(expr, Dim::Value(1));
    }
}
