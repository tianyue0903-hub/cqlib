# Cqlib-core

Cqlib-core 是 Cqlib 量子计算库的核心 Rust 实现，提供量子电路的构建、操作和转换功能。

## 功能特性

- **量子电路构建** - 创建和管理量子电路，支持任意数量的量子比特
- **丰富的量子门** - 36 种标准量子门（H, X, Y, Z, CX, RX, RY, RZ 等）
- **符号参数系统** - 支持参数化量子电路（PQV）和变分算法
  - 表达式解析（数学函数、三角函数等）
  - 符号微分
  - 表达式简化
- **多控门** - 支持多控制门（MCGate）
- **自定义酉门** - 支持用户定义的门矩阵
- **复合门** - 支持将电路作为门使用（CircuitGate）
- **IR 格式支持** - OpenQASM 2.0 和 QCIS 格式的导入/导出
- **电路转矩阵** - 将量子电路转换为矩阵表示

## 模块结构

```
cqlib-core/src/
├── lib.rs                    # 库入口
├── circuit/                  # 量子电路模块
│   ├── circuit_impl.rs       # Circuit 结构体
│   ├── circuit_to_matrix.rs  # 电路→矩阵转换
│   ├── bit.rs               # Qubit 句柄
│   ├── error.rs             # 错误类型
│   ├── operation.rs         # Operation 结构体
│   ├── param.rs             # CircuitParam, ParameterValue
│   ├── gate/                # 量子门
│   │   ├── standard_gate.rs  # StandardGate (36种门)
│   │   ├── instruction.rs    # Instruction 统一指令
│   │   ├── mc_gate.rs       # 多控门
│   │   ├── unitary_gate.rs   # 自定义酉门
│   │   ├── circuit_gate.rs  # 复合门
│   │   ├── gate_matrix.rs   # 门矩阵
│   │   └── ...
│   └── parameter/           # 符号参数系统
│       ├── impls.rs         # Parameter 类型
│       ├── expr_node.rs     # 表达式 AST
│       ├── parse.rs        # 表达式解析器
│       ├── simplify.rs      # 表达式简化
│       └── derivative.rs    # 符号微分
└── ir/                      # 中间表示
    ├── qasm2/               # OpenQASM 2.0
    └── qcis/                # QCIS 格式
```

## 快速开始

```rust
use cqlib_core::circuit::{Circuit, Qubit};
use cqlib_core::circuit::parameter::Parameter;

let mut circuit = Circuit::new(2);

// 应用量子门
circuit.h(Qubit::new(0)).unwrap();
circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();

// 参数化门
let theta = Parameter::try_from("theta").unwrap();
circuit.rx(Qubit::new(0), theta).unwrap();
```

## 构建与测试

```shell
# 构建
cargo build -p cqlib-core

# 测试
cargo test -p cqlib-core

# 发布构建
cargo build --release -p cqlib-core
```

## 依赖

```toml
cqlib-core = { path = "../cqlib-core" }
```

## 公开 API

### 核心类型

| 类型 | 说明 |
|------|------|
| `Circuit` | 量子电路主容器 |
| `Qubit` | 量子比特句柄 |
| `Parameter` | 符号参数 |
| `StandardGate` | 标准量子门枚举 |
| `Instruction` | 统一指令类型 |
