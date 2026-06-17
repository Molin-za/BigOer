# BigOer — 通用算法时间复杂度分析平台

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-teal.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

求解任意递推关系式 $T(n) = \sum_{i=1}^{k} k_i \, T(f_i(n)) + g(n)$ 的渐进复杂度。

## 6+1 策略拦截链

| # | 策略 | 适用场景 |
|---|------|----------|
| 0 | **主定理** | 单项 $T(n/b)$ |
| 1 | **Akra-Bazzi** | 多项 $T(n/b_i)$ |
| 2 | **线性特征方程** | $T(n-c_i)$ 减治型 |
| 3 | **换元构造法** | $\sqrt{n}$, $n^c$ 非线性 |
| 4 | **连续化微积分** | $T(n)-T(n-1)=g(n)$ |
| 5 | **Poincaré-Perron** | 非常系数极限分析 |
| 6 | **数值曲线拟合** | 全能兜底 |

## 快速启动

```bash
pip install -r requirements.txt
python main.py
# 浏览器打开 index.html
```

## 功能亮点

- **符号化简**：无理数指数保留 $\Theta(n^{\log_2 3})$，黄金分割比保留 $\Theta(\varphi^n)$
- **共振检测**：$T(n)=2T(n-1)+2^n \Rightarrow \Theta(n \cdot 2^n)$
- **换元代回**：$T(n)=2T(\sqrt{n})+\log n \Rightarrow \Theta(\log n \cdot \log\log n)$
- **16 位精度**：超越方程 $p$ 值显示 $p \approx 1.1673039782614187$
- **浮点污染防御**：case 分析法杜绝长串小数系数
- **全自定义输入**：系数、变元、驱动函数全部自由输入，无预设下拉菜单

## 技术栈

| 层 | 技术 |
|----|------|
| 后端 | Python · FastAPI · SymPy · SciPy · NumPy |
| 前端 | HTML5 · Tailwind CSS · KaTeX · Font Awesome |
| 数学引擎 | `rsolve` · `nsolve` · `dsolve` · `sp.limit` · `curve_fit` |

## 项目结构

```
BigOer/
├── main.py            # 后端引擎 (850 行)
├── index.html         # 前端 Dashboard
├── requirements.txt   # Python 依赖
├── LICENSE            # MIT
└── README.md
```

## License

MIT © 2026
