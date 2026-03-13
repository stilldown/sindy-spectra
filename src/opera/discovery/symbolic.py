from __future__ import annotations

from typing import List
import numpy as np

def build_latex_blocks_from_xi(
    xi: np.ndarray,
    operator_names: List[str],
    top_terms: int = 8,
) -> List[str]:
    """直接从隐式稀疏系数 Xi 构建 LaTeX 方程（尝试解析得到关于 f_k, g_k 的偏微分方程）。"""
    x = np.asarray(xi)
    k_eff = x.shape[1]
    blocks: List[str] = []

    try:
        import sympy as sp
        import re

        omega_sym = sp.Symbol('omega', real=True)

        # Parse number of c variables from operator names or assume up to 10
        max_c = 0
        for name in operator_names:
            matches = re.findall(r'c_(\d+)', name)
            for m in matches:
                max_c = max(max_c, int(m))
        
        c_syms = sp.symbols(f"c_1:{max_c+1}", real=True) if max_c > 0 else (sp.Symbol('c_1', real=True),)

        for k in range(k_eff):
            col = x[:, k]
            idx = np.argsort(np.abs(col))[::-1]
            idx = [i for i in idx if abs(col[i]) > 1e-10][:top_terms]

            if len(idx) == 0:
                blocks.append(rf"\text{{Component {k+1}: 无有效隐式项}}")
                continue

            f_func = sp.Function(f"f_{k+1}")
            g_func = sp.Function(f"g_{k+1}")
            P_func = sp.Function(f"P_{k+1}")

            f = f_func(*c_syms)
            g = g_func(*c_syms)
            P = P_func(omega_sym)
            D_expr = f * sp.exp(sp.I * omega_sym * g) * P # type: ignore
            D_base = sp.exp(sp.I * omega_sym * g) * P # type: ignore

            Eq = sp.S(0)
            for i in idx:
                coef = float(np.real(col[i]))
                opname = operator_names[i]
                
                parts = opname.split('·')
                expr = sp.S(1)
                for part in parts:
                    part = part.strip()
                    if part == '1' or part == '':
                        continue
                    if part == 'iω':
                        expr = expr * sp.I * omega_sym
                    elif part.startswith('c_'):
                        c_idx = int(part.split('c_')[-1]) - 1
                        expr = expr * c_syms[c_idx]
                    elif part.startswith('∂²D̂') or part.startswith('∂²D?'):
                        matches = re.findall(r'∂c_(\d+)(²?)', part)
                        sub_expr = D_expr
                        for m, p in matches:
                            sub_expr = sp.diff(sub_expr, c_syms[int(m)-1])
                            if p == '²':
                                sub_expr = sp.diff(sub_expr, c_syms[int(m)-1])
                        w_matches = re.findall(r'∂ω(²?)', part)
                        for p in w_matches:
                            sub_expr = sp.diff(sub_expr, omega_sym)
                            if p == '²':
                                sub_expr = sp.diff(sub_expr, omega_sym)
                        expr = expr * sub_expr
                    elif part.startswith('∂D̂(c,ω)/∂c_') or part.startswith('∂D?(c,ω)/∂c_'):
                        c_idx = int(part.split('c_')[-1]) - 1
                        expr = expr * sp.diff(D_expr, c_syms[c_idx])
                    elif part.startswith('∂D̂(c,ω)/∂ω') or part.startswith('∂D?(c,ω)/∂ω'):
                        expr = expr * sp.diff(D_expr, omega_sym)
                    elif part == 'D̂' or part == 'D̂(c,ω)' or part == 'D?(c,ω)':
                        expr = expr * D_expr
                    elif part == 'D̂²':
                        expr = expr * (D_expr**2)
                    elif part == '(iω)²':
                        expr = expr * (sp.I * omega_sym)**2
                    elif part.startswith('∂D̂/∂c_'):
                        c_idx = int(part.split('∂c_')[-1]) - 1
                        expr = expr * sp.diff(D_expr, c_syms[c_idx])
                    elif part.startswith('∂D̂/∂ω'):
                        expr = expr * sp.diff(D_expr, omega_sym)

                # 引入非线性项后，每一项的 P 维度不同。
                # 由于 D_expr = f * exp(iwg) * P，如果该算子原本对应的最高阶次为 m，
                # 我们可以统一令 P = 1，将流形的推导完全降维到 (f, g) 所在的空间。
                Eq = Eq + sp.Float(coef) * expr

            # == 对方程整体提出 P_k 因子 ==
            # 因为每一项都会有 f * exp(iwg) * P，非线性项(如D^2)会产生 P^2
            # 我们可以强制剥离 P 并在之后消除它们
            
            # 首先令 P=1 消除这层依赖，这是由于我们实际关心的就是响应分布而不是谱底
            Eq_no_P = Eq.subs(P_func(omega_sym), 1)

            # 对消除过 P 的公式进行整理和展开
            Eq_expanded = sp.expand(Eq_no_P)

            # 提取包含 f() 的常数项 (ω=0) 得到方程
            eq_f = sp.simplify(Eq_expanded.subs(omega_sym, 0))

            # 提取 ω 对应的一阶偏导得到 g 方程
            eq_g_term = sp.simplify(Eq_expanded.coeff(omega_sym, 1))
            eq_g = sp.simplify(eq_g_term / sp.I) if eq_g_term != 0 else sp.S(0)

            # --- 强制清洗任何残留的复数相位和 P(0)
            # 比如方程如果是 A * exp(...) / P(...) 我们把 exp 等提取消掉
            def _clean_equation(eq_in):
                if eq_in == 0:
                    return eq_in
                eq_clean = sp.simplify(eq_in)
                # 分离乘积因子，如果因子中没有我们要解的主元变量 f 或 g 等导数本身，其实就是常数因子等于0的废料
                # 这里我们使用一个强硬的手段：把指数项 subs 成 1 (因为 e^(i...) 永不为0)
                eq_clean = eq_clean.replace(sp.exp, lambda x: 1)
                
                # 获取分子
                num, _ = sp.fraction(sp.cancel(eq_clean))
                return num

            eq_f = _clean_equation(eq_f)
            eq_g = _clean_equation(eq_g)

            eq_f_str = sp.latex(eq_f) + " = 0" if eq_f != 0 else "0 = 0"
            eq_g_str = sp.latex(eq_g) + " = 0" if eq_g != 0 else "0 = 0"

            ics_f = {f.subs({c: 0 for c in c_syms}): 1}
            ics_g = {g.subs({c: 0 for c in c_syms}): 0}

            f_sol_str = ""
            g_sol_str = ""
            
            if eq_f != 0:
                try:
                    if len(c_syms) == 1:
                        f_sol = sp.dsolve(sp.Eq(eq_f, 0), f, ics=ics_f)
                    else:
                        try:
                            f_sol = sp.pdsolve(sp.Eq(eq_f, 0), f, ics=ics_f)
                        except Exception:
                            f_sol = sp.pdsolve(sp.Eq(eq_f, 0), f)
                    if hasattr(f_sol, 'rhs'):
                        rhs = f_sol.rhs
                        for sym in rhs.free_symbols:
                            if str(sym).startswith('C'):
                                rhs = rhs.subs(sym, 1)
                        for func in rhs.atoms(sp.Function):
                            if str(func.func).startswith('F') or str(func.func).startswith('C'):
                                rhs = rhs.subs(func, 1)
                        f_sol_str = rf"\quad \implies f_{{{k+1}}}: " + sp.latex(rhs)
                except Exception:
                    pass

            if eq_g != 0:
                try:
                    if len(c_syms) == 1:
                        g_sol = sp.dsolve(sp.Eq(eq_g, 0), g, ics=ics_g)
                    else:
                        try:
                            g_sol = sp.pdsolve(sp.Eq(eq_g, 0), g, ics=ics_g)
                        except Exception:
                            g_sol = sp.pdsolve(sp.Eq(eq_g, 0), g)
                    if hasattr(g_sol, 'rhs'):
                        rhs = g_sol.rhs
                        for sym in rhs.free_symbols:
                            if str(sym).startswith('C'):
                                rhs = rhs.subs(sym, 0)
                        for func in rhs.atoms(sp.Function):
                            if str(func.func).startswith('F') or str(func.func).startswith('C'):
                                rhs = rhs.subs(func, 0)
                        g_sol_str = rf"\quad \implies g_{{{k+1}}}: " + sp.latex(rhs)
                except Exception:
                    pass

            block = (
                "\\begin{aligned}\n"
                f"& \\text{{Component {k+1}}}: \\\\\n"
                f"& \\text{{f方程: }} {eq_f_str} {f_sol_str} \\\\\n"
                f"& \\text{{g方程: }} {eq_g_str} {g_sol_str}\n"
                "\\end{aligned}"
            )
            blocks.append(block)

    except Exception:
        # Fallback
        for k in range(k_eff):
            col = x[:, k]
            idx = np.argsort(np.abs(col))[::-1]
            idx = [i for i in idx if abs(col[i]) > 1e-10][:top_terms]

            if len(idx) == 0:
                blocks.append(rf"\text{{Component {k+1}: 无有效隐式项}}")
                continue

            terms = []
            for i in idx:
                coef = float(np.real(col[i]))
                opname = operator_names[i].replace("ω", r"\omega")
                opname = opname.replace("∂", r"\partial ")
                opname = opname.replace("·", r"\cdot ")
                terms.append(rf"({coef:.4g})\,{opname}")

            eq_terms = " \\\\ \n& \\quad + ".join(terms)
            eq = eq_terms + r" = 0"
            blocks.append(
                "\\begin{aligned}\n"
                f"& \\text{{隐式约束: }} \\\\\n"
                f"& \\quad {eq}\n"
                "\\end{aligned}"
            )

    return blocks
