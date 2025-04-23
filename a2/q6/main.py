import re
import numpy as np

def cif_to_data(cif_path):
    cell = {}
    asym_sites = []
    sym_strs = []

    with open(cif_path) as f:
        lines = iter(f)
        for raw in lines:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue

            m = re.match(r'_cell_length_([abc])\s+([\d.]+)', line)
            if m:
                cell[m.group(1)] = float(m.group(2))
                continue

            m = re.match(r'_cell_angle_(alpha|beta|gamma)\s+([\d.]+)', line)
            if m:
                cell[m.group(1)] = float(m.group(2))
                continue

            if line.lower() == 'loop_':
                headers = []
                for ln in lines:
                    s = ln.strip()
                    if s.startswith('_'):
                        headers.append(s)
                        continue
                    data_first = s
                    break

                if any(h.lower().startswith('_symmetry_equiv_pos_as_xyz') for h in headers):
                    row = data_first
                    while row and not row.lower().startswith('loop_') and not row.startswith('_'):
                        sym_strs.append(row)
                        row = next(lines, '').strip()
                elif '_atom_site_type_symbol' in headers:
                    want = {
                        '_atom_site_type_symbol',
                        '_atom_site_fract_x',
                        '_atom_site_fract_y',
                        '_atom_site_fract_z'
                    }
                    if want.issubset(headers):
                        idx = {h: i for i, h in enumerate(headers)}
                        row = data_first
                        while row:
                            parts = row.split()
                            if len(parts) != len(headers):
                                break

                            sym = parts[idx['_atom_site_type_symbol']]
                            fx = float(parts[idx['_atom_site_fract_x']].split('(')[0])
                            fy = float(parts[idx['_atom_site_fract_y']].split('(')[0])
                            fz = float(parts[idx['_atom_site_fract_z']].split('(')[0])

                            asym_sites.append((sym, np.array([fx, fy, fz])))
                            row = next(lines, '').strip()
                continue

    return cell, asym_sites, sym_strs

def parse_symop(expr):
    expr = expr.replace('x', 'frac[0]').replace('y', 'frac[1]').replace('z', 'frac[2]')
    expr = re.compile(r'(\d+)/(\d+)').sub(r'(\1/\2)', expr)
    return eval(f'lambda frac: np.array([{','.join(expr.split(','))}], float)')

def h_matrix(a, b, c, alpha, beta, gamma):
    alpha, beta, gamma = np.radians([alpha, beta, gamma])
    va = np.array([a, 0.0, 0.0])
    vb = np.array([b * np.cos(gamma), b * np.sin(gamma), 0.0])
    cx = c * np.cos(beta)
    cy = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
    cz = np.sqrt(max(c * c - cx * cx - cy * cy, 0.0))
    vc = np.array([cx, cy, cz])
    return np.vstack([va, vb, vc])

cell, asym_sites, sym_strs = cif_to_data('1008748.cif')
ops = [parse_symop(s) for s in sym_strs]
print(f'a: {cell['a']}, b: {cell['b']}, c: {cell['c']}')
print(f'alpha: {cell['alpha']}, beta: {cell['beta']}, gamma: {cell['gamma']}')
print(f'fract: {asym_sites}')

h = h_matrix(
    cell['a'], cell['b'], cell['c'],
    cell['alpha'], cell['beta'], cell['gamma']
)
print(f'h-matrix: {h}')
print(f'Determinant: {np.linalg.det(h):.2f}')
print(f'Scalar Triple Product: {np.dot(h[:, 0], np.cross(h[:, 1], h[:, 2])):.2f}')

coordinates = []
for sym, f0 in asym_sites:
    for op in ops:
        cart = op(f0) @ h
        coordinates.append((sym, cart))

repetitions = 2
supercell_coordinates = []
for i in range(repetitions):
    for j in range(repetitions):
        for k in range(repetitions):
            shift = np.array([i, j, k], float)
            for sym, f0 in asym_sites:
                for op in ops:
                    f1 = op(f0) + shift
                    cart = f1 @ h
                    supercell_coordinates.append((sym, cart))

with open('heavy-ice.xyz', 'w') as w:
    w.write(f'{len(coordinates)}\n')
    w.write(f'Heavy Ice\n')
    for sym, (x, y, z) in coordinates:
        w.write(f'{sym:2s} {x:12.4f} {y:12.4f} {z:12.4f}\n')

with open('heavy-ice-supercell.xyz', 'w') as w:
    w.write(f'{len(supercell_coordinates)}\n')
    w.write(f'Heavy Ice\n')
    for sym, (x, y, z) in supercell_coordinates:
        w.write(f'{sym:2s} {x:12.4f} {y:12.4f} {z:12.4f}\n')
