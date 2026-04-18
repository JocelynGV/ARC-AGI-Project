import numpy as np
from collections import deque

from ArcProblem import ArcProblem


class ArcAgent:

    def __init__(self):
        pass

    def make_predictions(self, arc_problem: ArcProblem) -> list[np.ndarray]:

        print(f"Solving problem {arc_problem._id}")

        test_input = np.array(
            arc_problem.test_set().get_input_data().data(),
            copy=True
        )

        predictions = [
            np.array(test_input, copy=True),
            np.array(test_input, copy=True),
            np.array(test_input, copy=True)
        ]

        # RULE IDENTIFICATION  (vote across all training pairs)

        rule_votes = {}

        for ex in arc_problem._training_data:
            in_g  = np.array(ex._input._arc_array)
            out_g = np.array(ex._output._arc_array)
            rule  = self.identify_rule(in_g, out_g)
            rule_votes[rule] = rule_votes.get(rule, 0) + 1

        identified_rule = max(rule_votes, key=rule_votes.get) if rule_votes else None
        print(f"Identified rule: {identified_rule}")

        # HARD-CODED RULE SOLVERS  (highest priority)

        RULE_SOLVERS = {
            "center_recolor":      
                self.solve_center_recolor,
            "diagonal_growth":     
                self.solve_diagonal_growth,
            "fill_inside_outside": 
                self.solve_fill_inside_outside,
            "edge_matching":       
                self.solve_edge_matching,
            "direction_growth":    
                self.solve_direction_growth,
            "rotation_flip":       
                self.solve_rotation_flip,
            "closed_recolor":      
                self.solve_closed_recolor,
            "divider_fill":        
                self.solve_divider_fill,
        }

        if identified_rule in RULE_SOLVERS:
            try:
                result = RULE_SOLVERS[identified_rule](test_input, arc_problem)
                predictions[0] = np.array(result, copy=True)
                print(f"  => Applied hard-coded solver: {identified_rule}")
            except Exception as e:
                print(f"  => Hard-coded solver failed ({e}), falling back to scoring")

        # =====================================================
        # BASIC GRID TRANSFORMS
        # =====================================================

        basic_transforms = [
            self.rotate90, 
            self.rotate180, 
            self.rotate270,
            self.flip_horizontal, 
            self.flip_vertical,
            self.transpose,
            self.mirror_left_to_right, 
            self.mirror_bottom_to_top,
            self.mirror_tile_2x2,
            self.mirror_tile_horizontal,
            self.mirror_tile_vertical,
            self.invert_majority_color,
            self.trim,
        ]

        color_flip = self.build_dynamic_color_flip(arc_problem)
        if color_flip is not None:
            basic_transforms.append(color_flip)

        # =====================================================
        # SCORE SINGLE TRANSFORMS
        # =====================================================

        transform_scores = []

        for transform in basic_transforms:
            score = self.score_transform_on_training(transform, arc_problem)
            transform_scores.append((transform, score))

        # GREEDY CHAIN SEARCH

        def score_chain(chain):
            total = 0
            for training in arc_problem._training_data:
                grid = np.array(training._input._arc_array, copy=True)
                for t in chain:
                    grid = t(grid)
                total += self.combined_score(training._output._arc_array, grid)
            return total / max(len(arc_problem._training_data), 1)

        best_chain = []
        best_chain_score = score_chain([])

        for _ in range(4):
            improved = False
            for transform, _ in transform_scores:
                candidate = best_chain + [transform]
                score     = score_chain(candidate)
                if score > best_chain_score:
                    best_chain       = candidate
                    best_chain_score = score
                    improved         = True
            if not improved:
                break

        def apply_chain(grid):
            g = np.array(grid, copy=True)
            for t in best_chain:
                g = t(g)
            return g

        apply_chain.__name__ = "apply_chain"
        transform_scores.append((apply_chain, best_chain_score))

        # OBJECT RECOLOR + SPATIAL CANDIDATES

        for t in self.object_based_candidates(arc_problem):
            score = self.score_transform_on_training(t, arc_problem)
            transform_scores.append((t, score))

        for t in self.object_spatial_candidates(arc_problem):
            score = self.score_transform_on_training(t, arc_problem)
            transform_scores.append((t, score))

        # RANK ALL CANDIDATES

        transform_scores.sort(key=lambda x: x[1], reverse=True)

        print("\nTop ranked strategies:")
        for t, s in transform_scores[:5]:
            print(f"  {t.__name__}  {round(s, 4)}")

        # fill slots 1 & 2 with best scored transforms
        filled = 0
        for transform, _ in transform_scores:
            if filled >= 2:
                break
            try:
                predictions[filled + 1] = np.array(transform(test_input), copy=True)
                filled += 1
            except Exception:
                pass

        return predictions

    # RULE IDENTIFICATION

    def identify_rule(self, input_grid, output_grid):

        in_objs  = self.find_objects(input_grid)
        out_objs = self.find_objects(output_grid)

        # Rule 1 — center recolor + shrink
        if self._many_single_pixels(in_objs) and output_grid.shape != input_grid.shape:
            return "center_recolor"

        # Rule 2 — diagonal growth from two equal squares
        if self._two_equal_squares(in_objs):
            return "diagonal_growth"

        # Rule 3 — fill inside red / outside green
        if self._large_shape_present(in_objs, input_grid):
            out_vals = np.unique(output_grid)
            if len(out_vals) >= 3:          # shape + 2 fill colors
                return "fill_inside_outside"

        # Rule 4 — edge matching lines
        if self._border_cells_present(input_grid) and self._has_full_rows_or_cols(output_grid):
            return "edge_matching"

        # Rule 5 — triangle / direction growth
        if self._triangle_present(in_objs):
            return "direction_growth"

        # Rule 6 — rotation then flip
        if self._rotation_match(input_grid, output_grid):
            return "rotation_flip"

        # Rule 7 — closed shapes get recolored
        if self._closed_shapes_exist(in_objs):
            return "closed_recolor"

        # Rule 8 — red divider, intersect sub-grids
        if self._divider_present(input_grid):
            return "divider_fill"

        if self._objects_moved(in_objs, out_objs):
            return "object_move"

        if self._objects_scaled(in_objs, out_objs):
            return "object_scale"

        if self._objects_duplicated(in_objs, out_objs):
            return "object_duplicate"

        return "unknown"


    # ---- detection helpers ----------------------------------------

    def _many_single_pixels(self, objs, threshold=4):
        return sum(1 for o in objs if o.size == 1) >= threshold

    def _two_equal_squares(self, objs):
        squares = [o for o in objs
                   if o.height == o.width and o.size == o.height * o.width and o.size >= 4]
        return len(squares) == 2 and squares[0].size == squares[1].size

    def _large_shape_present(self, objs, grid, frac=0.15):
        return any(o.size >= grid.size * frac for o in objs)

    def _border_cells_present(self, grid):
        border = np.concatenate([grid[0, :], grid[-1, :],
                                  grid[1:-1, 0], grid[1:-1, -1]])
        return np.any(border != 0)

    def _has_full_rows_or_cols(self, grid):
        for r in range(grid.shape[0]):
            if np.all(grid[r, :] != 0):
                return True
        for c in range(grid.shape[1]):
            if np.all(grid[:, c] != 0):
                return True
        return False

    def _triangle_present(self, objs):
        for o in objs:
            rows = sorted(set(r for r, c in o.cells))
            if len(rows) >= 3:
                widths = [sum(1 for r2, _ in o.cells if r2 == row) for row in rows]
                if widths == sorted(widths) or widths == sorted(widths, reverse=True):
                    return True
        return False

    def _rotation_match(self, input_grid, output_grid):
        return any(np.array_equal(np.rot90(input_grid, k), output_grid) for k in [1, 2, 3])

    def _closed_shapes_exist(self, objs):
        return any(o.has_hole() for o in objs)

    def _divider_present(self, grid):
        for r in range(grid.shape[0]):
            if np.all(grid[r, :] == grid[r, 0]) and grid[r, 0] != 0:
                return True
        for c in range(grid.shape[1]):
            if np.all(grid[:, c] == grid[0, c]) and grid[0, c] != 0:
                return True
        return False

    def _objects_moved(self, in_objs, out_objs):
        if len(in_objs) != len(out_objs):
            return False
        return any(
            (a.min_row != b.min_row or a.min_col != b.min_col) and a.color == b.color
            for a, b in zip(in_objs, out_objs)
        )

    def _objects_scaled(self, in_objs, out_objs):
        if not in_objs or not out_objs:
            return False
        ratio = out_objs[0].size / max(in_objs[0].size, 1)
        return any(abs(ratio - f) < 0.3 for f in [2, 3, 4, 0.5, 0.25])

    def _objects_duplicated(self, in_objs, out_objs):
        return len(out_objs) > len(in_objs)


    # ============================================================
    # RULE 1 — CENTER RECOLOR + SHRINK
    # Surrounding 1x1 pixels disappear; center object adopts
    # their color; grid crops to the center object's bounding box.
    # ============================================================

    def solve_center_recolor(self, grid, arc_problem=None):

        objs        = self.find_objects(grid)
        singles     = [o for o in objs if o.size == 1]
        non_singles = [o for o in objs if o.size > 1]

        if not singles or not non_singles:
            return grid

        new_color = singles[0].color
        center    = max(non_singles, key=lambda o: o.size)

        h   = center.height
        w   = center.width
        out = np.zeros((h, w), dtype=grid.dtype)

        for r, c in center.cells:
            out[r - center.min_row, c - center.min_col] = new_color

        return out


    # ============================================================
    # RULE 2 — DIAGONAL GROWTH FROM TWO EQUAL SQUARES
    # Each square emits a diagonal stream through the grid.
    # ============================================================

    def solve_diagonal_growth(self, grid, arc_problem=None):

        objs    = self.find_objects(grid)
        squares = [o for o in objs
                   if o.height == o.width and o.size == o.height * o.width and o.size >= 4]

        if len(squares) < 2:
            return grid

        sq1, sq2 = squares[0], squares[1]
        out      = grid.copy()
        h, w     = grid.shape

        r1 = (sq1.min_row + sq1.max_row) / 2
        c1 = (sq1.min_col + sq1.max_col) / 2
        r2 = (sq2.min_row + sq2.max_row) / 2
        c2 = (sq2.min_col + sq2.max_col) / 2

        dr = int(np.sign(r2 - r1))
        dc = int(np.sign(c2 - c1))

        if dr == 0 and dc == 0:
            return grid

        for sq in [sq1, sq2]:
            cr = int((sq.min_row + sq.max_row) / 2)
            cc = int((sq.min_col + sq.max_col) / 2)
            for direction in [1, -1]:
                r, c = cr + direction * dr, cc + direction * dc
                while 0 <= r < h and 0 <= c < w:
                    if out[r, c] == 0:
                        out[r, c] = sq.color
                    r += direction * dr
                    c += direction * dc

        return out


    # ============================================================
    # RULE 3 — FILL INSIDE RED, OUTSIDE GREEN
    # Flood from border = outside color; remainder = inside color.
    # ============================================================

    def solve_fill_inside_outside(self, grid, arc_problem=None):

        inside_color  = 2   # red  (ARC default)
        outside_color = 3   # green

        # infer colors from training output
        if arc_problem is not None:
            for ex in arc_problem._training_data:
                out_g = np.array(ex._output._arc_array)
                in_g  = np.array(ex._input._arc_array)
                # cells that were 0 in input but colored in output
                changed = (in_g == 0) & (out_g != 0)
                new_colors = out_g[changed]
                if len(new_colors) >= 2:
                    vals, counts = np.unique(new_colors, return_counts=True)
                    sorted_pairs = sorted(zip(vals, counts), key=lambda x: x[1])
                    inside_color  = sorted_pairs[0][0]   # less frequent = inside
                    outside_color = sorted_pairs[-1][0]  # more frequent = outside
                    break

        h, w    = grid.shape
        out     = grid.copy()
        visited = np.zeros((h, w), dtype=bool)
        visited[grid != 0] = True   # shape cells block flood

        queue = deque()
        for r in range(h):
            for c in [0, w - 1]:
                if not visited[r, c]:
                    visited[r, c] = True
                    out[r, c]     = outside_color
                    queue.append((r, c))
        for c in range(w):
            for r in [0, h - 1]:
                if not visited[r, c]:
                    visited[r, c] = True
                    out[r, c]     = outside_color
                    queue.append((r, c))

        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        while queue:
            r, c = queue.popleft()
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc]:
                    visited[nr, nc] = True
                    out[nr, nc]     = outside_color
                    queue.append((nr, nc))

        out[(grid == 0) & (~visited)] = inside_color
        return out


    # ============================================================
    # RULE 4 — EDGE MATCHING LINES
    # Opposite border cells with matching color → fill that row/col.
    # ============================================================

    def solve_edge_matching(self, grid, arc_problem=None):

        h, w = grid.shape
        out  = grid.copy()

        for r in range(h):
            left, right = grid[r, 0], grid[r, -1]
            if left != 0 and left == right:
                out[r, :] = left

        for c in range(w):
            top, bottom = grid[0, c], grid[-1, c]
            if top != 0 and top == bottom:
                out[:, c] = top

        return out


    # ============================================================
    # RULE 5 — DIRECTION GROWTH (TRIANGLE / ARROW)
    # Detect pointing direction from narrowing rows/cols,
    # project a stream from the tip to the grid edge.
    # ============================================================

    def solve_direction_growth(self, grid, arc_problem=None):

        objs = self.find_objects(grid)
        out  = grid.copy()
        h, w = grid.shape

        for o in objs:
            rows = sorted(set(r for r, c in o.cells))
            cols = sorted(set(c for r, c in o.cells))

            if len(rows) < 2 and len(cols) < 2:
                continue

            # --- determine direction from row widths ---
            row_widths = [sum(1 for r2, _ in o.cells if r2 == row) for row in rows]
            col_heights = [sum(1 for _, c2 in o.cells if c2 == col) for col in cols]

            if len(rows) >= 2:
                if row_widths[-1] < row_widths[0]:
                    # narrows downward → tip at bottom, points down
                    tip_row = rows[-1]
                    tip_col = int(np.mean([c for r, c in o.cells if r == tip_row]))
                    dr, dc  = 1, 0
                elif row_widths[0] < row_widths[-1]:
                    # narrows upward → tip at top, points up
                    tip_row = rows[0]
                    tip_col = int(np.mean([c for r, c in o.cells if r == tip_row]))
                    dr, dc  = -1, 0
                elif len(cols) >= 2 and col_heights[-1] < col_heights[0]:
                    tip_col = cols[-1]
                    tip_row = int(np.mean([r for r, c in o.cells if c == tip_col]))
                    dr, dc  = 0, 1
                elif len(cols) >= 2:
                    tip_col = cols[0]
                    tip_row = int(np.mean([r for r, c in o.cells if c == tip_col]))
                    dr, dc  = 0, -1
                else:
                    continue
            else:
                continue

            r, c = tip_row + dr, tip_col + dc
            while 0 <= r < h and 0 <= c < w:
                if out[r, c] == 0:
                    out[r, c] = o.color
                r += dr
                c += dc

        return out


    # ============================================================
    # RULE 6 — ROTATION THEN FLIP
    # Score all 8 dihedral transforms on training, apply the best.
    # ============================================================

    def solve_rotation_flip(self, grid, arc_problem=None):

        transforms = [
            ("r0",      lambda g: g.copy()),
            ("r90",     lambda g: np.rot90(g, 1)),
            ("r180",    lambda g: np.rot90(g, 2)),
            ("r270",    lambda g: np.rot90(g, 3)),
            ("r0_fh",   lambda g: np.fliplr(g)),
            ("r90_fh",  lambda g: np.fliplr(np.rot90(g, 1))),
            ("r180_fh", lambda g: np.fliplr(np.rot90(g, 2))),
            ("r270_fh", lambda g: np.fliplr(np.rot90(g, 3))),
        ]

        best_score = -1
        best_fn    = transforms[1][1]   # default: rot90

        if arc_problem is not None:
            for _, fn in transforms:
                score = 0
                for ex in arc_problem._training_data:
                    try:
                        pred  = fn(np.array(ex._input._arc_array))
                        score += self.combined_score(
                            np.array(ex._output._arc_array), pred)
                    except Exception:
                        pass
                score /= max(len(arc_problem._training_data), 1)
                if score > best_score:
                    best_score = score
                    best_fn    = fn

        return best_fn(grid)


    # ============================================================
    # RULE 7 — CLOSED SHAPES → CYAN (or inferred color)
    # ============================================================

    def solve_closed_recolor(self, grid, arc_problem=None):

        target_color = 8   # cyan default

        if arc_problem is not None:

            color_votes = {}

            for ex in arc_problem._training_data:

                in_g     = np.array(ex._input._arc_array)
                out_g    = np.array(ex._output._arc_array)
                in_objs  = self.find_objects(in_g)
                out_objs = self.find_objects(out_g)

                # index output objects by top-left corner for fast lookup
                out_by_pos = {(o.min_row, o.min_col): o for o in out_objs}

                for i_obj in in_objs:

                    if not i_obj.has_hole():
                        continue

                    # find matching output object at the same position
                    o_obj = out_by_pos.get((i_obj.min_row, i_obj.min_col))

                    if o_obj is not None and o_obj.color != i_obj.color:
                        color_votes[o_obj.color] = color_votes.get(o_obj.color, 0) + 1

            if color_votes:
                target_color = max(color_votes, key=color_votes.get)

        # open shapes keep their color (grid.copy() baseline)
        # closed shapes get recolored to target_color
        out = grid.copy()
        for o in self.find_objects(grid):
            if o.has_hole():
                for r, c in o.cells:
                    out[r, c] = target_color

        return out


    # ============================================================
    # RULE 8 — DIVIDER FILL (intersect sub-grid panels)
    # Red column/row divides grid into panels.
    # Output = cells where ALL panels share the same non-zero color.
    # ============================================================

    def solve_divider_fill(self, grid, arc_problem=None):

        h, w = grid.shape

        divider_cols = [c for c in range(w)
                        if np.all(grid[:, c] == grid[0, c]) and grid[0, c] != 0]
        divider_rows = [r for r in range(h)
                        if np.all(grid[r, :] == grid[r, 0]) and grid[r, 0] != 0]

        panels = []

        if divider_cols:
            splits = [-1] + divider_cols + [w]
            for i in range(len(splits) - 1):
                s, e = splits[i] + 1, splits[i + 1]
                if s < e:
                    panels.append(grid[:, s:e])
        elif divider_rows:
            splits = [-1] + divider_rows + [h]
            for i in range(len(splits) - 1):
                s, e = splits[i] + 1, splits[i + 1]
                if s < e:
                    panels.append(grid[s:e, :])

        if len(panels) < 2:
            return grid

        min_h = min(p.shape[0] for p in panels)
        min_w = min(p.shape[1] for p in panels)
        panels = [p[:min_h, :min_w] for p in panels]

        out = np.zeros((min_h, min_w), dtype=grid.dtype)
        for r in range(min_h):
            for c in range(min_w):
                colors = [p[r, c] for p in panels if p[r, c] != 0]
                if colors and len(set(colors)) == 1:
                    out[r, c] = colors[0]

        return out

    # OBJECT RECOLOR CANDIDATES  (scoring-based)
  
    def object_based_candidates(self, arc_problem):

        candidates = []

        for example in arc_problem._training_data:

            input_grid  = np.array(example._input._arc_array)
            output_grid = np.array(example._output._arc_array)
            in_objs = self.find_objects(input_grid)
            out_objs = self.find_objects(output_grid)

            changed = [
                (o, out_objs[i])
                for i, o in enumerate(in_objs)
                if i < len(out_objs) and o.color != out_objs[i].color
            ]

            if not changed:
                continue

            def _make(condition_fn, new_c, fname):
                def fn(grid, _c=new_c, _cond=condition_fn):
                    objs     = self.find_objects(grid)
                    new_grid = grid.copy()
                    for o in objs:
                        if _cond(o):
                            for r, c in o.cells:
                                new_grid[r, c] = _c
                    return new_grid
                fn.__name__ = fname
                return fn

            new_color = changed[0][1].color

            if all(o.has_hole() for o, _ in changed):
                candidates.append(_make(lambda o: o.has_hole(), new_color, "recolor_closed"))

            if all(o.touches_border() for o, _ in changed):
                candidates.append(_make(lambda o: o.touches_border(), new_color, "recolor_border"))

            smallest_size = min(o.size for o in in_objs)
            if any(o.size == smallest_size for o, _ in changed):
                candidates.append(_make(
                    lambda o, s=smallest_size: o.size == s, new_color, "recolor_smallest"))

            largest_size = max(o.size for o in in_objs)
            if any(o.size == largest_size for o, _ in changed):
                candidates.append(_make(
                    lambda o, s=largest_size: o.size == s, new_color, "recolor_largest"))

        return candidates

    # OBJECT SPATIAL CANDIDATES

    def object_spatial_candidates(self, arc_problem):
        candidates = []
        candidates += self._build_move_candidates(arc_problem)
        candidates += self._build_scale_candidates(arc_problem)
        candidates += self._build_per_object_rotation_candidates(arc_problem)
        candidates += self._build_duplicate_candidates(arc_problem)
        return candidates


    def _build_move_candidates(self, arc_problem):

        shifts = {}
        for ex in arc_problem._training_data:
            in_by  = {o.color: o for o in self.find_objects(np.array(ex._input._arc_array))}
            out_by = {o.color: o for o in self.find_objects(np.array(ex._output._arc_array))}
            for color in in_by:
                if color in out_by and color not in shifts:
                    shifts[color] = (out_by[color].min_row - in_by[color].min_row,
                                     out_by[color].min_col - in_by[color].min_col)

        candidates = []

        if shifts:
            def move_objects(grid, _s=shifts):
                objs     = self.find_objects(grid)
                h, w     = grid.shape
                new_grid = np.zeros_like(grid)
                for o in objs:
                    dr, dc = _s.get(o.color, (0, 0))
                    for r, c in o.cells:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            new_grid[nr, nc] = o.color
                return new_grid
            move_objects.__name__ = "move_objects"
            candidates.append(move_objects)

        def move_toward_center(grid):
            objs     = self.find_objects(grid)
            h, w     = grid.shape
            new_grid = np.zeros_like(grid)
            for o in objs:
                dr = int(np.sign(h // 2 - (o.min_row + o.max_row) // 2))
                dc = int(np.sign(w // 2 - (o.min_col + o.max_col) // 2))
                for r, c in o.cells:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        new_grid[nr, nc] = o.color
            return new_grid

        move_toward_center.__name__ = "move_toward_center"
        candidates.append(move_toward_center)
        return candidates


    def _build_scale_candidates(self, arc_problem):

        factor = None
        for ex in arc_problem._training_data:
            in_objs  = self.find_objects(np.array(ex._input._arc_array))
            out_objs = self.find_objects(np.array(ex._output._arc_array))
            if in_objs and out_objs:
                ratio = out_objs[0].size / max(in_objs[0].size, 1)
                for f in [2, 3, 4, 0.5, 0.25]:
                    if abs(ratio - f) < 0.3:
                        factor = f
                        break
            if factor:
                break

        candidates = []
        for f in ([factor] if factor else [2, 3]):
            def make_fn(scale):
                def scale_objects(grid):
                    objs     = self.find_objects(grid)
                    h, w     = grid.shape
                    new_h    = max(h, int(h * scale)) if scale > 1 else h
                    new_w    = max(w, int(w * scale)) if scale > 1 else w
                    new_grid = np.zeros((new_h, new_w), dtype=grid.dtype)
                    for o in objs:
                        shape = o.shape_matrix()
                        s     = int(scale) if scale >= 1 else 1
                        big   = np.kron(shape, np.ones((s, s), dtype=int)) \
                                if scale >= 1 else shape[::int(1/scale), ::int(1/scale)]
                        nr = int(o.min_row * (scale if scale > 1 else 1))
                        nc = int(o.min_col * (scale if scale > 1 else 1))
                        for dr in range(big.shape[0]):
                            for dc in range(big.shape[1]):
                                rr, cc = nr + dr, nc + dc
                                if 0 <= rr < new_h and 0 <= cc < new_w and big[dr, dc] != 0:
                                    new_grid[rr, cc] = big[dr, dc]
                    return new_grid
                scale_objects.__name__ = f"scale_objects_x{scale}"
                return scale_objects
            candidates.append(make_fn(f))

        return candidates


    def _build_per_object_rotation_candidates(self, arc_problem):

        ops = {
            "rot90_objects":  lambda m: np.rot90(m, 1),
            "rot180_objects": lambda m: np.rot90(m, 2),
            "rot270_objects": lambda m: np.rot90(m, 3),
            "fliph_objects":  lambda m: np.fliplr(m),
            "flipv_objects":  lambda m: np.flipud(m),
        }

        candidates = []
        for name, op in ops.items():
            def make_fn(operation, fname):
                def rotate_each(grid):
                    objs     = self.find_objects(grid)
                    new_grid = np.zeros_like(grid)
                    h, w     = grid.shape
                    for o in objs:
                        rotated = operation(o.shape_matrix())
                        for dr in range(rotated.shape[0]):
                            for dc in range(rotated.shape[1]):
                                rr, cc = o.min_row + dr, o.min_col + dc
                                if 0 <= rr < h and 0 <= cc < w and rotated[dr, dc] != 0:
                                    new_grid[rr, cc] = rotated[dr, dc]
                    return new_grid
                rotate_each.__name__ = fname
                return rotate_each
            candidates.append(make_fn(op, name))

        return candidates


    def _build_duplicate_candidates(self, arc_problem):

        candidates = []

        for axis, fname in [(0, "tile_axis0"), (1, "tile_axis1"), (-1, "tile_both")]:
            def make_fn(ax, fn):
                def tile(grid):
                    if ax == 0:   return np.vstack([grid, grid])
                    elif ax == 1: return np.hstack([grid, grid])
                    else:         return np.block([[grid, grid], [grid, grid]])
                tile.__name__ = fn
                return tile
            candidates.append(make_fn(axis, fname))

        def duplicate_objects_offset(grid):
            objs     = self.find_objects(grid)
            new_grid = grid.copy()
            h, w     = grid.shape
            for o in objs:
                for r, c in o.cells:
                    for nr, nc in [(r+o.height, c), (r, c+o.width), (r+o.height, c+o.width)]:
                        if 0 <= nr < h and 0 <= nc < w:
                            new_grid[nr, nc] = o.color
            return new_grid

        duplicate_objects_offset.__name__ = "duplicate_objects_offset"
        candidates.append(duplicate_objects_offset)

        def duplicate_objects_mirror(grid):
            objs     = self.find_objects(grid)
            new_grid = grid.copy()
            h, w     = grid.shape
            for o in objs:
                for r, c in o.cells:
                    nr, nc = h - 1 - r, w - 1 - c
                    if 0 <= nr < h and 0 <= nc < w:
                        new_grid[nr, nc] = o.color
            return new_grid

        duplicate_objects_mirror.__name__ = "duplicate_objects_mirror"
        candidates.append(duplicate_objects_mirror)

        return candidates

    # SCORING HELPERS

    def score_transform_on_training(self, transform, arc_problem):
        total = 0
        for training in arc_problem._training_data:
            try:
                pred  = transform(np.array(training._input._arc_array, copy=True))
                total += self.combined_score(training._output._arc_array, pred)
            except Exception:
                pass
        return total / max(len(arc_problem._training_data), 1)

    def combined_score(self, true, pred):
        return 0.8 * self.pixel_accuracy(true, pred) + 0.2 * self.color_match(true, pred)

    def pixel_accuracy(self, a, b):
        if a.shape != b.shape:
            return 0
        return float(np.mean(a == b))

    def color_match(self, a, b):
        av, ac = np.unique(a, return_counts=True)
        bv, bc = np.unique(b, return_counts=True)
        d1     = dict(zip(av, ac))
        d2     = dict(zip(bv, bc))
        colors = set(d1) | set(d2)
        diff   = sum(abs(d1.get(c, 0) - d2.get(c, 0)) for c in colors)
        return 1 - diff / (2 * max(a.size, 1))

    # GRID TRANSFORMS

    def rotate90(self, g):              
        return np.rot90(g, 1)
    def rotate180(self, g):             
        return np.rot90(g, 2)
    def rotate270(self, g):             
        return np.rot90(g, 3)
    def flip_horizontal(self, g):       
        return np.fliplr(g)
    def flip_vertical(self, g):         
        return np.flipud(g)
    def transpose(self, g):             
        return g.T

    def mirror_tile_2x2(self, g):
        """3x3 → 6x6: tile as [g, fliplr(g)] / [flipud(g), rot180(g)]"""
        top = np.hstack([g, np.fliplr(g)])
        bot = np.hstack([np.flipud(g), np.rot90(g, 2)])
        return np.vstack([top, bot])

    def mirror_tile_horizontal(self, g):
        """3x3 → 3x6: append flipped copy to the right"""
        return np.hstack([g, np.fliplr(g)])

    def mirror_tile_vertical(self, g):
        """3x3 → 6x3: append flipped copy below"""
        return np.vstack([g, np.flipud(g)])

    def mirror_left_to_right(self, g):
        new = g.copy()
        for j in range(g.shape[1] // 2):
            new[:, j] = g[:, g.shape[1] - 1 - j]
        return new

    def mirror_bottom_to_top(self, g):
        new = g.copy()
        for i in range(g.shape[0] // 2):
            new[i] = g[g.shape[0] - 1 - i]
        return new

    def invert_majority_color(self, g):
        v, c  = np.unique(g, return_counts=True)
        pairs = [(v[i], c[i]) for i in range(len(v)) if v[i] != 0]
        if not pairs:
            return g.copy()
        majority = max(pairs, key=lambda x: x[1])[0]
        new = g.copy()
        new[new == majority] = 0
        new[new != 0]        = majority
        return new

    def trim(self, g, background=0):
        rows = np.any(g != background, axis=1)
        cols = np.any(g != background, axis=0)
        return g[rows][:, cols]

    def build_dynamic_color_flip(self, arc_problem):
        colors = None
        for ex in arc_problem._training_data:
            grid = ex._input._arc_array
            vals, _ = np.unique(grid[grid != 0], return_counts=True)
            if len(vals) >= 2:
                colors = vals[:2]
        if colors is None:
            return None

        def flip(g, _colors=colors):
            a, b       = _colors
            new        = g.copy()
            new[g == a] = -1
            new[g == b] = a
            new[new == -1] = b
            return new

        flip.__name__ = "flip_colors"
        return flip

    # OBJECT DETECTION

    def detect_background(self, grid):
        """Return the most frequent color — that is the background."""
        vals, counts = np.unique(grid, return_counts=True)
        return int(vals[np.argmax(counts)])

    def find_objects(self, grid, background=None):

        if background is None:
            background = self.detect_background(grid)

        visited = np.zeros_like(grid, dtype=bool)
        objs    = []
        dirs    = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):

                if visited[r, c] or grid[r, c] == background:
                    continue

                color = grid[r, c]
                queue = deque([(r, c)])
                visited[r, c] = True
                cells = []

                while queue:
                    cr, cc = queue.popleft()
                    cells.append((cr, cc))
                    for dr, dc in dirs:
                        nr, nc = cr + dr, cc + dc
                        if (0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]
                                and not visited[nr, nc] and grid[nr, nc] == color):
                            visited[nr, nc] = True
                            queue.append((nr, nc))

                objs.append(ArcObject(color, cells, grid.shape, background))

        return objs

# ARC OBJECT

class ArcObject:

    def __init__(self, color, cells, grid_shape, background=0):

        self.color      = color
        self.cells      = cells
        self.size       = len(cells)
        self.grid_shape = grid_shape
        self.background = background

        rows = [r for r, c in cells]
        cols = [c for r, c in cells]

        self.min_row = min(rows)
        self.max_row = max(rows)
        self.min_col = min(cols)
        self.max_col = max(cols)

        self.height = self.max_row - self.min_row + 1
        self.width  = self.max_col - self.min_col + 1

    def touches_border(self):
        h, w = self.grid_shape
        return any(r in [0, h - 1] or c in [0, w - 1] for r, c in self.cells)

    def shape_matrix(self):
        # fill with background so has_hole flood-fill treats gaps correctly
        g = np.full((self.height, self.width), self.background, dtype=int)
        for r, c in self.cells:
            g[r - self.min_row, c - self.min_col] = self.color
        return g

    def has_hole(self):

        shape   = self.shape_matrix()
        visited = np.zeros_like(shape, dtype=bool)
        q       = deque()
        h, w    = shape.shape

        bg = self.background
        for r in range(h):
            for c in range(w):
                if (r in [0, h - 1] or c in [0, w - 1]) and shape[r, c] == bg:
                    visited[r, c] = True
                    q.append((r, c))

        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        while q:
            r, c = q.popleft()
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and shape[nr, nc] == bg:
                    visited[nr, nc] = True
                    q.append((nr, nc))

        return bool(np.any(np.logical_and(shape == bg, ~visited)))
