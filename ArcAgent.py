import numpy as np
from collections import deque

from ArcProblem import ArcProblem


class ArcAgent:

    def __init__(self):
        pass


    # ============================================================
    # MAIN SOLVER
    # ============================================================

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

        # =====================================================
        # RULE IDENTIFICATION  (vote across all training pairs)
        # =====================================================

        rule_votes = {}

        for ex in arc_problem._training_data:
            in_g  = np.array(ex._input._arc_array)
            out_g = np.array(ex._output._arc_array)
            rule  = self.identify_rule(in_g, out_g)
            rule_votes[rule] = rule_votes.get(rule, 0) + 1

        identified_rule = max(rule_votes, key=rule_votes.get) if rule_votes else None
        print(f"Identified rule: {identified_rule}")

        # =====================================================
        # HARD-CODED RULE SOLVERS  (highest priority)
        # =====================================================

        RULE_SOLVERS = {
            "center_recolor":      self.solve_center_recolor,
            "diagonal_growth":     self.solve_diagonal_growth,
            "fill_inside_outside": self.solve_fill_inside_outside,
            "edge_matching":       self.solve_edge_matching,
            "direction_growth":    self.solve_direction_growth,
            "rotation_flip":       self.solve_rotation_flip,
            "closed_recolor":      self.solve_closed_recolor,
            "divider_fill":        self.solve_divider_fill,
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
            self.rotate90, self.rotate180, self.rotate270,
            self.flip_horizontal, self.flip_vertical,
            self.transpose,
            self.mirror_left_to_right, self.mirror_bottom_to_top,
            self.mirror_tile_2x2,
            self.mirror_tile_horizontal,
            self.mirror_tile_vertical,
            self.invert_majority_color,
            self.trim,
            self.diagonal_x_fill,
            self.hollow_objects,
            self.trim_and_flip_colors,
            self.swap_gray_and_color,
            self.color_count_columns,
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

        # =====================================================
        # GREEDY CHAIN SEARCH
        # =====================================================

        def score_chain(chain):
            total = 0
            for training in arc_problem._training_data:
                grid = np.array(training._input._arc_array, copy=True)
                for t in chain:
                    grid = t(grid)
                total += self.combined_score(training._output._arc_array, grid)
            return total / max(len(arc_problem._training_data), 1)

        best_chain       = []
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

        # =====================================================
        # SEPARATOR ANALYSIS
        # =====================================================

        panel_comparisons = self.compare_panels_to_output(arc_problem)

        if panel_comparisons:
            print(f"  Found separator in {len(panel_comparisons)} training example(s)")

        # =====================================================
        # OBJECT RECOLOR + SPATIAL CANDIDATES
        # =====================================================

        for t in self.object_based_candidates(arc_problem):
            score = self.score_transform_on_training(t, arc_problem)
            transform_scores.append((t, score))

        for t in self.object_spatial_candidates(arc_problem):
            score = self.score_transform_on_training(t, arc_problem)
            transform_scores.append((t, score))

        # =====================================================
        # PANEL COMBINATION CANDIDATES
        # =====================================================

        for t in self._build_panel_candidates(arc_problem):
            score = self.score_transform_on_training(t, arc_problem)
            transform_scores.append((t, score))
            print(f"  Panel transform {t.__name__}: {round(score, 4)}")

        # =====================================================
        # CLOSED OBJECT FILL CANDIDATES
        # =====================================================

        for t in self._build_closed_fill_candidates(arc_problem):
            score = self.score_transform_on_training(t, arc_problem)
            transform_scores.append((t, score))

        # =====================================================
        # INTERIOR MAJORITY FILL CANDIDATES
        # =====================================================

        for t in self._build_interior_majority_candidates(arc_problem):
            score = self.score_transform_on_training(t, arc_problem)
            transform_scores.append((t, score))

        # =====================================================
        # INTERIOR REFLECTION CANDIDATES
        # =====================================================

        for t in self._build_interior_reflection_candidates(arc_problem):
            score = self.score_transform_on_training(t, arc_problem)
            transform_scores.append((t, score))

        # =====================================================
        # DIAGONAL X FILL CANDIDATES
        # =====================================================

        for t in self._build_diagonal_x_candidates(arc_problem):
            score = self.score_transform_on_training(t, arc_problem)
            transform_scores.append((t, score))

        # =====================================================
        # SPIRAL FILL CANDIDATES
        # =====================================================

        for t in self._build_spiral_candidates(arc_problem):
            score = self.score_transform_on_training(t, arc_problem)
            transform_scores.append((t, score))

        # =====================================================
        # EXPAND DOTS TO 3×3 BLOCKS
        # =====================================================

        for t in self._build_expand_dots_candidates(arc_problem):
            score = self.score_transform_on_training(t, arc_problem)
            transform_scores.append((t, score))

        # =====================================================
        # COLOR COUNT COLUMNS
        # =====================================================

        score = self.score_transform_on_training(
            self.color_count_columns, arc_problem)
        transform_scores.append((self.color_count_columns, score))

        # =====================================================
        # MULTI-PANEL PRIORITY MERGE
        # =====================================================

        for t in self._build_multi_panel_candidates(arc_problem):
            score = self.score_transform_on_training(t, arc_problem)
            transform_scores.append((t, score))

        # =====================================================
        # HOLLOW + TRIM-FLIP + SWAP-GRAY (basic scored)
        # =====================================================

        for t in [self.hollow_objects,
                  self.trim_and_flip_colors,
                  self.swap_gray_and_color]:
            score = self.score_transform_on_training(t, arc_problem)
            transform_scores.append((t, score))

        # =====================================================
        # NOISE-STRIPPED VARIANTS OF CLOSE CANDIDATES
        # =====================================================

        for t in self._build_noise_stripped_candidates(arc_problem, transform_scores):
            score = self.score_transform_on_training(t, arc_problem)
            transform_scores.append((t, score))

        # =====================================================
        # RANK ALL CANDIDATES
        # =====================================================

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


    # ============================================================
    # RULE IDENTIFICATION
    # ============================================================

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

        # Rule 8 — separator divider (check BEFORE edge_matching to avoid
        # misidentifying the separator column as a border-cell pattern)
        if self.find_separator(input_grid) is not None:
            return "divider_fill"

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
        """
        For closed shapes: try ring-fill candidates first (scored against training).
        Falls back to simple wall recolor if nothing scores well.
        """
        if arc_problem is not None:
            candidates = self._build_closed_fill_candidates(arc_problem)
            if candidates:
                best_t, best_score = None, -1
                for t in candidates:
                    score = self.score_transform_on_training(t, arc_problem)
                    if score > best_score:
                        best_score, best_t = score, t
                if best_t is not None and best_score > 0:
                    return best_t(grid)

        # fallback: recolor closed shape walls only
        target_color = 8
        if arc_problem is not None:
            color_votes = {}
            for ex in arc_problem._training_data:
                in_g  = np.array(ex._input._arc_array)
                out_g = np.array(ex._output._arc_array)
                out_by_pos = {(o.min_row, o.min_col): o for o in self.find_objects(out_g)}
                for i_obj in self.find_objects(in_g):
                    if not i_obj.has_hole():
                        continue
                    o_obj = out_by_pos.get((i_obj.min_row, i_obj.min_col))
                    if o_obj is not None and o_obj.color != i_obj.color:
                        color_votes[o_obj.color] = color_votes.get(o_obj.color, 0) + 1
            if color_votes:
                target_color = max(color_votes, key=color_votes.get)

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


    # ============================================================
    # SEPARATOR DETECTION & PANEL SPLITTING
    # ============================================================

    def find_separator(self, grid):
        """
        Detect a full-width or full-height line of a single non-background color
        that is a minority/accent color (i.e. not the dominant foreground color).
        Returns a dict:
            {
                'axis':  'row' | 'col',
                'index': int,           # row or col index of the separator
                'color': int,           # separator color
            }
        or None if no separator is found.
        """

        h, w = grid.shape
        bg   = self.detect_background(grid)

        # dominant non-background color — we don't want to treat it as a separator
        vals, counts = np.unique(grid, return_counts=True)
        non_bg = [(v, c) for v, c in zip(vals, counts) if v != bg]
        dominant_fg = max(non_bg, key=lambda x: x[1])[0] if non_bg else None

        # check every column first (more common separator orientation)
        for c in range(w):
            col = grid[:, c]
            color = col[0]
            if color != bg and color != dominant_fg and np.all(col == color):
                return {'axis': 'col', 'index': c, 'color': int(color)}

        # check every row
        for r in range(h):
            row = grid[r, :]
            color = row[0]
            if color != bg and color != dominant_fg and np.all(row == color):
                return {'axis': 'row', 'index': r, 'color': int(color)}

        return None


    def split_by_separator(self, grid, separator):
        """
        Split a grid into two panels using a detected separator.
        Returns (panel_a, panel_b) — the two halves, not including the separator line.
        """

        idx = separator['index']

        if separator['axis'] == 'row':
            panel_a = grid[:idx, :]
            panel_b = grid[idx + 1:, :]
        else:
            panel_a = grid[:, :idx]
            panel_b = grid[:, idx + 1:]

        return panel_a, panel_b


    def split_by_halving(self, grid):
        """
        Split the grid into two equal halves along the longer axis.
        Returns (panel_a, panel_b, axis) where axis is 'row' or 'col'.
        If the grid is square, splits by columns (left/right).
        """
        h, w = grid.shape
        if h >= w:
            # taller — split top / bottom
            mid = h // 2
            return grid[:mid, :], grid[mid:, :], 'row'
        else:
            # wider — split left / right
            mid = w // 2
            return grid[:, :mid], grid[:, mid:], 'col'


    def describe_panels(self, grid):
        """
        High-level summary of the two panels in a separated grid.
        Returns a dict with everything downstream transformations need:
        {
            'separator':  {...},       # from find_separator
            'panel_a':    np.ndarray,  # first half
            'panel_b':    np.ndarray,  # second half
            'objects_a':  [ArcObject], # objects in panel_a
            'objects_b':  [ArcObject], # objects in panel_b
            'colors_a':   set,         # non-background colors in panel_a
            'colors_b':   set,         # non-background colors in panel_b
            'shared_colors': set,      # colors present in both panels
        }
        or None if no separator exists.
        """

        sep = self.find_separator(grid)
        bg  = self.detect_background(grid)

        if sep is not None:
            panel_a, panel_b = self.split_by_separator(grid, sep)
            # degenerate split — separator at edge, one panel empty
            if panel_a.size == 0 or panel_b.size == 0:
                sep = None

        if sep is None:
            # no separator — fall back to halving along the longer axis
            panel_a, panel_b, axis = self.split_by_halving(grid)
            sep = {'axis': axis, 'index': None, 'color': None, 'halved': True}

        objs_a = self.find_objects(panel_a)
        objs_b = self.find_objects(panel_b)

        colors_a = set(np.unique(panel_a)) - {bg}
        colors_b = set(np.unique(panel_b)) - {bg}

        return {
            'separator':     sep,
            'panel_a':       panel_a,
            'panel_b':       panel_b,
            'objects_a':     objs_a,
            'objects_b':     objs_b,
            'colors_a':      colors_a,
            'colors_b':      colors_b,
            'shared_colors': colors_a & colors_b,
        }


    def compare_panels_to_output(self, arc_problem):
        """
        For each training example that has a separator, describe both
        input panels and compare them to the output.  Returns a list of:
        {
            'input_panels':  describe_panels() result,
            'output':        np.ndarray,
            'output_objects': [ArcObject],
        }
        Only includes examples where a separator was found.
        """

        results = []

        for ex in arc_problem._training_data:

            in_g  = np.array(ex._input._arc_array)
            out_g = np.array(ex._output._arc_array)

            panels = self.describe_panels(in_g)

            if panels is None:
                continue

            results.append({
                'input_panels':   panels,
                'output':         out_g,
                'output_objects': self.find_objects(out_g),
            })

            # debug summary
            sep = panels['separator']
            print(f"  Separator: {sep['axis']} {sep['index']} "
                  f"color={sep['color']}")
            print(f"  Panel A colors: {panels['colors_a']}")
            print(f"  Panel B colors: {panels['colors_b']}")
            print(f"  Shared colors:  {panels['shared_colors']}")
            print(f"  Output shape:   {out_g.shape}")

        return results


    # ============================================================
    # PANEL COMBINATION TRANSFORMS  (overlap / XOR)
    # ============================================================

    def _align_panels(self, a, b):
        """
        Trim both panels to the same shape (min rows x min cols)
        so cell-wise operations are always valid.
        """
        h = min(a.shape[0], b.shape[0])
        w = min(a.shape[1], b.shape[1])
        return a[:h, :w], b[:h, :w]


    def panel_overlap(self, grid, fill_color=1):
        """
        OR — cell is filled (fill_color) if EITHER panel has a non-bg value there.
        fill_color defaults to 1; inferred from training by solve_divider_fill /
        the scored pipeline which picks the best version.
        """
        panels = self.describe_panels(grid)
        if panels is None:
            return grid

        bg = self.detect_background(grid)
        a, b = self._align_panels(panels['panel_a'], panels['panel_b'])

        filled = (a != bg) | (b != bg)
        out = np.where(filled, fill_color, bg)
        return out

    def panel_xor(self, grid, fill_color=1):
        """
        XOR — output cell is filled (fill_color) only when EXACTLY ONE panel
        has data there. Both filled or neither filled → background.
        """
        panels = self.describe_panels(grid)
        if panels is None:
            return grid

        bg = self.detect_background(grid)
        a, b = self._align_panels(panels['panel_a'], panels['panel_b'])

        a_filled = a != bg
        b_filled = b != bg

        exactly_one = a_filled ^ b_filled

        out = np.full_like(a, bg)
        out[exactly_one] = fill_color
        return out

    def panel_intersection(self, grid, fill_color=1):
        """
        AND — output cell is filled (fill_color) only when BOTH panels
        have data there.
        """
        panels = self.describe_panels(grid)
        if panels is None:
            return grid

        bg = self.detect_background(grid)
        a, b = self._align_panels(panels['panel_a'], panels['panel_b'])

        both_filled = (a != bg) & (b != bg)

        out = np.full_like(a, bg)
        out[both_filled] = fill_color
        return out


    def panel_neither(self, grid, fill_color=1):
        """
        NOR — output cell is filled (fill_color) only when NEITHER panel
        has data there (both are background).
        """
        panels = self.describe_panels(grid)
        if panels is None:
            return grid

        bg = self.detect_background(grid)
        a, b = self._align_panels(panels['panel_a'], panels['panel_b'])

        neither_filled = (a == bg) & (b == bg)

        out = np.full_like(a, bg)
        out[neither_filled] = fill_color
        return out

    def _build_panel_candidates(self, arc_problem):
        """
        Build panel combination transforms for every non-background color
        seen in training outputs.
        - If a separator exists: use separator-based split (tagged 'sep').
        - Always also try halving split (tagged 'halved') as a fallback.
        Scoring picks whichever wins.
        """

        has_sep = any(
            self.find_separator(np.array(ex._input._arc_array)) is not None
            for ex in arc_problem._training_data
        )

        # collect candidate fill colors from training outputs
        fill_colors = set()
        for ex in arc_problem._training_data:
            out_g = np.array(ex._output._arc_array)
            bg    = self.detect_background(out_g)
            for c in np.unique(out_g):
                if c != bg:
                    fill_colors.add(int(c))

        if not fill_colors:
            fill_colors = {1}

        candidates = []
        ops = [
            ("overlap",  self.panel_overlap),
            ("xor",      self.panel_xor),
            ("and",      self.panel_intersection),
            ("neither",  self.panel_neither),
        ]

        for color in sorted(fill_colors):
            for op_name, fn in ops:

                # separator-based (only when separator detected)
                if has_sep:
                    def make_sep(f, c, n):
                        def t(grid):
                            return f(grid, fill_color=c)
                        t.__name__ = f"sep_{n}_fill{c}"
                        return t
                    candidates.append(make_sep(fn, color, op_name))

                # halving-based (always — forced split along longer axis)
                def make_halved(f, c, n):
                    def t(grid):
                        h, w = grid.shape
                        bg   = self.detect_background(grid)
                        if h >= w:
                            mid = h // 2
                            panel_a, panel_b = grid[:mid, :], grid[mid:, :]
                        else:
                            mid = w // 2
                            panel_a, panel_b = grid[:, :mid], grid[:, mid:]
                        a, b = self._align_panels(panel_a, panel_b)
                        a_filled = a != bg
                        b_filled = b != bg
                        if n == "overlap":
                            mask = a_filled | b_filled
                        elif n == "xor":
                            mask = a_filled ^ b_filled
                        elif n == "neither":
                            mask = (~a_filled) & (~b_filled)
                        else:  # and
                            mask = a_filled & b_filled
                        out = np.full_like(a, bg)
                        out[mask] = c
                        return out
                    t.__name__ = f"halved_{n}_fill{c}"
                    return t
                candidates.append(make_halved(fn, color, op_name))

        return candidates

    # ============================================================
    # CLOSED OBJECT FILL  (interior + exterior lining)
    # ============================================================

    def _flood_fill_region(self, grid, seed_cells, passable_color, h, w):
        """
        BFS from seed_cells through cells equal to passable_color.
        Returns the set of (r,c) reached.
        """
        visited = set(seed_cells)
        queue   = deque(seed_cells)
        dirs    = [(-1,0),(1,0),(0,-1),(0,1)]
        while queue:
            r, c = queue.popleft()
            for dr, dc in dirs:
                nr, nc = r+dr, c+dc
                if (0 <= nr < h and 0 <= nc < w
                        and (nr,nc) not in visited
                        and grid[nr,nc] == passable_color):
                    visited.add((nr,nc))
                    queue.append((nr,nc))
        return visited


    def _get_interior_exterior(self, grid, obj, bg):
        """
        For a closed ArcObject, return:
          interior  — set of bg cells enclosed by the object (in global coords)
          exterior  — set of bg cells reachable from the grid border (global coords)

        Interior is computed locally within the object's bounding box to avoid
        leaking across other objects on the grid.
        Exterior is the complement: all bg cells not in the interior.
        """
        # --- interior: work in local bounding-box space (same as has_hole) ---
        shape  = obj.shape_matrix()
        lh, lw = shape.shape
        visited = np.zeros((lh, lw), dtype=bool)
        q       = deque()
        dirs    = [(-1,0),(1,0),(0,-1),(0,1)]

        # seed flood from edges of the bounding box
        for r in range(lh):
            for c in range(lw):
                if (r in [0, lh-1] or c in [0, lw-1]) and shape[r, c] == 0:
                    visited[r, c] = True
                    q.append((r, c))

        while q:
            r, c = q.popleft()
            for dr, dc in dirs:
                nr, nc = r+dr, c+dc
                if 0 <= nr < lh and 0 <= nc < lw and not visited[nr,nc] and shape[nr,nc] == 0:
                    visited[nr, nc] = True
                    q.append((nr, nc))

        # interior = local bg cells not reachable from bounding-box edge
        interior = {
            (r + obj.min_row, c + obj.min_col)
            for r in range(lh) for c in range(lw)
            if shape[r, c] == 0 and not visited[r, c]
        }

        # --- exterior: all bg cells on the full grid not in the interior ---
        obj_cells = set(obj.cells)
        h, w = grid.shape
        all_bg = {
            (r, c) for r in range(h) for c in range(w)
            if grid[r, c] == bg and (r, c) not in obj_cells and (r, c) not in interior
        }

        # flood from border to get truly reachable exterior
        border_seeds = [
            (r, c) for r in range(h) for c in [0, w-1]
            if grid[r,c] == bg and (r,c) not in obj_cells
        ] + [
            (r, c) for c in range(w) for r in [0, h-1]
            if grid[r,c] == bg and (r,c) not in obj_cells
        ]
        exterior = self._flood_fill_region(grid, border_seeds, bg, h, w) - interior

        return interior, exterior


    def _infer_fill_colors(self, arc_problem, bg):
        """
        Look at training pairs and infer:
          wall_color     — color the object walls become (or None = keep)
          interior_color — color filling the inside hole
          exterior_color — color filling the outside region
        Returns a list of (wall_color, interior_color, exterior_color) tuples
        seen across training, deduplicated.
        """
        combos = set()

        for ex in arc_problem._training_data:
            in_g  = np.array(ex._input._arc_array)
            out_g = np.array(ex._output._arc_array)
            in_bg = self.detect_background(in_g)

            for obj in self.find_objects(in_g):
                if not obj.has_hole():
                    continue

                interior, exterior = self._get_interior_exterior(in_g, obj, in_bg)

                # sample wall color from output at one of the obj's cells
                wr, wc = obj.cells[0]
                wall_c = int(out_g[wr, wc]) if out_g.shape == in_g.shape else None

                # sample interior color
                int_c = None
                for (r,c) in interior:
                    if r < out_g.shape[0] and c < out_g.shape[1]:
                        int_c = int(out_g[r,c])
                        break

                # sample exterior color
                ext_c = None
                for (r,c) in exterior:
                    if r < out_g.shape[0] and c < out_g.shape[1]:
                        v = int(out_g[r,c])
                        if v != in_bg:          # skip if exterior stays background
                            ext_c = v
                            break

                if int_c is not None:
                    combos.add((wall_c, int_c, ext_c))

        return list(combos) if combos else [(None, 1, None)]


    def _outer_ring(self, obj, grid, bg):
        """
        New bg cells added OUTSIDE the object: exterior bg cells
        that are 8-connected (including diagonal) neighbors of the object walls,
        fully encapsulating the object.
        """
        _, exterior = self._get_interior_exterior(grid, obj, bg)
        obj_set     = set(obj.cells)
        h, w        = grid.shape
        dirs        = [(-1,-1),(-1,0),(-1,1),
                       ( 0,-1),        ( 0,1),
                       ( 1,-1),( 1,0),( 1,1)]
        ring        = set()
        for r, c in obj.cells:
            for dr, dc in dirs:
                nr, nc = r+dr, c+dc
                if (0 <= nr < h and 0 <= nc < w
                        and (nr,nc) not in obj_set
                        and (nr,nc) in exterior):
                    ring.add((nr,nc))
        return ring


    def _inner_ring(self, obj, grid, bg):
        """
        New bg cells added INSIDE the object hole: interior bg cells
        that are 8-connected (including diagonal) neighbors of the object walls.
        """
        interior, _ = self._get_interior_exterior(grid, obj, bg)
        obj_set     = set(obj.cells)
        h, w        = grid.shape
        dirs        = [(-1,-1),(-1,0),(-1,1),
                       ( 0,-1),        ( 0,1),
                       ( 1,-1),( 1,0),( 1,1)]
        ring        = set()
        for r, c in obj.cells:
            for dr, dc in dirs:
                nr, nc = r+dr, c+dc
                if (0 <= nr < h and 0 <= nc < w
                        and (nr,nc) not in obj_set
                        and (nr,nc) in interior):
                    ring.add((nr,nc))
        return ring


    def _interior_majority_color(self, obj, grid, bg):
        """
        Return the most common non-background color found inside the
        object's hole, or None if the interior contains only background.
        """
        interior, _ = self._get_interior_exterior(grid, obj, bg)
        if not interior:
            return None

        color_counts = {}
        for r, c in interior:
            v = grid[r, c]
            if v != bg:
                color_counts[v] = color_counts.get(v, 0) + 1

        if not color_counts:
            return None

        return max(color_counts, key=color_counts.get)


    def _all_interior_cells(self, obj):
        """
        Return every cell inside the object's bounding box that is NOT
        a wall cell — i.e. all cells enclosed by the shape, regardless
        of what color they currently are (bg or otherwise).
        This is derived from shape_matrix so it never leaks to other objects.
        """
        shape    = obj.shape_matrix()
        lh, lw   = shape.shape
        visited  = np.zeros((lh, lw), dtype=bool)
        q        = deque()
        dirs     = [(-1,0),(1,0),(0,-1),(0,1)]

        # flood from bounding-box edges to find exterior-of-hole cells
        for r in range(lh):
            for c in range(lw):
                if (r in [0, lh-1] or c in [0, lw-1]) and shape[r,c] == 0:
                    visited[r,c] = True
                    q.append((r,c))

        while q:
            r, c = q.popleft()
            for dr, dc in dirs:
                nr, nc = r+dr, c+dc
                if 0<=nr<lh and 0<=nc<lw and not visited[nr,nc] and shape[nr,nc] == 0:
                    visited[nr,nc] = True
                    q.append((nr,nc))

        # every non-wall cell not reached = inside the hole
        return [
            (r + obj.min_row, c + obj.min_col)
            for r in range(lh) for c in range(lw)
            if not visited[r,c] and shape[r,c] == 0
        ]


    def solve_interior_majority_fill(self, grid, arc_problem=None):
        """
        For every closed object whose interior contains at least one
        non-background cell:
          - Find the majority non-background color inside the hole.
          - Overwrite EVERY interior cell (including other colors) with it.
        Objects whose interiors are all background are left untouched.
        Walls are NOT recolored — only the cells inside the hole.
        """
        bg  = self.detect_background(grid)
        out = grid.copy()

        for obj in self.find_objects(grid):
            if not obj.has_hole():
                continue

            fill_color = self._interior_majority_color(obj, grid, bg)
            if fill_color is None:
                continue

            # overwrite every cell inside the hole, whatever color it is
            for r, c in self._all_interior_cells(obj):
                out[r, c] = fill_color

        return out


    def _build_interior_majority_candidates(self, arc_problem):
        """
        Returns the interior-majority-fill transform if any training input
        has a closed object with non-background cells inside it.
        Also tries variants that additionally apply the outer/inner ring fill
        on top, scored against training to find the best combo.
        """
        has_candidate = False
        for ex in arc_problem._training_data:
            in_g = np.array(ex._input._arc_array)
            bg   = self.detect_background(in_g)
            for obj in self.find_objects(in_g):
                if obj.has_hole() and self._interior_majority_color(obj, in_g, bg) is not None:
                    has_candidate = True
                    break
            if has_candidate:
                break

        if not has_candidate:
            return []

        candidates = []

        # base: just fill the interior with majority color
        def majority_fill(grid):
            return self.solve_interior_majority_fill(grid, arc_problem)
        majority_fill.__name__ = "interior_majority_fill"
        candidates.append(majority_fill)

        # combos: majority fill + outer ring in each output color
        output_colors = set()
        for ex in arc_problem._training_data:
            out_g  = np.array(ex._output._arc_array)
            out_bg = self.detect_background(out_g)
            for v in np.unique(out_g):
                if v != out_bg:
                    output_colors.add(int(v))

        for oc in sorted(output_colors):
            def make_t(outer_c):
                def t(grid):
                    # first fill interior with majority color
                    g = self.solve_interior_majority_fill(grid, arc_problem)
                    # then add outer ring
                    bg2 = self.detect_background(grid)
                    for obj in self.find_objects(grid):
                        if obj.has_hole():
                            for r, c in self._outer_ring(obj, grid, bg2):
                                g[r, c] = outer_c
                    return g
                t.__name__ = f"interior_majority_fill_outer{outer_c}"
                return t
            candidates.append(make_t(oc))

        return candidates

    def solve_closed_object_fill(self, grid, arc_problem=None,
                                  outer_color=None,
                                  inner_color=None,
                                  wall_color=None):
        """
        For every closed object (has_hole), ADD new filled cells:
          outer_color — bg cells directly outside the object walls get this color
          inner_color — bg cells directly inside the hole (touching walls) get this color
          wall_color  — recolor the object walls themselves (None = keep original)
        Open objects are left untouched.
        Everything else (non-adjacent bg cells) stays unchanged.
        """
        bg  = self.detect_background(grid)
        out = grid.copy()

        for obj in self.find_objects(grid):
            if not obj.has_hole():
                continue

            if outer_color is not None:
                for r, c in self._outer_ring(obj, grid, bg):
                    out[r, c] = outer_color

            if inner_color is not None:
                for r, c in self._inner_ring(obj, grid, bg):
                    out[r, c] = inner_color

            if wall_color is not None:
                for r, c in obj.cells:
                    out[r, c] = wall_color

        return out


    def _build_closed_fill_candidates(self, arc_problem):
        """
        Build a scored candidate for each (wall, interior, exterior) combo
        inferred from training, plus a broad sweep of common color combos.
        Only active when closed shapes exist in training input.
        """

        has_closed = any(
            any(o.has_hole() for o in self.find_objects(np.array(ex._input._arc_array)))
            for ex in arc_problem._training_data
        )

        if not has_closed:
            return []

        bg = self.detect_background(
            np.array(arc_problem._training_data[0]._input._arc_array)
        )

        # infer from training
        inferred = self._infer_fill_colors(arc_problem, bg)

        # also collect all non-bg colors from training outputs as candidates
        output_colors = set()
        for ex in arc_problem._training_data:
            out_g = np.array(ex._output._arc_array)
            out_bg = self.detect_background(out_g)
            for v in np.unique(out_g):
                if v != out_bg:
                    output_colors.add(int(v))

        # build full combo list: inferred + sweeps
        combos = list(inferred)
        for ic in output_colors:
            combos.append((None, ic, None))           # interior only
            for ec in output_colors:
                if ec != ic:
                    combos.append((None, ic, ec))     # interior + exterior

        # deduplicate
        combos = list(dict.fromkeys(combos))

        candidates = []

        # generate all combinations of outer/inner/wall colors
        # None means "don't touch those cells"
        color_opts = [None] + sorted(output_colors)

        seen_names = set()
        for wall_c, int_c, ext_c in combos:
            # int_c → inner ring, ext_c → outer ring
            for outer_c in [ext_c, None]:
                for inner_c in [int_c, None]:
                    if outer_c is None and inner_c is None:
                        continue  # nothing to add
                    def make_t(oc, ic, wc):
                        def t(grid):
                            return self.solve_closed_object_fill(
                                grid, arc_problem,
                                outer_color=oc,
                                inner_color=ic,
                                wall_color=wc,
                            )
                        parts = []
                        if oc is not None: parts.append(f"outer{oc}")
                        if ic is not None: parts.append(f"inner{ic}")
                        if wc is not None: parts.append(f"wall{wc}")
                        t.__name__ = "closed_fill_" + "_".join(parts)
                        return t
                    name = f"{outer_c}_{inner_c}_{wall_c}"
                    if name not in seen_names:
                        seen_names.add(name)
                        candidates.append(make_t(outer_c, inner_c, wall_c))

        return candidates

    # ============================================================
    # OBJECT RECOLOR CANDIDATES  (scoring-based)
    # ============================================================

    def object_based_candidates(self, arc_problem):

        candidates = []

        for example in arc_problem._training_data:

            input_grid  = np.array(example._input._arc_array)
            output_grid = np.array(example._output._arc_array)
            in_objs     = self.find_objects(input_grid)
            out_objs    = self.find_objects(output_grid)

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


    # ============================================================
    # OBJECT SPATIAL CANDIDATES
    # ============================================================

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


    # ============================================================
    # INTERIOR REFLECTION
    # ============================================================

    def _reflect_interior(self, grid, obj, bg, mode="both"):
        """
        Reflect the non-background content inside a closed object's hole
        onto the opposite side of the interior.

        mode options:
          "horizontal" — flip left↔right inside the hole
          "vertical"   — flip top↔bottom inside the hole
          "both"       — rotate 180° inside the hole (flip both axes)

        Non-background interior cells are copied to their reflected position.
        The reflected position overwrites whatever was there.
        Wall cells are never touched.
        """
        all_interior = set(self._all_interior_cells(obj))
        if not all_interior:
            return grid

        rows = [r for r, c in all_interior]
        cols = [c for r, c in all_interior]
        min_r, max_r = min(rows), max(rows)
        min_c, max_c = min(cols), max(cols)

        out = grid.copy()

        for r, c in all_interior:
            v = grid[r, c]
            if v == bg:
                continue  # only reflect filled cells

            if mode == "horizontal":
                nr, nc = r, min_c + max_c - c
            elif mode == "vertical":
                nr, nc = min_r + max_r - r, c
            else:  # both / 180
                nr, nc = min_r + max_r - r, min_c + max_c - c

            if (nr, nc) in all_interior:
                out[nr, nc] = v

        return out


    def solve_interior_reflection(self, grid, mode="both"):
        """
        For every closed object, reflect the non-bg interior content
        across the hole's center using the given mode.
        """
        bg  = self.detect_background(grid)
        out = grid.copy()

        for obj in self.find_objects(grid):
            if not obj.has_hole():
                continue

            # check there's anything non-bg inside
            interior_cells = self._all_interior_cells(obj)
            if not any(grid[r, c] != bg for r, c in interior_cells):
                continue

            out = self._reflect_interior(out, obj, bg, mode=mode)

        return out


    def _build_interior_reflection_candidates(self, arc_problem):
        """
        Generate reflection candidates for each mode.
        Only active when at least one training input has a closed object
        with non-background content inside.
        """
        has_filled_interior = False
        for ex in arc_problem._training_data:
            in_g = np.array(ex._input._arc_array)
            bg   = self.detect_background(in_g)
            for obj in self.find_objects(in_g):
                if not obj.has_hole():
                    continue
                if any(in_g[r, c] != bg for r, c in self._all_interior_cells(obj)):
                    has_filled_interior = True
                    break
            if has_filled_interior:
                break

        if not has_filled_interior:
            return []

        candidates = []
        for mode in ["both", "horizontal", "vertical"]:
            def make_t(m):
                def t(grid):
                    return self.solve_interior_reflection(grid, mode=m)
                t.__name__ = f"interior_reflect_{m}"
                return t
            candidates.append(make_t(mode))

            # also try reflection THEN majority fill — covers cases where
            # the reflection completes a pattern that gets flood-filled
            def make_reflect_then_fill(m):
                def t(grid):
                    reflected = self.solve_interior_reflection(grid, mode=m)
                    return self.solve_interior_majority_fill(reflected)
                t.__name__ = f"interior_reflect_{m}_then_fill"
                return t
            candidates.append(make_reflect_then_fill(mode))

        return candidates

    # ============================================================
    # HOLLOW OBJECTS
    # ============================================================

    def hollow_objects(self, grid):
        """
        For every non-background object, keep only its outermost shell —
        any cell whose 4 in-bounds neighbors are all the same color becomes
        background (interior cleared).  Works on any solid filled shape.
        """
        bg  = self.detect_background(grid)
        out = grid.copy()
        h, w = grid.shape
        dirs4 = [(-1,0),(1,0),(0,-1),(0,1)]
        visited = np.zeros((h,w), dtype=bool)

        for r in range(h):
            for c in range(w):
                if visited[r,c] or grid[r,c] == bg:
                    continue
                color = grid[r,c]
                q = deque([(r,c)]); visited[r,c] = True; cells = []
                while q:
                    cr,cc = q.popleft(); cells.append((cr,cc))
                    for dr,dc in dirs4:
                        nr,nc = cr+dr,cc+dc
                        if (0<=nr<h and 0<=nc<w
                                and not visited[nr,nc]
                                and grid[nr,nc]==color):
                            visited[nr,nc]=True; q.append((nr,nc))

                cell_set = set(cells)
                for r2,c2 in cells:
                    # interior = every in-bounds 4-neighbor is also a wall cell
                    all_nbrs_filled = all(
                        grid[r2+dr,c2+dc] == color
                        for dr,dc in dirs4
                        if 0<=r2+dr<h and 0<=c2+dc<w
                    )
                    on_grid_edge = (r2==0 or r2==h-1 or c2==0 or c2==w-1)
                    if all_nbrs_filled and not on_grid_edge:
                        out[r2,c2] = bg

        return out


    # ============================================================
    # TRIM + COLOR FLIP
    # ============================================================

    def trim_and_flip_colors(self, grid):
        """
        Trim background border, then swap the two most common non-background
        colors — e.g. 3→8 and 8→3 inside the cropped region.
        """
        bg   = self.detect_background(grid)
        rows = np.any(grid != bg, axis=1)
        cols = np.any(grid != bg, axis=0)
        trimmed = grid[rows][:,cols]

        vals = [int(v) for v in np.unique(trimmed) if v != bg]
        if len(vals) < 2:
            return trimmed

        out = trimmed.copy()
        a, b = vals[0], vals[1]
        out[trimmed==a] = b
        out[trimmed==b] = a
        return out


    # ============================================================
    # EXPAND SINGLE CELLS TO 3×3 BLOCKS
    # ============================================================

    def expand_dots_to_blocks(self, grid):
        """
        Each isolated non-background cell becomes a 3×3 filled block
        centered on it.  Fill color is inferred as the most common
        non-background, non-source color in training outputs; defaults
        to 1 if none found.
        """
        bg        = self.detect_background(grid)
        src_color = None
        for v in np.unique(grid):
            if v != bg:
                src_color = int(v); break

        # fill color = 1 unless overridden by caller via _build_expand_dots_candidates
        fill_color = 1
        out = np.full_like(grid, bg)
        h, w = grid.shape

        for r in range(h):
            for c in range(w):
                if grid[r,c] != bg:
                    for dr in range(-1,2):
                        for dc in range(-1,2):
                            nr,nc = r+dr,c+dc
                            if 0<=nr<h and 0<=nc<w:
                                out[nr,nc] = fill_color
        return out


    def _build_expand_dots_candidates(self, arc_problem):
        """
        Try expand_dots_to_blocks with every non-background output color.
        Only activates when inputs contain isolated single cells.
        """
        has_singles = any(
            any(o.size==1 for o in self.find_objects(np.array(ex._input._arc_array)))
            for ex in arc_problem._training_data
        )
        if not has_singles:
            return []

        fill_colors = set()
        for ex in arc_problem._training_data:
            out_g = np.array(ex._output._arc_array)
            bg    = self.detect_background(out_g)
            for v in np.unique(out_g):
                if v != bg: fill_colors.add(int(v))

        if not fill_colors: fill_colors = {1}

        candidates = []
        for fc in sorted(fill_colors):
            def make_t(c):
                def t(grid):
                    bg2 = self.detect_background(grid)
                    out = np.full_like(grid, bg2)
                    h, w = grid.shape
                    for r in range(h):
                        for cc in range(w):
                            if grid[r,cc] != bg2:
                                for dr in range(-1,2):
                                    for dc in range(-1,2):
                                        nr,nc = r+dr,cc+dc
                                        if 0<=nr<h and 0<=nc<w:
                                            out[nr,nc] = c
                    return out
                t.__name__ = f"expand_dots_fill{c}"
                return t
            candidates.append(make_t(fc))
        return candidates


    # ============================================================
    # SWAP GRAY (5) AND COLOR
    # ============================================================

    def swap_gray_and_color(self, grid):
        """
        Cells with value 5 (gray) → become the dominant non-gray color.
        Cells with the dominant non-gray color → become 0 (black/background).
        5 is always treated as gray regardless of frequency.
        """
        gray = 5
        # dominant non-gray color = most frequent value that isn't 5
        vals, counts = np.unique(grid, return_counts=True)
        non_gray = [(int(v), int(c)) for v,c in zip(vals,counts) if v != gray]
        if not non_gray:
            return grid.copy()
        color = max(non_gray, key=lambda x: x[1])[0]

        out = np.zeros_like(grid)          # black background
        out[grid==gray] = color            # gray → color
        # colored cells → 0 (already 0 from np.zeros_like)
        return out


    # ============================================================
    # COLOR COUNT COLUMNS
    # ============================================================

    def color_count_columns(self, grid):
        """
        Count occurrences of each non-background color.
        Output: one column per color, sorted descending by count.
        Each column is filled top-down with that color for `count` rows,
        then 0 (black) for the remaining rows.
        Height = count of most frequent color.
        Width  = number of distinct non-bg colors.
        """
        # always use 0 as background — output empty cells are always 0
        bg = 0
        vals, counts = np.unique(grid, return_counts=True)
        color_counts = sorted(
            [(int(v), int(c)) for v,c in zip(vals,counts) if v != bg],
            key=lambda x: -x[1]
        )
        if not color_counts:
            return grid.copy()

        max_count = color_counts[0][1]
        n_cols    = len(color_counts)
        out       = np.zeros((max_count, n_cols), dtype=grid.dtype)

        for col_idx, (color, count) in enumerate(color_counts):
            out[:count, col_idx] = color

        return out


    # ============================================================
    # MULTI-PANEL DETECTION & PRIORITY MERGE
    # ============================================================

    def find_all_separators(self, grid):
        """
        Find ALL full-span uniform non-background, non-dominant column
        separators, or row separators if no column separators exist.
        The dominant foreground color is excluded to avoid mistaking
        a panel column that happens to be all one color for a separator.
        Returns list of {'axis','index','color'} dicts.
        """
        bg   = self.detect_background(grid)
        h, w = grid.shape
        seps = []

        # find all full-height uniform non-bg column candidates
        candidates = []
        for c in range(w):
            col   = grid[:, c]
            color = int(col[0])
            if color != bg and np.all(col == color):
                candidates.append({'axis': 'col', 'index': c, 'color': color})

        if candidates:
            # pick the separator color: prefer the color appearing most times
            # as a full-column (e.g. 2 occurrences of color-2 beats 1 of color-4)
            from collections import Counter
            color_freq = Counter(c['color'] for c in candidates)
            # tie-break: least frequent overall in the grid
            vals2, counts2 = np.unique(grid, return_counts=True)
            grid_freq = dict(zip(vals2.tolist(), counts2.tolist()))
            best_color = max(color_freq, key=lambda c: (color_freq[c], -grid_freq.get(c, 0)))
            seps = [c for c in candidates if c['color'] == best_color]

        # rows only if no column seps found
        if not seps:
            row_candidates = []
            for r in range(h):
                row   = grid[r, :]
                color = int(row[0])
                if color != bg and np.all(row == color):
                    row_candidates.append({'axis': 'row', 'index': r, 'color': color})
            if row_candidates:
                from collections import Counter
                color_freq = Counter(c['color'] for c in row_candidates)
                vals2, counts2 = np.unique(grid, return_counts=True)
                grid_freq = dict(zip(vals2.tolist(), counts2.tolist()))
                best_color = max(color_freq, key=lambda c: (color_freq[c], -grid_freq.get(c, 0)))
                seps = [c for c in row_candidates if c['color'] == best_color]

        return seps


    def split_into_all_panels(self, grid):
        """
        Split grid into all panels separated by uniform separator lines.
        Returns (axis, [panel_array, ...]) or (None, []) if none found.
        """
        seps = self.find_all_separators(grid)
        if not seps:
            return None, []

        bg   = self.detect_background(grid)
        axis = seps[0]['axis']
        indices = sorted([s['index'] for s in seps if s['axis'] == axis])

        panels, prev = [], 0
        for idx in indices:
            p = grid[:, prev:idx] if axis == 'col' else grid[prev:idx, :]
            if p.size > 0:
                panels.append(p)
            prev = idx + 1

        p = grid[:, prev:] if axis == 'col' else grid[prev:, :]
        if p.size > 0:
            panels.append(p)

        return axis, panels


    def panel_priority_merge(self, grid):
        """
        Split into 2 or 3 panels, then merge using priority:
        start with panel 1 as the base output; for each subsequent panel,
        copy its non-background cells into any cell that is still background
        in the output.  Panels are ordered left-to-right (or top-to-bottom).
        """
        bg     = self.detect_background(grid)
        axis, panels = self.split_into_all_panels(grid)

        if not panels or len(panels) < 2:
            return grid

        min_h = min(p.shape[0] for p in panels)
        min_w = min(p.shape[1] for p in panels)
        panels = [p[:min_h, :min_w] for p in panels]

        out = panels[0].copy()
        for panel in panels[1:]:
            mask      = (out == bg) & (panel != bg)
            out[mask] = panel[mask]

        return out


    def _build_multi_panel_candidates(self, arc_problem):
        """
        Build panel_priority_merge candidates when 2+ separators are detected.
        Also adds the existing 2-panel ops (overlap/xor/and/neither) applied
        to the multi-panel split so all are scored together.
        """
        has_multi = any(
            len(self.find_all_separators(np.array(ex._input._arc_array))) >= 2
            for ex in arc_problem._training_data
        )
        has_any = any(
            len(self.find_all_separators(np.array(ex._input._arc_array))) >= 1
            for ex in arc_problem._training_data
        )

        if not has_any:
            return []

        candidates = []

        # priority merge (works for 2 or 3 panels)
        def priority_merge(grid):
            return self.panel_priority_merge(grid)
        priority_merge.__name__ = "panel_priority_merge"
        candidates.append(priority_merge)

        # reverse priority (start from last panel)
        def priority_merge_rev(grid):
            bg = self.detect_background(grid)
            _, panels = self.split_into_all_panels(grid)
            if not panels or len(panels) < 2:
                return grid
            min_h = min(p.shape[0] for p in panels)
            min_w = min(p.shape[1] for p in panels)
            panels = [p[:min_h,:min_w] for p in panels]
            panels = panels[::-1]
            out = panels[0].copy()
            for panel in panels[1:]:
                mask = (out == bg) & (panel != bg)
                out[mask] = panel[mask]
            return out
        priority_merge_rev.__name__ = "panel_priority_merge_reversed"
        candidates.append(priority_merge_rev)

        return candidates

    # ============================================================
    # SCORING HELPERS
    # ============================================================

    def _strip_single_pixels(self, grid):
        """
        Remove isolated single-pixel objects (size == 1) from the grid,
        replacing them with background. Open multi-cell objects are kept.
        Closed objects are kept.
        """
        bg   = self.detect_background(grid)
        out  = grid.copy()
        objs = self.find_objects(grid)
        for o in objs:
            if o.size == 1:
                r, c = o.cells[0]
                out[r, c] = bg
        return out


    def _with_noise_strip(self, base_transform, arc_problem, threshold=0.6):
        """
        Returns a wrapped transform that:
          1. Applies base_transform to get result A.
          2. Strips isolated single pixels from A to get result B.
          3. Scores A and B against training.
          4. Returns whichever scores higher.

        Only activates if the base score exceeds `threshold` — below that
        the transform isn't close enough for noise-stripping to be meaningful.
        """
        def wrapped(grid):
            result_a = base_transform(grid)
            result_b = self._strip_single_pixels(result_a)

            # score both against training
            score_a, score_b = 0.0, 0.0
            n = max(len(arc_problem._training_data), 1)

            for ex in arc_problem._training_data:
                in_g  = np.array(ex._input._arc_array, copy=True)
                out_g = ex._output._arc_array
                try:
                    pred_a = base_transform(in_g)
                    score_a += self.combined_score(out_g, pred_a)
                except Exception:
                    pass
                try:
                    pred_b = self._strip_single_pixels(pred_a if 'pred_a' in dir() else in_g)
                    score_b += self.combined_score(out_g, pred_b)
                except Exception:
                    pass

            score_a /= n
            score_b /= n

            # only bother if the base is reasonably close
            if score_a < threshold:
                return result_a

            return result_b if score_b > score_a else result_a

        wrapped.__name__ = f"{base_transform.__name__}_stripped"
        return wrapped


    def _build_noise_stripped_candidates(self, arc_problem, transform_scores, threshold=0.6):
        """
        For every existing candidate that scores above `threshold`,
        generate a noise-stripped variant and add it to the pool.
        """
        candidates = []
        for transform, score in transform_scores:
            if score >= threshold and not transform.__name__.endswith("_stripped"):
                wrapped = self._with_noise_strip(transform, arc_problem, threshold)
                candidates.append(wrapped)
        return candidates

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


    # ============================================================
    # GRID TRANSFORMS
    # ============================================================

    def rotate90(self, g):              return np.rot90(g, 1)
    def rotate180(self, g):             return np.rot90(g, 2)
    def rotate270(self, g):             return np.rot90(g, 3)
    def flip_horizontal(self, g):       return np.fliplr(g)
    def flip_vertical(self, g):         return np.flipud(g)
    def transpose(self, g):             return g.T

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

    def diagonal_x_fill(self, g):
        """
        For every non-background cell, draw diagonals in all 4 directions
        (↖ ↗ ↙ ↘) until the grid edge, painting with that cell's color.
        Existing non-background cells are never overwritten.
        """
        bg  = self.detect_background(g)
        out = g.copy()
        h, w = g.shape
        dirs = [(-1,-1), (-1,1), (1,-1), (1,1)]
        for r in range(h):
            for c in range(w):
                if g[r, c] == bg:
                    continue
                color = g[r, c]
                for dr, dc in dirs:
                    nr, nc = r + dr, c + dc
                    while 0 <= nr < h and 0 <= nc < w:
                        if out[nr, nc] == bg:
                            out[nr, nc] = color
                        nr += dr
                        nc += dc
        return out


    def _spiral_path(self, h, w):
        """
        Generate the full clockwise inward spiral path for an h×w grid,
        starting at (0,0) going right.  Returns list of (r,c) in order.
        """
        visited = np.zeros((h, w), dtype=bool)
        path    = []
        r, c    = 0, 0
        dirs    = [(0,1),(1,0),(0,-1),(-1,0)]
        d       = 0
        for _ in range(h * w):
            path.append((r, c))
            visited[r, c] = True
            nr, nc = r + dirs[d][0], c + dirs[d][1]
            if not (0 <= nr < h and 0 <= nc < w) or visited[nr, nc]:
                d      = (d + 1) % 4
                nr, nc = r + dirs[d][0], c + dirs[d][1]
            if not (0 <= nr < h and 0 <= nc < w) or visited[nr, nc]:
                break
            r, c = nr, nc
        return path


    def spiral_fill(self, g, color):
        """
        Fill the grid with concentric rectangular rings of `color` separated
        by single-cell gaps, following the spiral path from the top-left corner.

        The spiral path is divided into arms (each straight run before a turn).
        Every 4 arms = 1 ring.  Even-numbered rings (0, 2, 4 ...) are filled;
        odd-numbered rings are left as background (the gap rows/cols).
        This exactly matches the ARC spiral pattern where the outermost border
        is filled, then a 1-cell gap, then the next border, etc.
        """
        h, w  = g.shape
        path  = self._spiral_path(h, w)
        out   = g.copy()

        if len(path) < 2:
            return out

        # find arm boundaries (indices where direction changes)
        arm_starts = [0]
        prev_d = (path[1][0] - path[0][0], path[1][1] - path[0][1])
        for i in range(1, len(path) - 1):
            d = (path[i+1][0] - path[i][0], path[i+1][1] - path[i][1])
            if d != prev_d:
                arm_starts.append(i)
                prev_d = d
        arm_starts.append(len(path))

        # fill even rings (groups of 4 arms), skip odd rings
        for arm_idx in range(len(arm_starts) - 1):
            ring = arm_idx // 4
            if ring % 2 == 0:
                for i in range(arm_starts[arm_idx], arm_starts[arm_idx + 1]):
                    r, c = path[i]
                    out[r, c] = color

        return out


    def _build_diagonal_x_candidates(self, arc_problem):
        """
        Score diagonal-X fill for every non-background color found in
        training outputs, plus a version that uses each cell's own color.
        """
        candidates = []

        # variant 1: keep each source cell's own color along its diagonals
        def diag_own_color(g):
            return self.diagonal_x_fill(g)
        diag_own_color.__name__ = "diagonal_x_own_color"
        candidates.append(diag_own_color)

        # variant 2+: flood all diagonals with a single inferred output color
        fill_colors = set()
        for ex in arc_problem._training_data:
            out_g = np.array(ex._output._arc_array)
            bg    = self.detect_background(out_g)
            for v in np.unique(out_g):
                if v != bg:
                    fill_colors.add(int(v))

        for fc in sorted(fill_colors):
            def make_t(c):
                def t(g):
                    bg2 = self.detect_background(g)
                    out = g.copy()
                    h, w = g.shape
                    for r in range(h):
                        for c2 in range(w):
                            if g[r, c2] == bg2:
                                continue
                            for dr, dc in [(-1,-1),(-1,1),(1,-1),(1,1)]:
                                nr, nc = r+dr, c2+dc
                                while 0<=nr<h and 0<=nc<w:
                                    if out[nr,nc] == bg2:
                                        out[nr,nc] = c
                                    nr+=dr; nc+=dc
                    return out
                t.__name__ = f"diagonal_x_fill{c}"
                return t
            candidates.append(make_t(fc))

        return candidates


    def _build_spiral_candidates(self, arc_problem):
        """
        Score spiral fill for every non-background color found in
        training outputs.
        """
        fill_colors = set()
        for ex in arc_problem._training_data:
            out_g = np.array(ex._output._arc_array)
            bg    = self.detect_background(out_g)
            for v in np.unique(out_g):
                if v != bg:
                    fill_colors.add(int(v))

        if not fill_colors:
            fill_colors = {1}

        candidates = []
        for fc in sorted(fill_colors):
            def make_t(c):
                def t(g):
                    return self.spiral_fill(g, c)
                t.__name__ = f"spiral_fill{c}"
                return t
            candidates.append(make_t(fc))

        return candidates


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


    # ============================================================
    # OBJECT DETECTION
    # ============================================================

    def detect_background(self, grid):
        """
        Return the background color.
        0 is always treated as background if it appears in the grid.
        Otherwise fall back to the most frequent color.
        """
        if grid.size == 0:
            return 0
        if np.any(grid == 0):
            return 0
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


# ============================================================
# ARC OBJECT
# ============================================================

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
