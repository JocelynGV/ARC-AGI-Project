import numpy as np
from collections import deque

from ArcProblem import ArcProblem
from ArcData import ArcData
from ArcSet import ArcSet


class ArcAgent:
    def __init__(self):
        pass

    def make_predictions(self, arc_problem: ArcProblem) -> list[np.ndarray]:
        print(f"Making predictions for problem: {arc_problem._id}")
        predictions = [None, None, None]

        for example in arc_problem._training_data:
            input_grid = np.array(example._input._arc_array)

            objects = self.find_objects(input_grid)

            # example reasoning:
            largest = max(objects, key=lambda o: o.size)
            for o in objects:
                print(f"Object color: {o.color}, size: {o.size}, bounding box: {o.bounding_box()}")
            

            # print(o.color, o.size, o.bounding_box())

        basic_transformations = [
            self.rotate90,
            self.rotate180,
            self.rotate270,
            self.flip_horizontal,
            self.flip_vertical,
            self.transpose,
            self.mirror_left_to_right,
            self.mirror_bottom_to_top,
            self.invert_majority_color,
            self.trim,
        ]

        # Score each basic transformation against training data
        training_transformations = []
        for transformation in basic_transformations:
            average_score = 0
            for training in arc_problem._training_data:
                input_data = training._input._arc_array
                output_data = training._output._arc_array
                transformed = transformation(input_data)
                score = self.combined_score(output_data, transformed)
                average_score += score
            average_score = average_score / len(arc_problem._training_data)
            training_transformations.append((transformation, average_score))

        # Score flip_colors transformation
        average_score = 0
        valid_color_flips = 0
        colors = []
        for training in arc_problem._training_data:
            input_data = training._input._arc_array
            output_data = training._output._arc_array
            get_first_two_colors = self.top_two_colors(input_data)
            if len(get_first_two_colors) < 2:
                continue
            colors = get_first_two_colors
            transformed = self.flip_colors(input_data, get_first_two_colors[0], get_first_two_colors[1])
            score = self.combined_score(output_data, transformed)
            average_score += score
            valid_color_flips += 1

        if valid_color_flips > 0 and len(colors) == 2:
            average_score = average_score / valid_color_flips
            def make_flip_colors_fn():
                def flip_colors_fn(grid):
                    dynamic_colors = self.top_two_colors(grid)
                    if len(dynamic_colors) < 2:
                        return grid
                    return self.flip_colors(grid, dynamic_colors[0], dynamic_colors[1])
                flip_colors_fn.__name__ = "flip_colors"
                return flip_colors_fn
            training_transformations.append((make_flip_colors_fn(), average_score))

        training_transformations.sort(key=lambda x: x[1], reverse=True)

        print("All transformations ranked:")
        for transform, score in training_transformations:
            print(f"  {transform.__name__}: {score:.4f}")

        # Greedy chain search
        def score_chain(chain):
            total = 0
            for training in arc_problem._training_data:
                result = training._input._arc_array.copy()
                for transform in chain:
                    result = transform(result)
                total += self.combined_score(training._output._arc_array, result)
            return total / len(arc_problem._training_data)

        current_chain = []
        current_score = score_chain([])
        print(f"Baseline score (no transforms): {current_score:.4f}")

        for step in range(5):
            best_next_transform = None
            best_next_score = current_score

            for transform, _ in training_transformations:
                candidate_chain = current_chain + [transform]
                candidate_score = score_chain(candidate_chain)

                if candidate_score > best_next_score:
                    best_next_score = candidate_score
                    best_next_transform = transform

            if best_next_transform is None:
                print(f"Greedy search stopped at step {step} — no improvement found.")
                break

            current_chain.append(best_next_transform)
            current_score = best_next_score
            print(f"Step {step + 1}: added {best_next_transform.__name__}, score={current_score:.4f}")

        print(f"Final chain: {[t.__name__ for t in current_chain]}")

        # Apply greedy chain to test input
        greedy_result = arc_problem._test._input._arc_array.copy()
        for transform in current_chain:
            greedy_result = transform(greedy_result)
            print(f"After {transform.__name__}:\n{greedy_result}")

        predictions[0] = np.array(greedy_result, copy=True)

        predictions[1] = np.array(
            training_transformations[0][0](arc_problem._test._input._arc_array.copy()), copy=True
        )

        predictions[2] = np.array(
            training_transformations[1][0](arc_problem._test._input._arc_array.copy()), copy=True
        ) if len(training_transformations) > 1 else np.array(greedy_result, copy=True)

        print(f"Total predictions: {len(predictions)}")
        for i, p in enumerate(predictions):
            print(f"Prediction {i}:\n{p}")

        return predictions


    @staticmethod
    def rotate90(grid):
        return np.rot90(grid, 1)

    @staticmethod
    def rotate180(grid):
        return np.rot90(grid, 2)

    @staticmethod
    def rotate270(grid):
        return np.rot90(grid, 3)

    @staticmethod
    def flip_horizontal(grid):
        return np.fliplr(grid)

    @staticmethod
    def flip_vertical(grid):
        return np.flipud(grid)

    @staticmethod
    def transpose(grid):
        return grid.T
    
    @staticmethod
    def mirror_bottom_to_top(grid):
        new_grid = grid.copy()
        
        h = grid.shape[0]
        
        for i in range(h // 2):
            new_grid[i] = grid[h - 1 - i]
        
        return new_grid


    @staticmethod
    def mirror_left_to_right(grid):
        new_grid = grid.copy()
        
        w = grid.shape[1]  # number of columns
        
        for j in range(w // 2):
            new_grid[:, j] = grid[:, w - 1 - j]
        
        return new_grid



    @staticmethod
    def trim(grid, background=0):
        rows = np.any(grid != background, axis=1)
        cols = np.any(grid != background, axis=0)
        if not rows.any() or not cols.any():
            return grid
        return grid[rows][:, cols]

    @staticmethod
    def recolor(grid, old_color, new_color):
        new_grid = grid.copy()
        new_grid[new_grid == old_color] = new_color
        return new_grid

    @staticmethod
    def flip_colors(grid, color1, color2):
        new_grid = grid.copy()
        temp = -1
        new_grid[new_grid == color1] = temp
        new_grid[new_grid == color2] = color1
        new_grid[new_grid == temp] = color2
        return new_grid

    @staticmethod
    def top_two_colors(grid):
        flat = grid.flatten()
        non_zero = flat[flat != 0]
        if len(non_zero) == 0:
            return []
        values, counts = np.unique(non_zero, return_counts=True)
        sorted_indices = np.argsort(-counts)
        top_colors = values[sorted_indices][:2]
        return top_colors.tolist()
    
    @staticmethod
    def invert_majority_color(grid):
        values, counts = np.unique(grid, return_counts=True)

        # Remove background 0
        color_counts = [(v, c) for v, c in zip(values, counts) if v != 0]

        if len(color_counts) == 0:
            return grid.copy()

        # Find majority color
        majority_color = max(color_counts, key=lambda x: x[1])[0]

        new_grid = grid.copy()

        # Step 1: turn majority color into 0
        new_grid[new_grid == majority_color] = 0

        # Step 2: turn everything else (nonzero) into majority color
        new_grid[(new_grid != 0)] = majority_color

        return new_grid


    def get_percentage_accuracy(self, training_output, prediction):
        if training_output.shape != prediction.shape:
            return 0.0
        return (training_output == prediction).mean()

    def color_content_score(self, training_output, prediction):
        true_vals, true_counts = np.unique(training_output, return_counts=True)
        pred_vals, pred_counts = np.unique(prediction, return_counts=True)

        true_dict = dict(zip(true_vals, true_counts))
        pred_dict = dict(zip(pred_vals, pred_counts))

        all_colors = set(true_dict.keys()) | set(pred_dict.keys())

        total_diff = 0
        total_pixels = training_output.size

        for color in all_colors:
            total_diff += abs(true_dict.get(color, 0) - pred_dict.get(color, 0))

        return 1 - (total_diff / (2 * total_pixels))

    def combined_score(self, training_output, prediction):
        return 0.8 * self.get_percentage_accuracy(training_output, prediction) + \
               0.2 * self.color_content_score(training_output, prediction)
    
    
    def find_objects(self,grid):

        rows, cols = grid.shape

        visited = np.zeros_like(grid, dtype=bool)

        objects = []

        directions = [

            (-1,0),(1,0),
            (0,-1),(0,1)
        ]

        for r in range(rows):

            for c in range(cols):

                if visited[r,c]:

                    continue

                color = grid[r,c]

                if color == 0:

                    continue

                queue = deque([(r,c)])

                visited[r,c] = True

                cells = []

                while queue:

                    cr, cc = queue.popleft()

                    cells.append((cr,cc))

                    for dr,dc in directions:

                        nr = cr + dr
                        nc = cc + dc

                        if 0 <= nr < rows and 0 <= nc < cols:

                            if not visited[nr,nc] and grid[nr,nc] == color:

                                visited[nr,nc] = True
                                queue.append((nr,nc))

                objects.append(

                    ArcObject(color, cells, grid.shape)

                )

        return objects


class ArcObject:

    def __init__(self, color, cells, grid_shape):

        self.color = color
        self.cells = cells
        self.size = len(cells)
        self.grid_shape = grid_shape

        rows = [r for r,c in cells]
        cols = [c for r,c in cells]

        self.min_row = min(rows)
        self.max_row = max(rows)
        self.min_col = min(cols)
        self.max_col = max(cols)

        self.height = self.max_row - self.min_row + 1
        self.width = self.max_col - self.min_col + 1

    # -------------------------
    # basic spatial properties
    # -------------------------

    def bounding_box(self):

        return (
            self.min_row,
            self.min_col,
            self.max_row,
            self.max_col
        )

    def center(self):

        return (
            (self.min_row + self.max_row) // 2,
            (self.min_col + self.max_col) // 2
        )

    def shape_matrix(self):

        grid = np.zeros(
            (self.height, self.width),
            dtype=int
        )

        for r,c in self.cells:

            grid[
                r-self.min_row,
                c-self.min_col
            ] = self.color

        return grid

    def density(self):

        return self.size / (self.height * self.width)

    # -------------------------
    # shape classification
    # -------------------------

    def is_single_pixel(self):

        return self.size == 1

    def is_line(self):

        return self.height == 1 or self.width == 1

    def is_rectangle(self):

        return self.size == self.height * self.width

    def is_square(self):

        return (
            self.height == self.width
            and self.is_rectangle()
        )

    # -------------------------
    # orientation
    # -------------------------

    def orientation(self):

        if self.is_line():

            if self.height == 1:
                return "horizontal"

            if self.width == 1:
                return "vertical"

        if self.height > self.width:
            return "vertical-ish"

        if self.width > self.height:
            return "horizontal-ish"

        return "balanced"

    # -------------------------
    # symmetry
    # -------------------------

    def is_vertically_symmetric(self):

        shape = self.shape_matrix()

        return np.array_equal(
            shape,
            np.fliplr(shape)
        )

    def is_horizontally_symmetric(self):

        shape = self.shape_matrix()

        return np.array_equal(
            shape,
            np.flipud(shape)
        )

    # -------------------------
    # border relationships
    # -------------------------

    def touches_border(self):

        rows, cols = self.grid_shape

        for r,c in self.cells:

            if r == 0 or r == rows-1:
                return True

            if c == 0 or c == cols-1:
                return True

        return False

    # -------------------------
    # closure detection
    # -------------------------

    def is_closed_shape(self):

        shape = self.shape_matrix()

        h, w = shape.shape

        visited = np.zeros_like(shape, dtype=bool)

        queue = deque()

        # start flood fill from bounding box edges
        for r in range(h):

            for c in range(w):

                if r in [0,h-1] or c in [0,w-1]:

                    if shape[r,c] == 0:

                        queue.append((r,c))
                        visited[r,c] = True

        directions = [

            (-1,0),(1,0),
            (0,-1),(0,1)
        ]

        while queue:

            r,c = queue.popleft()

            for dr,dc in directions:

                nr = r + dr
                nc = c + dc

                if 0 <= nr < h and 0 <= nc < w:

                    if shape[nr,nc] == 0 and not visited[nr,nc]:

                        visited[nr,nc] = True
                        queue.append((nr,nc))

        # if empty space exists that was NOT reached,
        # shape encloses an area

        enclosed_spaces = np.logical_and(

            shape == 0,
            visited == False
        )

        return np.any(enclosed_spaces)

    # -------------------------
    # direction estimation
    # -------------------------

    def direction_vector(self):

        shape = self.shape_matrix()

        top = np.sum(shape[0,:] > 0)
        bottom = np.sum(shape[-1,:] > 0)

        left = np.sum(shape[:,0] > 0)
        right = np.sum(shape[:,-1] > 0)

        vertical_bias = bottom - top
        horizontal_bias = right - left

        return (vertical_bias, horizontal_bias)

    # -------------------------
    # comparison helpers
    # -------------------------

    def same_shape(self, other):

        return np.array_equal(

            self.shape_matrix() > 0,
            other.shape_matrix() > 0

        )

    def same_size(self, other):

        return self.size == other.size

    def same_color(self, other):

        return self.color == other.color