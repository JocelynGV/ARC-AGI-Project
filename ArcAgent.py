import numpy as np

from ArcProblem import ArcProblem
from ArcData import ArcData
from ArcSet import ArcSet


class ArcAgent:
    def __init__(self):
        """
        You may add additional variables to this init method. Be aware that it gets called only once
        and then the make_predictions method will get called several times.
        """
        pass

    def make_predictions(self, arc_problem: ArcProblem) -> list[np.ndarray]:
        
        test_array = np.array([
            [3,3,3,0],
            [3,0,3,0],
            [0,3,0,3],
            [3,3,3,0]
        ])

        test_array2 = np.array([
            [0, 0, 0],
            [2, 0, 0],
            [0, 0, 2]
        ])

        test_array3 = np.array([
            [0, 0, 0],
            [0, 2, 0],
            [0, 0, 0]
        ])

        print(arc_problem._id)
        for trn in arc_problem._training_data:
            # print(trn)
            # print("input:")
            # print(trn._input._arc_array)
            # print("output:")
            # print(trn._output._arc_array)
            print("Combined score for test array:", self.combined_score(trn._output._arc_array, test_array3))


        """
        Write the code in this method to solve the incoming ArcProblem.
        Your agent will receive 1 problem at a time.

        You can add up to THREE (3) the predictions to the
        predictions list provided below that you need to
        return at the end of this method.

        In the Autograder, the test data output in the arc problem will be set to None
        so your agent cannot peek at the answer (even on the public problems).

        Also, if you return more than 3 predictions in the list it
        is considered an ERROR and the test will be automatically
        marked as INCORRECT.
        """

        predictions: list[np.ndarray] = list()

        '''
        The next 2 lines are only an example of how to populate the predictions list.
        This will just be an empty answer the size of the input data;
        delete it before you start adding your own predictions.
        '''
        # output = np.zeros_like(arc_problem.test_set().get_input_data().data())
        # predictions.append(output)


        # predictions.append(test_array)

        basic_transformations = [
            self.rotate90,
            self.rotate180,
            self.rotate270,
            self.flip_horizontal,
            self.flip_vertical,
            self.transpose,
            self.trim,
        ]

        test_data_transformations = []


        # for training in arc_problem._training_data:
        #     input_data = training._input._arc_array
        #     output_data = training._output._arc_array

        #     for transform in basic_transformations:
        #         transformed_input = transform(input_data)
        #         transformed_output = transform(output_data)

        #         score = self.combined_score(transformed_output, transformed_input)
        #         print(f"Score for transformation {transform.__name__}: {score}")

        training_transformations = []
        average_score = 0
        for transformation in basic_transformations:
            for training in arc_problem._training_data:
                input_data = training._input._arc_array
                output_data = training._output._arc_array

                transformed = transformation(input_data)
                score = self.combined_score(output_data, transformed)
                average_score += score
            average_score = average_score / len(arc_problem._training_data)
            # print(f"Average score for transformation {transformation.__name__}: {average_score}")

            training_transformations.append((transformation, average_score))

            average_score = 0

        # print(training_transformations)
        # trim grid 
        for training in arc_problem._training_data:
            input_data = training._input._arc_array
            output_data = training._output._arc_array

            transformed = self.trim(input_data)
            score = self.combined_score(output_data, transformed)
            average_score += score
 
        average_score = average_score / len(arc_problem._training_data)
        training_transformations.append((self.trim, average_score))
        average_score = 0


        # swap colors
        get_first_two_colors = []
        for training in arc_problem._training_data:
            input_data = training._input._arc_array
            output_data = training._output._arc_array

            get_first_two_colors = self.top_two_colors(input_data)
            if len(get_first_two_colors) < 2:
                continue
            else: 
                transformed = self.flip_colors(input_data, get_first_two_colors[0], get_first_two_colors[1])
                score = self.combined_score(output_data, transformed)
                average_score += score
                get_first_two_colors = []

        if len(get_first_two_colors) == 2:
            average_score = average_score / len(arc_problem._training_data)
            training_transformations.append((self.flip_colors, average_score))
       
        average_score = 0


        training_transformations.sort(key=lambda x: x[1], reverse=True)

        top_3 = training_transformations[:3]
        print("Top 3 transformations:")
        test_data_transformations = []
        for transform, score in top_3:
            print(f"{transform.__name__} with average score {score}")
            test_data = transform(arc_problem._test._input._arc_array)
            test_data_transformations.append((transform, arc_problem._test._output._arc_array, score))
            # predictions.append(test_data)
            print(test_data)





        # training_transformations = []
        # average_score = 0
        # for transformation in basic_transformations:
        #     for training in test_data_transformations:
        #         input_data = training[0](arc_problem._test._input._arc_array)
        #         output_data = training[1]

        #         transformed = transformation(input_data)
        #         score = self.combined_score(output_data, transformed)
        #         average_score += score
        #     average_score = average_score / len(arc_problem._training_data)
        #     # print(f"Average score for transformation {transformation.__name__}: {average_score}")

        #     training_transformations.append((transformation, average_score))

        #     average_score = 0

        # # print(training_transformations)
        # # trim grid 
        # for training in arc_problem._training_data:
        #     input_data = training._input._arc_array
        #     output_data = training._output._arc_array

        #     transformed = self.trim(input_data)
        #     score = self.combined_score(output_data, transformed)
        #     average_score += score
 
        # average_score = average_score / len(arc_problem._training_data)
        # training_transformations.append((self.trim, average_score))
        # average_score = 0


        # # swap colors
        # get_first_two_colors = []
        # for training in arc_problem._training_data:
        #     input_data = training._input._arc_array
        #     output_data = training._output._arc_array

        #     get_first_two_colors = self.top_two_colors(input_data)
        #     if len(get_first_two_colors) < 2:
        #         continue
        #     else: 
        #         transformed = self.flip_colors(input_data, get_first_two_colors[0], get_first_two_colors[1])
        #         score = self.combined_score(output_data, transformed)
        #         average_score += score
        #         get_first_two_colors = []

        # if len(get_first_two_colors) == 2:
        #     average_score = average_score / len(arc_problem._training_data)
        #     training_transformations.append((self.flip_colors, average_score))
       
        # average_score = 0
        
        # predictions.append(np.array([
        #     [3,3,3,0],
        #     [3,0,3,0],
        #     [0,3,0,3],
        #     [3,3,3,0]
        # ]))

        
        # predictions.append(np.array([
        #     [3,3,3,0],
        #     [3,0,3,0],
        #     [0,3,0,3],
        #     [3,3,3,0]
        # ]))

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
    def trim(grid, background=0):
        rows = np.any(grid != background, axis=1)
        cols = np.any(grid != background, axis=0)
        
        if not rows.any() or not cols.any():
            return grid  # nothing to trim
        
        return grid[rows][:, cols]
    
    @staticmethod
    def recolor(grid, old_color, new_color):
        new_grid = grid.copy()
        new_grid[new_grid == old_color] = new_color
        return new_grid
    
    @staticmethod
    def flip_colors(grid, color1, color2):
        new_grid = grid.copy()
        
        temp = -1  # must not be a valid ARC color (ARC uses 0–9)
        
        new_grid[new_grid == color1] = temp
        new_grid[new_grid == color2] = color1
        new_grid[new_grid == temp] = color2
        
        return new_grid


    @staticmethod
    def top_two_colors(grid):
        # Flatten grid
        flat = grid.flatten()
        
        # Remove background (0)
        non_zero = flat[flat != 0]
        
        if len(non_zero) == 0:
            return []

        # Count occurrences
        values, counts = np.unique(non_zero, return_counts=True)

        # Sort by frequency (descending)
        sorted_indices = np.argsort(-counts)

        # Get top two
        top_colors = values[sorted_indices][:2]

        return top_colors.tolist()



    
    def get_percentage_accuracy(self, training_output, prediction):
        # Calculate the percentage of pixels that are correct in the prediction and account for the size difference 
        # of the training output and prediction
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
            total_diff += abs(true_dict.get(color, 0) - 
                            pred_dict.get(color, 0))

        # Normalize difference
        return 1 - (total_diff / (2 * total_pixels))
    
    def combined_score(self, training_output, prediction):
        return 0.8 * self.get_percentage_accuracy(training_output, prediction) + \
            0.2 * self.color_content_score(training_output, prediction)


 
