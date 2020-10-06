from typing import Any

from LetterFinder import find_letters
from PredictImg import predict_img


def get_string(model: Any, image_file: str):
    letters = find_letters(image_file)
    result_string = ""
    for i in range(len(letters)):
        dn = letters[i+1][0] - letters[i][0] - letters[i][1] if i < len(letters) - 1 else 0
        result_string += predict_img(model, letters[i][2])
        if (dn > letters[i][1]/4):
            result_string += ' '
    return result_string