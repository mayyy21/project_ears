import sys
import time
from typing import List, Optional, Union
from os import path
import os
import shutil

import numpy as np
#import onnx
from onnxruntime import InferenceSession, NodeArg
#from onnx import ModelProto
from scipy.io import wavfile

from featureextraction import extract_features




def get_take_user_input() -> int:
    """Get user input

    Returns:
        int: [description]
    """
    print("Do you want to test a single audio?")
    print("Press '1' to enter filename or any key to test entire test set...")
    take: int = int(float(input().strip()))
    return take


def print_input_output_info(sess: InferenceSession) -> None:
    # print input and output node info
    print("input nodes info:")
    for input_node in sess.get_inputs():
        print(input_node)
    print("output nodes info:")
    for output_node in sess.get_outputs():
        print(output_node)


def get_audio_filename(target: str):
    #print("Enter the filename from test audio sample collection:")
    #filename: str = input().strip()
    filename: str = str(os.listdir(target))
#    print("1filename: ", filename)
    filename = filename[2:len(filename) - 2]
#    print("2filename: ", filename)
    return filename


def load_feature_vector(filename: str, base_folder: Union[str, None] = None):
    audio_path: str
    audio_path = filename
#    if take == 1:
#    audio_path = get_audio_filename(target = base_folder)

#    else:
        # TODO implement a way to select input audio
#        audio_path = get_audio_filename()

    if base_folder is not None:
        audio_path = path.join(base_folder, audio_path)
    sr, audio = wavfile.read(audio_path)
    feature_vector: np.ndarray = extract_features(audio, sr)
    return feature_vector


def get_inference_session(onnx_filename: str) -> InferenceSession:
    sess: InferenceSession = InferenceSession(
        onnx_filename, providers=["CPUExecutionProvider"])
    return sess


#def main(argc: int, argv: List[str]):
def detect() -> bool:
    #ONNX_FILENAME: str = "su.onnx"
    ONNX_FOLDER: str = "onnx_model/"
    BASE_FOLDER: str = "targetFile/"
    HISTORY_FOLDER: str = "history/"
    onnx_filenames: List[str] = os.listdir(ONNX_FOLDER)
    sessions: List[InferenceSession] = list()
    for file in onnx_filenames:
        sess: InferenceSession = get_inference_session(ONNX_FOLDER + file)
        sessions.append(sess)
#    sess: InferenceSession = get_inference_session(ONNX_FILENAME)
    # print node info
#    print("input nodes info:")
#    for input_node in sess.get_inputs():
#        print(input_node)
#    print("output nodes info:")
#    for output_node in sess.get_outputs():
#        print(output_node)
    #take: int = get_take_user_input()

    theFile: str
    theFile = get_audio_filename(target = BASE_FOLDER)
    feature_vector: np.ndarray = load_feature_vector(filename = theFile, base_folder=BASE_FOLDER)
    feature_vector = feature_vector.astype(np.float32)
    print(feature_vector.shape)

    outputs: List[np.adarray] = list()
    labels: List[np.adarray] = list()
    probabilities: List[np.adarray] = list()
    scores: List[np.adarray] = list()
    for sess in sessions:
        output = sess.run(output_names=None, input_feed={"X": feature_vector})
        outputs.append(output)
        label: np.ndarray = output[0]
        labels.append(label)
        probability: np.ndarray = output[1]
        probabilities.append(probability)
        score: np.ndarray = output[2]
        scores.append(score)

    avg_scores: List[float] = list()
    for score in scores:
        avg_score: float = score.mean()
        print(avg_score)
        avg_scores.append(avg_score)
    winner_index = avg_scores.index(max(avg_scores))

    if avg_scores[winner_index] >= -15:
        print("detect as :", onnx_filenames[winner_index])
        print("score: ", avg_scores[winner_index])
        return True
    else:
        print("sorry, failed to detect")
        return False
    #os.remove(BASE_FOLDER + theFile) #delete after detect
    #shutil.move(BASE_FOLDER + theFile, HISTORY_FOLDER) #move to history

#testing functionality of detect
#result: bool = detect()
#print(result)

    #print(str.format("score: {}", avg_score))
    # if take == 1:
    #     print("Enter the File name from Test Audio Sample Collection :")
    #     path: str = input().strip()
    #     print("Testing Audio : ", path)
    #     sr, audio = wavfile.read(source + path)
    #     vector: np.ndarray = extract_features(audio, sr)
    #     print(str.format("vector_shape {}", vector.shape))

    #     log_likelihood: float = 0.0
    #     sess: InferenceSession = InferenceSession(model)
    #     print(sess.get_inputs())
    #     print(sess.get_outputs())
    #     gmm = models[i]  #checking with each model one by one
    #     scores = np.array(gmm.score(vector))
    #     log_likelihood = scores.sum()
    #     # #print(i)
    #     # #print(log_likelihood[i])

    #     # for i in range(len(models)):
    #     #     gmm = models[i]  #checking with each model one by one
    #     #     scores = np.array(gmm.score(vector))
    #     #     log_likelihood[i] = scores.sum()
    #     #     #print(i)avg_score
    #     #     #print(log_likelihood[i])
    #     winner = np.argmax(log_likelihood)

    #     print("\tdetected as - ", speakers[winner])

    #     time.sleep(1.0)
    # elif take == 0:
    #     test_file = "testSamplePath.txt"
    #     file_paths = open(test_file, 'r')

    #     # Read the test directory and get the list of test audio files
    #     for path in file_paths:

    #         total_sample += 1.0
    #         path = path.strip()
    #         print("Testing Audio : ", path)
    #         sr, audio = read(source + path)
    #         vector = extract_features(audio, sr)

    #         log_likelihood = np.zeros(len(models))

    #         for i in range(len(models)):
    #             gmm = models[i]  #checking with each model one by one
    #             scores = np.array(gmm.score(vector))
    #             log_likelihood[i] = scores.sum()

    #         winner = np.argmax(log_likelihood)
    #         print("\tdetected as - ", speakers[winner])

    #         checker_name = path.split("_")[0]
    #         if speakers[winner] != checker_name:
    #             error += 1
    #         time.sleep(1.0)

    #     print(error, total_sample)
    #     accuracy = ((total_sample - error) / total_sample) * 100

    #     print(
    #         "The Accuracy Percentage for the current testing Performance with MFCC + GMM is : ",
    #         accuracy, "%")

    # print("Hurrey ! Speaker identified. Mission Accomplished Successfully. ")


#if __name__ == '__main__':
#    args: List[str] = sys.argv[1:]
#    main(len(args), args)
