##############################################
# voice recognition
##############################################
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
##############################################
#sound position
##############################################
import pyaudio
import RPi.GPIO as GPIO
from time import sleep
import wave
import datetime
from pydub import AudioSegment
import threading
import shutil
##############################################
# set GPIO
##############################################
def tonum(num):
        fm=10.0/180
        num=num*fm+2.5
        num=int(num*10)/10.0
        return num

dir = 0
tail_flag = True

rightservo=27
leftservo=17
headservo=3
tailservo=4

m_left11=10
m_left12=22
m_left21=11
m_left22=9
m_right11=13
m_right12=26
m_right21=23
m_right22=15

enc11=25
enc12=8
enc21=12
enc22=7
#enable=24

trigger=21
echo=16

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

GPIO.setup(leftservo,GPIO.OUT,initial=False)
GPIO.setup(rightservo,GPIO.OUT,initial=False)
GPIO.setup(headservo,GPIO.OUT,initial=False)

GPIO.setup(m_left11,GPIO.OUT,initial=False)
GPIO.setup(m_left12,GPIO.OUT,initial=False)
GPIO.setup(m_left21,GPIO.OUT,initial=False)
GPIO.setup(m_left22,GPIO.OUT,initial=False)
GPIO.setup(m_right11,GPIO.OUT,initial=False)
GPIO.setup(m_right12,GPIO.OUT,initial=False)
GPIO.setup(m_right21,GPIO.OUT,initial=False)
GPIO.setup(m_right22,GPIO.OUT,initial=False)

GPIO.setup(enc11,GPIO.IN,pull_up_down=GPIO.PUD_UP)
GPIO.setup(enc12,GPIO.IN,pull_up_down=GPIO.PUD_UP)
GPIO.setup(enc21,GPIO.IN,pull_up_down=GPIO.PUD_UP)
GPIO.setup(enc22,GPIO.IN,pull_up_down=GPIO.PUD_UP)
#GPIO.setup(enable,GPIO.OUT)
#GPIO.output(enable,0)

GPIO.setup(tailservo,GPIO.OUT,initial=False)

GPIO.setup(trigger,GPIO.OUT)
GPIO.setup(echo,GPIO.IN)

m_fre=80
le=GPIO.PWM(leftservo,50)
re=GPIO.PWM(rightservo,50)
head=GPIO.PWM(headservo,50)
left11=GPIO.PWM(m_left11,m_fre)
left12=GPIO.PWM(m_left12,m_fre)
left21=GPIO.PWM(m_left21,m_fre)
left22=GPIO.PWM(m_left22,m_fre)
right11=GPIO.PWM(m_right11,m_fre)
right12=GPIO.PWM(m_right12,m_fre)
right21=GPIO.PWM(m_right21,m_fre)
right22=GPIO.PWM(m_right22,m_fre)

tail=GPIO.PWM(tailservo,50)

left11.start(0)
left12.start(0)
left21.start(0)
left22.start(0)
right11.start(0)
right12.start(0)
right21.start(0)
right22.start(0)

le.start(tonum(90))
re.start(tonum(90))
head.start(tonum(90))
tail.start(tonum(90))
sleep(0.1)
le.ChangeDutyCycle(0)
re.ChangeDutyCycle(0)
head.ChangeDutyCycle(0)
tail.ChangeDutyCycle(0)
sleep(0.01)

q=[0,10,20,30,45,50,60,70,80,90,100,110,120,135,140,150,160,170,180]

##############################################
# function for setting up pyserial
##############################################
#
def pyserial_start():
    audio = pyaudio.PyAudio() # create pyaudio instantiation
    ##############################
    ### create pyaudio stream  ###
    # -- streaming can be broken down as follows:
    # -- -- format             = bit depth of audio recording (16-bit is standard)
    # -- -- rate               = Sample Rate (44.1kHz, 48kHz, 96kHz)
    # -- -- channels           = channels to read (1-2, typically)
    # -- -- input_device_index = index of sound device
    # -- -- input              = True (let pyaudio know you want input)
    # -- -- frmaes_per_buffer  = chunk to grab and keep in buffer before reading
    ##############################
    stream = audio.open(format = pyaudio_format,rate = samp_rate,channels = chans, \
                        input_device_index = dev_index,input = True, \
                        frames_per_buffer=CHUNK)
    stream.stop_stream() # stop stream to prevent overload
    return stream,audio

def pyserial_end():
    stream.close() # close the stream
    audio.terminate() # close the pyaudio connection

def check_i2s():
    i2s_name = 'snd_rpi_i2s_card'
    p = pyaudio.PyAudio()
    flag = False
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
#        print((i,dev['name'],dev['maxInputChannels']))
        if i2s_name in dev['name']:
            flag = True
            break
    if not flag:
        sys.exit()
    return i

##############################################
# record
##############################################
def data_saver():
#    shutil.rmtree("./targetFile")
#    os.mkdir("./targetFile")
    data_folder = './targetFile/' # folder where data will be saved locally
#    root_folder = '../' # folder where data will be saved locally
#    if os.path.isdir(data_folder)==False:
#        os.mkdir(data_folder) # create folder if it doesn't exist
    filename = datetime.datetime.now().strftime('%m_%d_%H_%M_%S') # filename based on recording time
    wf = wave.open(data_folder+filename+'.wav','wb') # open .wav file for saving
    wf.setnchannels(chans) # set channels in .wav file
    wf.setsampwidth(audio.get_sample_size(pyaudio_format)) # set bit depth in .wav file
    wf.setframerate(samp_rate) # set sample rate in .wav file
    stream.read(samp_rate,exception_on_overflow=False)
#    print(3)
#    sleep(1)
#    print(2)
#    sleep(1)
#    print(1)
#    sleep(1)
    print('Recording Started.')
    for i in range(0,5):
        record = stream.read(CHUNK,exception_on_overflow=False)
        wf.writeframes(record)
    print('Recording Stopped.')
    wf.close() # close .wav file
    old_sound = AudioSegment.from_wav(data_folder+filename+'.wav')
    loud_sound = old_sound + 40
    loud_sound.export(data_folder+filename+'.wav',format='wav')
    global tail_flag
    tail_flag = False

##############################################
# move car
##############################################
def stop():
    print("stop")
    left11.ChangeDutyCycle(0)
    left12.ChangeDutyCycle(0)
    left21.ChangeDutyCycle(0)
    left22.ChangeDutyCycle(0)
    right11.ChangeDutyCycle(0)
    right12.ChangeDutyCycle(0)
    right21.ChangeDutyCycle(0)
    right22.ChangeDutyCycle(0)

def forward():
    print ("Forward")
    left11.ChangeDutyCycle(25)
    left12.ChangeDutyCycle(0)
    left21.ChangeDutyCycle(25)
    left22.ChangeDutyCycle(0)
    right11.ChangeDutyCycle(25)
    right12.ChangeDutyCycle(0)
    right21.ChangeDutyCycle(25)
    right22.ChangeDutyCycle(0)
    #return

def back():
    print ("back")
    left11.ChangeDutyCycle(0)
    left12.ChangeDutyCycle(25)
    left21.ChangeDutyCycle(0)
    left22.ChangeDutyCycle(25)
    right11.ChangeDutyCycle(0)
    right12.ChangeDutyCycle(25)
    right21.ChangeDutyCycle(0)
    right22.ChangeDutyCycle(25)
    #return

def left():
    print ("left")
    left11.ChangeDutyCycle(0)
    left12.ChangeDutyCycle(45)
    left21.ChangeDutyCycle(0)
    left22.ChangeDutyCycle(45)
    right11.ChangeDutyCycle(55)
    right12.ChangeDutyCycle(0)
    right21.ChangeDutyCycle(55)
    right22.ChangeDutyCycle(0)

def right():
    print ("right")
    left11.ChangeDutyCycle(55)
    left12.ChangeDutyCycle(0)
    left21.ChangeDutyCycle(55)
    left22.ChangeDutyCycle(0)
    right11.ChangeDutyCycle(0)
    right12.ChangeDutyCycle(45)
    right21.ChangeDutyCycle(0)
    right22.ChangeDutyCycle(45)
    #return

##################################
# set callback
##################################
counter11=0
counter21=0
dir_flag = True
thread11_flag = False
thread21_flag = False
dir_name = ''
start_time=0
end_time=0
distance=0
echo_flag = False


def counter_threshold():
    global thread11_flag
    global thread21_flag
    global dir_flag
    global tail_flag

    if dir_name == 'left':
       if counter21 >= 10:
           thread11_flag = False
           thread21_flag = False
           dir_flag = True
           stop()
    elif dir_name == 'right':
       if counter11 >= 10:
           thread11_flag = False
           thread21_flag = False
           dir_flag = True
           stop()
    elif dir_name == 'forward':
#       diff_time=end_time-start_time
#       distance=17150*diff_time
       print('distance in:',distance)
       if distance >= 20 and distance <= 70:
           tail_flag = False
           stop()

def callback11(channel11):
#    print('callback11')
    global counter11
    global dir_flag
#    if GPIO.event_detected(enc11):
    if thread11_flag:
        counter11 += 1
        counter_threshold()
#        print('counter11:',counter11)
#        if counter11 >= 5:
#            thread11_flag = False
#            GPIO.setup(enable,GPIO.OUT)
#            GPIO.output(enable,0)
#            dir_flag = True
#            stop()

def callback21(channel21):
#    print('callback21')
    global counter21
    global dir_flag
#    if GPIO.event_detected(enc21):
    if thread21_flag:
        counter21 += 1
        counter_threshold()
#        print('counter21:',counter21)
#        if counter21 >= 5:
#            thread_flag = False
#            GPIO.setup(enable,GPIO.OUT)
#            GPIO.output(enable,0)
#            dir_flag = True
#            stop()

def callback_echo(channel_echo):
#    print('echo')
    global start_time
    global end_time
    global distance
    global echo_flag
    if GPIO.input(echo)==1:
        start_time=time.time()
    elif GPIO.input(echo)==0:
        end_time=time.time()
        diff_time=abs(end_time-start_time)
        echo_flag = True
        if diff_time > 0:
            distance=17150*diff_time
#            print('distance:',distance)
            counter_threshold()

GPIO.add_event_detect(enc11,GPIO.RISING,callback=callback11)
GPIO.add_event_detect(enc21,GPIO.RISING,callback=callback21)
GPIO.add_event_detect(echo,GPIO.BOTH,callback=callback_echo)

##################################
# mic localization
##################################
def mic_localization():
    ##############################
    ###### initial variable ######
    data = []
    left_arr = []
    right_arr = []
    l_amp_max = 0
    r_amp_max = 0
    l_max_indx = -1
    r_max_indx = -1
    indx_arr = []
    l_r_value = []
    pos_indx = [0] * 6
    pos_arr = []
    global dir
    ##############################

    ##############################
    # stream info
    ##############################
    stream_data = stream.read(CHUNK,exception_on_overflow=False)
    data.append(np.frombuffer(stream_data,dtype=buffer_format))
    for chan in range(0,chans):
        data_chunks = [data[ii][chan:][::2] for ii in range(0,np.shape(data)[0])]
        if chan == 0:
            for frame in data_chunks:
                left_arr.extend(frame)
        elif chan == 1:
            for frame in data_chunks:
                right_arr.extend(frame)
    ##############################

    ##############################
    # get peak index diff
    ##############################
    for indx in range(0, len(left_arr)):
        if left_arr[indx] > l_amp_max:
            l_amp_max = left_arr[indx]
            l_max_indx = indx
        if right_arr[indx] > r_amp_max:
            r_amp_max = right_arr[indx]
            r_max_indx = indx
        if indx % clip == 0 and l_amp_max > noice_amp and r_amp_max > noice_amp:
            if l_max_indx != -1 and r_max_indx != -1:
                l_r_value.append((left_arr[l_max_indx], right_arr[r_max_indx],\
                                  left_arr[l_max_indx]-right_arr[r_max_indx]))
                indx_arr.append(l_max_indx - r_max_indx)
            l_max_indx = -1
            r_max_indx = -1
            l_amp_max = 0
            r_amp_max = 0
    ##############################

    ##############################
    # counter each direction
    ##############################
    for item in range(0, len(indx_arr)):
        if abs(indx_arr[item]) < 35:
#            print('indx_diff:', indx_arr[item])
#            print('left amp:', l_r_value[item][0], ',right amp:', l_r_value[item][1],\
#                  'l-r:', l_r_value[item][2])
            if indx_arr[item] >= -4 and indx_arr[item] <= 4: #in the degree 90
                pos_indx[3] = pos_indx[3] + 1
            elif indx_arr[item] > 4 and indx_arr[item] <= 13: #in the degree 135
                pos_indx[2] = pos_indx[2] + 1
            elif indx_arr[item] > 13: #in the degree 180
                pos_indx[1] = pos_indx[1] + 1
            elif indx_arr[item] >= -13 and indx_arr[item] < -4: #in the degree 45
                pos_indx[4] = pos_indx[4] + 1
            elif indx_arr[item] < -13: #in the degree 0
                pos_indx[5] = pos_indx[5] + 1
    ##############################
    if any(pos_indx):
        pos_max = max(pos_indx)
        if pos_max > 3:
#            pos_arr.append(pos_indx.index(pos_max))
            flag = True
            for elem in range(1, len(pos_indx)):
                if pos_indx[elem] == pos_max and elem != pos_indx.index(pos_max):
                    flag = False
                    break
            if flag:
                final_pos = pos_indx.index(pos_max)
                if final_pos != 0:
                    dir = final_pos
                    print('dir: ',dir)
                print('pos:', final_pos, ', pos value:', pos_max)
#                    pos_arr.append(elem)
#            for pos in pos_arr:
#                print('pos:', pos, ', pos value:', pos_indx[pos])
#                print('len',len(pos_arr))
    print('')
#    return pos_arr
#    return final_pos

##################################
# set thread
##################################
def clean_gpio():
    le.ChangeDutyCycle(0)
    re.ChangeDutyCycle(0)
    head.ChangeDutyCycle(0)
    tail.ChangeDutyCycle(0)
    sleep(0.01)

def trigger_thread():
#    print('trigger')
    global echo_flag
    while tail_flag:
        echo_flag = False
        GPIO.output(trigger,GPIO.LOW)
        GPIO.output(trigger,GPIO.HIGH)
        sleep(0.000015)
        GPIO.output(trigger,GPIO.LOW)
        while echo_flag==False:
            sleep(0.01)

def turn_tail(mode):
#    print('tail')
    arr_count = 0
    if mode == 'slow':
       tail_arr = [80,70,60,50,40,50,60,70,80,90,100,110,120,130,140,130,120,\
                   110,100,90]
    elif mode == 'fast':
       tail_arr = [70,90,110,90]
    tail_len = len(tail_arr)-1
    while tail_flag:
        if arr_count > tail_len:
           arr_count = 0
        tail.ChangeDutyCycle(tonum(tail_arr[arr_count]))
        sleep(0.05)
        clean_gpio()
        arr_count += 1

###############################################
# void recognition function
###############################################
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
    #print(feature_vector.shape)

    outputs: List[np.adarray] = list()
    #labels: List[np.adarray] = list()
    #probabilities: List[np.adarray] = list()
    scores: List[np.adarray] = list()
    for sess in sessions:
        output = sess.run(output_names=None, input_feed={"X": feature_vector})
        outputs.append(output)
        #label: np.ndarray = output[0]
        #labels.append(label)
        #probability: np.ndarray = output[1]
        #probabilities.append(probability)
        score: np.ndarray = output[2]
        scores.append(score)

    avg_scores: List[float] = list()
    for score in scores:
        avg_score: float = score.mean()
#        print(onnx_filenames[scores.index(score)])
        print(avg_score)
        avg_scores.append(avg_score)
    winner_index = avg_scores.index(max(avg_scores))

    print(theFile)
    os.remove(BASE_FOLDER + theFile) #delete after detect
    #shutil.move(BASE_FOLDER + theFile, HISTORY_FOLDER) #move to history
    if avg_scores[winner_index] >= -19:
        print("detect as :", onnx_filenames[winner_index])
        print("score: ", avg_scores[winner_index])
        return True
    else:
        print("sorry, failed to detect")
        print("the highest score is", onnx_filenames[winner_index])
        print("score: ", avg_scores[winner_index])
        return False
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


##################################
# turn
##################################
# 1 to 5: 180 135 90 45 0
def turn_angle(dir_p):
#    if dir == 1:
#        le.ChangeDutyCycle(tonum(q[0]))
#        re.ChangeDutyCycle(tonum(q[18]))
#        print('dir: ',dir,'degree: ',q[18])

#        sleep(0.5)
    global dir_flag
    global dir
    global counter11
    global counter21
    global thread11_flag
    global thread21_flag
    global dir_name
    global tail_flag

    dir_flag = False

    if dir_p == 2 or dir_p == 1:
        re.ChangeDutyCycle(tonum(q[4]))
        le.ChangeDutyCycle(tonum(q[9]))
#        print('dir: ',dir,'degree: ',q[13])
        sleep(0.1)
        clean_gpio()
        sleep(0.5)
        re.ChangeDutyCycle(tonum(q[9]))
        sleep(0.1)
        clean_gpio()

        for i in range(88, 44, -2):
            head.ChangeDutyCycle(tonum(i))
            sleep(0.05)
            clean_gpio()

#        sleep(0.4)
        counter11 = 0
        counter21 = 0
#        GPIO.setup(enable,GPIO.IN)
        dir = 0
        thread11_flag = True
        thread21_flag = True
        dir_name = 'right'
        right()
#        turn_angle_sleep(2)
#        stop()
#        sleep_sensor(0.5)
        for i in range(46, 90, 2):
            head.ChangeDutyCycle(tonum(i))
            sleep(0.05)
            clean_gpio()

    elif dir_p == 3:
        le.ChangeDutyCycle(tonum(q[9]))
        re.ChangeDutyCycle(tonum(q[9]))
        head.ChangeDutyCycle(tonum(q[9]))
#        print('dir: ',dir,'degree: ',q[9])
        sleep(0.1)
        clean_gpio()
        dir = 0
        # check sound if T then go
        print('detecting')
#        t1=threading.Thread(target=turn_tail,args=('slow',))
#        t1.setDaemon(True)
#        t1.start()
        print('t1 start')
        data_saver()
#        t1.join()
        tail_flag = True
#        sleep(1)
        if detect():
            dir_name = 'forward'
            forward()
            t2=threading.Thread(target=trigger_thread)
            t2.setDaemon(True)
            t2.start()
#            GPIO.output(trigger,GPIO.LOW)
#            GPIO.output(trigger,GPIO.HIGH)
#            sleep(0.000015)
#            GPIO.output(trigger,GPIO.LOW)
            print('true')
            t3=threading.Thread(target=turn_tail,args=('fast',))
            print('t3 start')
            t3.setDaemon(True)
            t3.start()
            t2.join()
            t3.join()
        tail_flag = True
        dir_flag = True
        sleep(1)

    elif dir_p == 4 or dir_p == 5:
        re.ChangeDutyCycle(tonum(q[9]))
        le.ChangeDutyCycle(tonum(q[13]))
#        print('dir: ',dir,'degree: ',q[4])
        sleep(0.1)
        clean_gpio()
        sleep(0.5)
        le.ChangeDutyCycle(tonum(q[9]))
        sleep(0.1)
        clean_gpio()

        for i in range(92, 134, 2):
            head.ChangeDutyCycle(tonum(i))
            sleep(0.05)
            clean_gpio()

#        sleep(0.4)
        counter11=0
        counter21=0
#        GPIO.setup(enable,GPIO.IN)
        dir = 0
        thread11_flag = True
        thread21_flag = True
        dir_name = 'left'
        left()
#        turn_angle_sleep(2)
#        stop()
#        sleep_sensor(0.5)
        for i in range(132, 90, -2):
            head.ChangeDutyCycle(tonum(i))
            sleep(0.05)
            clean_gpio()

#    if dir == 5:
#        p1.ChangeDutyCycle(tonum(q[0]))
#        p2.ChangeDutyCycle(tonum(q[0]))
#        print('dir: ',dir,'degree: ',q[0])

#        sleep(0.5)

#
##############################################
# Main Data Acquisition Procedure
##############################################
#
if __name__ == '__main__':
#    args: List[str] = sys.argv[1:]
#    main(len(args), args)
#
    ###########################
    # acquisition parameters
    ###########################
    #
    CHUNK          = 44100 // 2  # frames to keep in buffer between reads
    samp_rate      = 44100 # sample rate [Hz]
    pyaudio_format = pyaudio.paInt16 # 16-bit device
    buffer_format  = np.int16 # 16-bit for buffer
    chans          = 2 # only read 1 channel
    dev_index      = 2 # index of sound device
    #
    #############################
    # stream info and data saver
    #############################
    # initial variable
    #############################
    #
    stream,audio = pyserial_start() # start the pyaudio stream
    clip_count = 30
    clip = CHUNK / clip_count
    noice_amp = 20
    #
    #############################

    stream.start_stream()
    stream.read(samp_rate,exception_on_overflow=False)
    print('Detecting...')
##    data_saver()
##    print(detect())

#    GPIO.add_event_detect(enc21,GPIO.RISING,callback=callback21)
#    GPIO.add_event_detect(enc11,GPIO.RISING,callback=callback11)

#    left()
#    sleep(4)
#    while True:
#        forward()
#        sleep(3)
#        stop()
#        sleep(0.5)
#        right()
#        sleep(3)
#        stop()
#        sleep(0.5)
#        left()
#        sleep(3)
#        stop()
#        sleep(0.5)
#        back()
#        sleep(3)
#        stop()
#        sleep(0.5)

    while True:
        try:
##            dir_arr = []
            mic_localization()
##            if len(dir_arr) == 1:
##                 turn_angle(dir_arr[0])
            print('dir: ',dir, ',dir_flag: ',dir_flag)
            if dir_flag and dir != 0:
                turn_angle(dir)
##            print('distance:',distance)
##            print('counter11:',counter11)
##            print('counter21:',counter21)
##            left()
##            sleep(4)
##            stop()
##            sleep(0.5)
##            right()
##            sleep(2)
##            stop()
##            sleep(0.5)
##            forward()
##            sleep(0.5)
##            stop()
##            sleep(0.5)
##            back()
##            sleep(0.5)
        except KeyboardInterrupt:
            print('keyboard interrupt.')
            stream.stop_stream()
            pyserial_end()
            left11.stop(0)
            left12.stop(0)
            left21.stop(0)
            left22.stop(0)
            right11.stop(0)
            right12.stop(0)
            right21.stop(0)
            right22.stop(0)
            GPIO.cleanup()
            print('System Stopped.')
            sys.exit()
            raise

