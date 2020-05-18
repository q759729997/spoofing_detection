# import pickle
# import codecs
import os

import cv2
import numpy as np
from sklearn.externals import joblib
# import joblib


def detect_face(img, faceCascade):
    faces = faceCascade.detectMultiScale(img,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(110, 110)
                                         #  flags = cv2.CV_HAAR_SCALE_IMAGE
                                         )
    return faces


def calc_hist(img):
    histogram = [0] * 3
    for j in range(3):
        histr = cv2.calcHist([img], [j], None, [256], [0, 256])
        histr *= 255.0 / histr.max()
        histogram[j] = histr
    return np.array(histogram)


def test_photo(clf_model, img_file, out_img_file):
    # width = 320
    # height = 240
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    # cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

    # # Initialize face detector
    cascPath = "./python_scripts/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)

    sample_number = 1
    count = 0
    measures = np.zeros(sample_number, dtype=np.float)
    # print('img_file:{}'.format(img_file))
    img_bgr = cv2.imread(img_file)
    # print('img_bgr:{}'.format(img_bgr))
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = detect_face(img_gray, faceCascade)
    measures[count % sample_number] = 0
    point = (0, 0)
    for i, (x, y, w, h) in enumerate(faces):

        roi = img_bgr[y:y + h, x:x + w]

        img_ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCR_CB)
        img_luv = cv2.cvtColor(roi, cv2.COLOR_BGR2LUV)

        ycrcb_hist = calc_hist(img_ycrcb)
        luv_hist = calc_hist(img_luv)

        feature_vector = np.append(ycrcb_hist.ravel(), luv_hist.ravel())
        feature_vector = feature_vector.reshape(1, len(feature_vector))

        prediction = clf_model.predict_proba(feature_vector)
        prob = prediction[0][1]

        measures[count % sample_number] = prob

        cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (255, 0, 0), 2)

        point = (x, y - 5)
        print(measures, np.mean(measures), img_file)
        if 0 not in measures:
            text = "True"
            if np.mean(measures) >= 0.7:
                text = "False"
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img=img_bgr,
                            text=text,
                            org=point,
                            fontFace=font,
                            fontScale=0.9,
                            color=(0, 0, 255),
                            thickness=2,
                            lineType=cv2.LINE_AA)
            else:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img=img_bgr,
                            text=text,
                            org=point,
                            fontFace=font,
                            fontScale=0.9,
                            color=(0, 255, 0),
                            thickness=2,
                            lineType=cv2.LINE_AA)

        count += 1
        # cv2.imshow('img_rgb', img_bgr)
        cv2.imwrite(out_img_file, img_bgr)

    # cap.release()
    cv2.destroyAllWindows()


# def load_serialize_obj(filename):
#     """ pickle加载.

#     @return:
#         On success - 文件内保存的对象.
#     """
#     with codecs.open(filename, 'rb') as fread:
#         return pickle.load(fread)


if __name__ == "__main__":
    """demo"""
    model_file_name = './trained_models/print-attack_trained_models/print-attack_ycrcb_luv_extraTreesClassifier.pkl'
    # model_file_name = './trained_models/replay_attack_trained_models/replay-attack_ycrcb_luv_extraTreesClassifier.pkl'
    clf_model = joblib.load(model_file_name)
    image_names = [
        'qiaoyongtian_true_1',
        'qiaoyongtian_true_2',
        'lijiale_true_1',
        'lijiale_true_2',
        'zhaoshengao_true_1',
        'zhaoshengao_true_2',
        'qiaoyongtian_false_1',
        'qiaoyongtian_false_2',
        'lijiale_false_1',
        'lijiale_false_2',
        'zhaoshengao_false_1',
        'zhaoshengao_false_2',
    ]
    for image_name in image_names:
        img_file = os.path.join('./data/images', '{}.jpg'.format(image_name))
        out_img_file = os.path.join('./data/images', 'masked_{}.jpg'.format(image_name))
        test_photo(clf_model, img_file, out_img_file)
    """print-attack
    [0.08571429] 0.08571428571428572 ./data/images\qiaoyongtian_true_1.jpg
    [0.38571429] 0.38571428571428573 ./data/images\qiaoyongtian_true_2.jpg
    [0.56] 0.5599999999999999 ./data/images\lijiale_true_1.jpg
    [0.58888889] 0.5888888888888889 ./data/images\lijiale_true_2.jpg
    [0.2] 0.2 ./data/images\zhaoshengao_true_1.jpg
    [0.2] 0.2 ./data/images\zhaoshengao_true_2.jpg
    [0.66] 0.6599999999999999 ./data/images\qiaoyongtian_false_2.jpg
    [0.78] 0.78 ./data/images\lijiale_false_1.jpg
    [0.49166667] 0.4916666666666666 ./data/images\lijiale_false_2.jpg
    [0.31428571] 0.3142857142857143 ./data/images\zhaoshengao_false_1.jpg
    [0.5] 0.5 ./data/images\zhaoshengao_false_2.jpg
    [0.11666667] 0.11666666666666667 ./data/images\zhaoshengao_false_2.jpg
    """
    """replay-attack
    [0.94285714] 0.9428571428571428 ./data/images\qiaoyongtian_true_1.jpg
    [0.89285714] 0.8928571428571429 ./data/images\qiaoyongtian_true_2.jpg
    [1.] 1.0 ./data/images\lijiale_true_1.jpg
    [1.] 1.0 ./data/images\lijiale_true_2.jpg
    [0.5875] 0.5875 ./data/images\zhaoshengao_true_1.jpg
    [0.5875] 0.5875 ./data/images\zhaoshengao_true_2.jpg
    [0.94166667] 0.9416666666666667 ./data/images\qiaoyongtian_false_2.jpg
    [0.975] 0.975 ./data/images\lijiale_false_1.jpg
    [0.85076923] 0.8507692307692307 ./data/images\lijiale_false_2.jpg
    [0.9] 0.9 ./data/images\zhaoshengao_false_1.jpg
    [0.78035714] 0.7803571428571429 ./data/images\zhaoshengao_false_2.jpg
    [0.5075] 0.5075000000000001 ./data/images\zhaoshengao_false_2.jpg
    """
