import random
import json
from tqdm import tqdm

def collect_emotion_features():
    emotion_dict = {}
    missing_list = ''
    for i in tqdm(range(0, 70000)):
        file_num = str(i)
        file_num = '0' * (5 - len(file_num)) + file_num   # pad to 5 digits
        with open(f'data/ffhq-features/features/{file_num}.json', 'r') as f:
            try:
                data = json.load(f)[0]
            except:
                emotion_dict[file_num] = 'MISSING'
                missing_list += file_num + '\n'
                continue
            top_emotion = max(data['faceAttributes']['emotion'], key=data['faceAttributes']['emotion'].get)
            emotion_dict[file_num] = top_emotion

    with open('data/ffhq-features/emotion_dict.json', 'w') as f:
        json.dump(emotion_dict, f)

    with open('data/ffhq-features/missing_feat.txt', 'w') as f:
        f.write(missing_list)

def emotion_dict_to_value():
    emotions = {"anger": 0,
                "contempt": 1,
                "disgust": 2,
                "fear": 3,
                "happiness": 4,
                "neutral": 5,
                "sadness": 6,
                "surprise": 7}
                # "MISSING": 8}   # come up with better solution
    emotion_dict_numeric = {}
    with open('data/ffhq-features/emotion_dict.json', 'r') as f:
        data = json.load(f)
        for file_number, emotion in data.items():
            if emotion == 'MISSING':
                continue
            emotion_dict_numeric[file_number] = emotions[emotion]

    # Save to somewhere
    with open('data/ffhq-features/emotion_numeric_class.json', 'w') as f:
        json.dump(emotion_dict_numeric, f)

def create_train_test_val_txt():    
    with open('data/ffhq-features/emotion_numeric_class.json', 'r') as f:
        data = json.load(f)

    size = len(data)
    train_len = int(size * 0.7)
    val_len = int(size * 0.15)
    test_len = val_len

    file_nums = list(data.keys())
    random.Random(0).shuffle(file_nums)
    train_file_nums = file_nums[0:train_len]
    val_file_nums = file_nums[train_len:train_len + val_len]
    test_file_nums = file_nums[train_len + val_len:]
    print(f'Split of {len(train_file_nums), len(val_file_nums), len(test_file_nums)}')

    output_files = ['ffhqtrain.txt', 'ffhqvalidation.txt', 'ffhqtest.txt']
    num_list = [train_file_nums, val_file_nums, test_file_nums]
    for out_file, numbers in zip(output_files, num_list):
        with open('data/' + out_file, 'w') as f:
            for file_num in numbers:
                sub_file_directory = str(int(file_num) // 1000 * 1000)    # round down to nearest 1000
                if len(sub_file_directory) < 5:
                    sub_file_directory = '0' * (5 - len(sub_file_directory)) + sub_file_directory
                file_path = 'images1024x1024/' + sub_file_directory + '/' + file_num + '.png\n'
                f.write(file_path)


if __name__=='__main__':
    collect_emotion_features()
    emotion_dict_to_value()
    create_train_test_val_txt()