import os
import pickle
import mfcc
import vq


def train_all(train_dir, model_dir, frame_size=256, frame_shift=128,
              num_filters=26, num_ceps=12, fft_size=512, codebook_size=16):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    for i in range(1, 9):
        filename = f"s{i}.wav"
        filepath = os.path.join(train_dir, filename)

        signal, sample_rate = mfcc.read_wav(filepath)
        features = mfcc.extract_mfcc(signal, sample_rate,
                                     frame_size=frame_size,
                                     frame_shift=frame_shift,
                                     num_filters=num_filters,
                                     num_ceps=num_ceps,
                                     fft_size=fft_size)

        codebook = vq.lbg_train(features, target_size=codebook_size)

        model_path = os.path.join(model_dir, f"s{i}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(codebook, f)

        print(f"Speaker S{i}: trained with {features.shape[0]} frames, "
              f"codebook size {codebook.shape[0]}")


if __name__ == '__main__':
    train_all('D:/data/TRAIN', 'D:/JIEDAN/models')
