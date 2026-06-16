import os
import pickle
import mfcc
import vq


def recognize(test_file, model_dir, frame_size=256, frame_shift=128,
              num_filters=26, num_ceps=12, fft_size=512):
    signal, sample_rate = mfcc.read_wav(test_file)
    features = mfcc.extract_mfcc(signal, sample_rate,
                                 frame_size=frame_size,
                                 frame_shift=frame_shift,
                                 num_filters=num_filters,
                                 num_ceps=num_ceps,
                                 fft_size=fft_size)

    best_speaker = None
    best_distortion = float('inf')

    for i in range(1, 9):
        model_path = os.path.join(model_dir, f"s{i}.pkl")
        with open(model_path, 'rb') as f:
            codebook = pickle.load(f)

        distortion = vq.compute_distortion(features, codebook)

        if distortion < best_distortion:
            best_distortion = distortion
            best_speaker = i

    return best_speaker, best_distortion


def test_all(test_dir, model_dir, frame_size=256, frame_shift=128,
             num_filters=26, num_ceps=12, fft_size=512):
    correct = 0
    total = 0

    for i in range(1, 9):
        test_file = os.path.join(test_dir, f"s{i}.wav")
        predicted, distortion = recognize(test_file, model_dir,
                                          frame_size=frame_size,
                                          frame_shift=frame_shift,
                                          num_filters=num_filters,
                                          num_ceps=num_ceps,
                                          fft_size=fft_size)
        total += 1
        status = "CORRECT" if predicted == i else "WRONG"
        if predicted == i:
            correct += 1
        print(f"Test S{i}.wav -> Recognized as S{predicted} "
              f"(distortion={distortion:.4f}) [{status}]")

    accuracy = correct / total * 100.0
    print(f"\nAccuracy: {correct}/{total} = {accuracy:.1f}%")

    return correct, total


if __name__ == '__main__':
    test_all('D:/data/TEST', 'D:/JIEDAN/models')
