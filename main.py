import train
import test

TRAIN_DIR = 'D:/data/TRAIN'
TEST_DIR = 'D:/data/TEST'
MODEL_DIR = 'D:/JIEDAN/models'

FRAME_SIZE = 256
FRAME_SHIFT = 128
NUM_FILTERS = 26
NUM_CEPS = 12
FFT_SIZE = 512
CODEBOOK_SIZE = 16

print("=" * 50)
print("SPEAKER RECOGNITION SYSTEM (MFCC + VQ/LBG)")
print("=" * 50)

print(f"\n[Parameters]")
print(f"  Frame size: {FRAME_SIZE} samples ({FRAME_SIZE/12500*1000:.1f} ms)")
print(f"  Frame shift: {FRAME_SHIFT} samples ({FRAME_SHIFT/12500*1000:.1f} ms)")
print(f"  Mel filters: {NUM_FILTERS}")
print(f"  MFCC coefficients: {NUM_CEPS}")
print(f"  FFT size: {FFT_SIZE}")
print(f"  Codebook size: {CODEBOOK_SIZE}")

print(f"\n[Phase 1: Training]")
print(f"  Training data: {TRAIN_DIR}")
print(f"  Model output: {MODEL_DIR}")
print("-" * 40)
train.train_all(TRAIN_DIR, MODEL_DIR,
                frame_size=FRAME_SIZE,
                frame_shift=FRAME_SHIFT,
                num_filters=NUM_FILTERS,
                num_ceps=NUM_CEPS,
                fft_size=FFT_SIZE,
                codebook_size=CODEBOOK_SIZE)

print(f"\n[Phase 2: Recognition]")
print(f"  Test data: {TEST_DIR}")
print(f"  Model source: {MODEL_DIR}")
print("-" * 40)
correct, total = test.test_all(TEST_DIR, MODEL_DIR,
                               frame_size=FRAME_SIZE,
                               frame_shift=FRAME_SHIFT,
                               num_filters=NUM_FILTERS,
                               num_ceps=NUM_CEPS,
                               fft_size=FFT_SIZE)

print("\n" + "=" * 50)
print(f"FINAL RESULT: {correct}/{total} ({correct/total*100:.1f}%)")
print("=" * 50)
