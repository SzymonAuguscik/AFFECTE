class Tags:
    SYMBOLS_TO_IGNORE = ('', '\x01 Aux', 'MISSB', 'PSE', 'MB', 'M')
    AF_SYMBOL = '(AFIB'
    CLASSIFICATION_SYMBOLS = [ '(N', AF_SYMBOL]
    ARRHYTHMIA_SYMBOLS = [ *CLASSIFICATION_SYMBOLS, '(SVTA', '(VT', '(B', '(T', '(IVR', '(AB', '(SBR' ]

    NOTES_TO_ANNOTATIONS = {
        '(AFIB' : 'A',
        '(AFL'  : 'A',
        '(J'    : 'J',
        '(N'    : 'N'
    }

class Paths:
    class Directories:
        DATA = "data"
        DATASETS = "datasets"
        RESULTS = "results"
        LONG_TERM_AF = "long-term-af-database-1.0.0/files"
        FOLDS = "folds"
    class Files:
        RECORDS = "RECORDS"
        FEATURES = "features.pt"
        LABELS = "labels.pt"
        RESULTS = "results.txt"

class Results:
    class Metrics:
        ACCURACY = "Dokładność"
        F1_SCORE = "Miara F1"
        PRECISION = "Precyzja"
        RECALL = "Czułość"
        LOSS = "Funkcja straty"

        TRAIN = "trening"
        TEST = "test"

        TRAIN_ACCURACY = f"{ACCURACY} ({TRAIN})"
        TRAIN_F1_SCORE = f"{F1_SCORE} ({TRAIN})"
        TRAIN_PRECISION = f"{PRECISION} ({TRAIN})"
        TRAIN_RECALL = f"{RECALL} ({TRAIN})"

        TEST_ACCURACY = f"{ACCURACY} ({TEST})"
        TEST_F1_SCORE = f"{F1_SCORE} ({TEST})"
        TEST_PRECISION = f"{PRECISION} ({TEST})"
        TEST_RECALL = f"{RECALL} ({TEST})"
    class Visualization:
        class Names:
            X_LABEL = "Liczba epok"
            LR_Y_LABEL = "Wartość metryki"
            LR_TITLE = "Krzywe uczenia podczas treningu"
            LOSS_TITLE = "Funkcja straty podczas treningu"
        class Files:
            EXTENSION = "svg"
            LOSS = f"loss.{EXTENSION}"
            ACCURACY = f"accuracy.{EXTENSION}"
            F1_SCORE = f"f1_score.{EXTENSION}"
            PRECISION = f"precision.{EXTENSION}"
            RECALL = f"recall.{EXTENSION}"

class CV:
    X_TRAIN = "X_train"
    Y_TRAIN = "y_train"
    X_TEST = "X_test"
    Y_TEST = "y_test"

class Time:
    SECONDS_IN_MINUTE = 60
    MINUTES_IN_HOUR = 60

class Hyperparameters:
    INITIAL_BIAS = 0.01
    class Names:
        TRANSFORMER_DIMENSION = "transformerDimension"
        ECG_CHANNELS = "ecgChannels"
        WINDOW_LENGTH = "windowLength"
        TRANSFORMER_HIDDEN_DIMENSION = "transformerHiddenDimension"

        SECONDS = "seconds"
        LEARNING_RATE = "learningRate"
        BATCH_SIZE = "batchSize"
        EPOCHS = "epochs"

    class Cnn:
        class Layer1:
            class Conv:
                KERNEL_SIZE = 30
                STRIDE = 1
                PADDING = 1
            class MaxPool:
                KERNEL_SIZE = 5
                STRIDE = 1
                PADDING = 0
            DROPOUT = 0.03
        class Layer2:
            class Conv:
                KERNEL_SIZE = 10
                STRIDE = 1
                PADDING = 1
            class MaxPool:
                KERNEL_SIZE = 4
                STRIDE = 1
                PADDING = 0
            DROPOUT = 0.03
        class Layer3:
            class Conv:
                KERNEL_SIZE = 10
                STRIDE = 1
                PADDING = 1
            class MaxPool:
                KERNEL_SIZE = 4
                STRIDE = 2
                PADDING = 0
            DROPOUT = 0.03
        class Layer4:
            class Conv:
                KERNEL_SIZE = 5
                STRIDE = 1
                PADDING = 1
            class MaxPool:
                KERNEL_SIZE = 4
                STRIDE = 2
                PADDING = 0
            DROPOUT = 0.03
        class Layer5:
            class Conv:
                KERNEL_SIZE = 5
                STRIDE = 1
                PADDING = 1
            class MaxPool:
                KERNEL_SIZE = 3
                STRIDE = 1
                PADDING = 0
            DROPOUT = 0.03
        class Layer6:
            class Conv:
                KERNEL_SIZE = 3
                STRIDE = 1
                PADDING = 1
            class MaxPool:
                KERNEL_SIZE = 3
                STRIDE = 1
                PADDING = 0
            DROPOUT = 0.03
        class Layer7:
            class Conv:
                KERNEL_SIZE = 3
                STRIDE = 1
                PADDING = 1
            class MaxPool:
                KERNEL_SIZE = 2
                STRIDE = 1
                PADDING = 0
            DROPOUT = 0.03

    class Transformer:
        # HIDDEN_LAYER = 256
        # N_HEAD = 16
        LAYER_NORM_EPS = 1e-6

    class FeedForwardClassifier:
        HIDDEN_LAYER_DIMENSION = 1024
        class Layer1:
            DROPOUT = 0.03
        class Layer2:
            DROPOUT = 0.03
