import numpy as np

from config import MC_PASSES


def mc_predict(model, images, passes=MC_PASSES, batch_size=8):
    accumulator = None
    square_accumulator = None
    for _ in range(passes):
        preds = model.predict(images, batch_size=batch_size, verbose=0)
        if accumulator is None:
            accumulator = preds.astype(np.float64)
            square_accumulator = preds.astype(np.float64) ** 2
        else:
            accumulator += preds
            square_accumulator += preds ** 2
    mean_probs = accumulator / passes
    variance = square_accumulator / passes - mean_probs ** 2
    variance = np.clip(variance, 0.0, None)
    uncertainty = variance.sum(axis=-1)
    return mean_probs, uncertainty


def deterministic_predict(model, images, batch_size=8):
    return model.predict(images, batch_size=batch_size, verbose=0)


def predict_labels(mean_probs):
    return mean_probs.argmax(axis=-1)


def case_uncertainty(uncertainty_map, foreground):
    if foreground.sum() == 0:
        return float(uncertainty_map.mean())
    return float(uncertainty_map[foreground].mean())
