import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import baseline_ridge as base

OUT = base.OUTPUT_DIRECTORY
LANDMARKS = OUT / "landmarks_mp_raw"
AUDIO = OUT / "audio_lld_raw"
ID = base.ID_COLUMN
MAX_LEN = 150          # seconds kept per person (median is 70, max 248)
N_PCA_RESIDUAL = 18    # 16 regional + 16 within-person delta + 18 residual PCA = 50 dims
MIN_FACE_FRACTION = 0.5
MAX_LENGTH_GAP = 5     # seconds between face and audio sequences
ZERO_AS_MISSING = [
    "jitterLocal", "shimmerLocaldB", "logRelF0-H1-A3",
    "logRelF0-H1-H2", "F0semitoneFrom27.5Hz", "HNRdBACF"
]
warnings.filterwarnings("ignore", category=RuntimeWarning)


def select_people():
    dataset = pd.read_csv(OUT / "dataset_all.csv")
    words = base.load_feature_table(dataset[ID].tolist())[[ID, "word_count"]]
    check = pd.read_csv(OUT / "sequence_check.csv")
    table = dataset.merge(words, on=ID, how="left").merge(check, on=ID, how="left")
    reason = pd.Series("", index=table.index)
    rules = [
        ("missing audio or landmarks", table["n_face"].isna()),
        ("empty transcript", table["word_count"].fillna(0) == 0),
        ("face detected < 50%", table["face_detected"] < MIN_FACE_FRACTION),
        ("face/audio length gap", table["length_gap"] > MAX_LENGTH_GAP),
    ]
    for name, condition in rules:
        reason[(reason == "") & condition] = name
    table["excluded_reason"] = reason
    print("Exclusions (first matching rule):")
    print(reason[reason != ""].value_counts().to_string())
    table[table["excluded_reason"] != ""][[ID, "excluded_reason"]].to_csv(
        OUT / "sequence_exclusions.csv", index=False)
    kept = table[table["excluded_reason"] == ""].reset_index(drop=True)
    print("Kept:", len(kept), "| train:", (kept["split"] == "train").sum(),
          "| test:", (kept["split"] == "test").sum())
    return kept


def align_and_extract_4regions(points):
    """
    points: (T, 468, 2)
    1. Align face in-plane rotation (roll) using outer eye corners (idx 33, 263)
       and normalize scale by interocular distance (IOD).
    2. Extract 16 targeted regional features (Mouth/Smile, Eyes/EAR, Brows, Cheeks, Pose & Interaction).
    Returns:
       aligned_flat: (T, 936)
       regional_16:  (T, 16)
    """
    # Eye centers using standard MediaPipe canonical indices
    left_eye = 0.5 * (points[:, 33, :] + points[:, 133, :])   # (T, 2)
    right_eye = 0.5 * (points[:, 263, :] + points[:, 362, :]) # (T, 2)
    eye_mid = 0.5 * (left_eye + right_eye)                    # (T, 2)

    d_eye = right_eye - left_eye                              # (T, 2)
    iod = np.linalg.norm(d_eye, axis=1, keepdims=True)        # (T, 1) Interocular distance
    iod = np.where((iod < 1e-6) | np.isnan(iod), 1.0, iod)

    # Rotate each frame so the interocular line is horizontal (removes head tilt/roll)
    angle = np.arctan2(d_eye[:, 1], d_eye[:, 0])              # (T,)
    cos_a = np.cos(-angle)[:, None]
    sin_a = np.sin(-angle)[:, None]

    c = (points - eye_mid[:, None, :]) / iod[:, :, None]      # (T, 468, 2)
    x_rot = c[:, :, 0] * cos_a - c[:, :, 1] * sin_a
    y_rot = c[:, :, 0] * sin_a + c[:, :, 1] * cos_a
    aligned = np.stack([x_rot, y_rot], axis=2)                # (T, 468, 2)

    def dist(i, j):
        return np.linalg.norm(aligned[:, i, :] - aligned[:, j, :], axis=1)

    # --- REGION 1: MOUTH & SMILE (AU12, AU15, AU25/26) ---
    smile_width = dist(61, 291)                               # Mouth width (AU12 Smile)
    mouth_open = dist(13, 14)                                 # Inner lip height (Mouth open)
    mouth_outer_h = dist(0, 17)                               # Outer lip height
    mouth_aspect = mouth_open / (smile_width + 1e-4)
    # In image coords, y points down: smaller y means corners pulled UP (smiling)
    mouth_center_y = 0.5 * (aligned[:, 13, 1] + aligned[:, 14, 1])
    corner_lift_l = mouth_center_y - aligned[:, 61, 1]
    corner_lift_r = mouth_center_y - aligned[:, 291, 1]
    corner_lift = 0.5 * (corner_lift_l + corner_lift_r)       # Positive = smile, Negative = droop
    lip_asymmetry = np.abs(corner_lift_l - corner_lift_r)     # Smirk / asymmetry

    # --- REGION 2: EYES & EYELID DROOP (EAR / AU5 / AU43) ---
    ear_l = (dist(159, 145) + dist(158, 153)) / (2.0 * dist(33, 133) + 1e-4)
    ear_r = (dist(386, 374) + dist(385, 380)) / (2.0 * dist(263, 362) + 1e-4)
    ear_mean = 0.5 * (ear_l + ear_r)
    ear_diff = np.abs(ear_l - ear_r)

    # --- REGION 3: EYEBROWS (AU1 Inner Brow Raise, AU4 Brow Lowerer/Frown) ---
    brow_furrow = dist(107, 336)                              # Inner brow distance (smaller = frown)
    brow_lift_l = aligned[:, 159, 1] - aligned[:, 105, 1]     # Eye-to-brow vertical height
    brow_lift_r = aligned[:, 386, 1] - aligned[:, 334, 1]
    brow_lift = 0.5 * (brow_lift_l + brow_lift_r)

    # --- REGION 4: CHEEKS, POSE & DUCHENNE INTERACTION (AU6 + AU12) ---
    # Shorter mouth-corner to eye-corner distance = Cheek Raise (AU6)
    cheek_raise = -0.5 * (dist(61, 33) + dist(291, 263))
    # Duchenne marker: simultaneous smile lift and eye squint (smaller EAR)
    duchenne_proxy = (smile_width + corner_lift) * (-ear_mean)
    # Head pitch proxy (nose tip y relative to eye-line, normalized by IOD)
    head_pitch = aligned[:, 1, 1]
    # Head yaw proxy (nose tip x relative to eye midpoint)
    head_yaw = aligned[:, 1, 0]
    # Jaw drop (chin tip 152 to nose tip 1)
    jaw_drop = dist(1, 152)

    regional_16 = np.stack([
        smile_width, mouth_open, mouth_outer_h, mouth_aspect,
        corner_lift, lip_asymmetry, ear_mean, ear_diff,
        brow_furrow, brow_lift, cheek_raise, duchenne_proxy,
        head_pitch, head_yaw, jaw_drop, np.abs(head_yaw)
    ], axis=1).astype(np.float32)

    return aligned.reshape(len(points), -1), regional_16


def load_person(identifier, zero_columns, f0_column, loudness_column):
    face = np.load(LANDMARKS / f"{identifier}.npy")
    voice = np.load(AUDIO / f"{identifier}.npy")
    length = min(len(face), len(voice), MAX_LEN)
    face, voice = face[:length], voice[:length].copy()

    detected = ~np.isnan(face[:, 0])
    points = face.reshape(length, -1, 2)

    aligned_flat = np.zeros((length, points.shape[1] * 2), dtype=np.float32)
    regional = np.zeros((length, 16), dtype=np.float32)
    delta_regional = np.zeros((length, 16), dtype=np.float32)
    residual_flat = np.zeros((length, points.shape[1] * 2), dtype=np.float32)

    if detected.any():
        af, reg = align_and_extract_4regions(points[detected])
        aligned_flat[detected] = af
        regional[detected] = reg
        # Within-person baseline subtraction: removes static skull shape, keeps pure expression dynamics
        person_face_median = np.nanmedian(reg, axis=0, keepdims=True)
        delta_regional[detected] = reg - person_face_median
        residual_flat[detected] = af - np.nanmedian(af, axis=0, keepdims=True)

    # Better voiced / active speech detection per second:
    # A second is truly active speech if F0 > 0 AND loudness is above the person's 20th percentile
    raw_f0 = voice[:, f0_column]
    raw_loud = voice[:, loudness_column]
    valid_loud = raw_loud[~np.isnan(raw_loud)]
    loud_thresh = np.percentile(valid_loud, 20) if len(valid_loud) > 5 else -np.inf
    voiced = ((raw_f0 > 0) & (~np.isnan(raw_f0)) & (raw_loud >= loud_thresh)).astype(np.float32)

    block = voice[:, zero_columns]
    voice[:, zero_columns] = np.where(block == 0, np.nan, block)
    return residual_flat, regional, delta_regional, detected, voice, voiced


def main():
    people = select_people()
    columns = pd.read_csv(OUT / "audio_lld_columns.csv", header=None)[0].tolist()
    zero_columns = [i for i, c in enumerate(columns) if any(c.startswith(p) for p in ZERO_AS_MISSING)]
    f0_column = [i for i, c in enumerate(columns) if c.startswith("F0semitoneFrom27.5Hz")][0]
    loud_candidates = [i for i, c in enumerate(columns) if "loudness" in c.lower() or "Loudness" in c]
    loudness_column = loud_candidates[0] if loud_candidates else 0
    assert len(zero_columns) == len(ZERO_AS_MISSING), (len(zero_columns), "check descriptor names")

    loaded = [load_person(i, zero_columns, f0_column, loudness_column) for i in people[ID]]
    is_train = (people["split"] == "train").to_numpy()

    # Audio standardization from train seconds only.
    train_voice = np.concatenate([loaded[k][4] for k in np.flatnonzero(is_train)])
    audio_mean, audio_std = np.nanmean(train_voice, axis=0), np.nanstd(train_voice, axis=0)
    audio_std[audio_std < 1e-8] = 1.0

    # Standardize the 32 interpretable face features (16 regional + 16 within-person delta) on train
    train_reg32 = np.concatenate([
        np.hstack([loaded[k][1][loaded[k][3]], loaded[k][2][loaded[k][3]]])
        for k in np.flatnonzero(is_train) if loaded[k][3].any()
    ])
    reg_mean = np.nanmean(train_reg32, axis=0)
    reg_std = np.nanstd(train_reg32, axis=0)
    reg_std[reg_std < 1e-8] = 1.0

    # PCA WITHOUT whitening on within-person dynamic landmark residuals (18 dims -> total 50 face dims)
    train_res_frames = np.concatenate([
        loaded[k][0][loaded[k][3]] for k in np.flatnonzero(is_train) if loaded[k][3].any()
    ])
    rng = np.random.default_rng(base.RANDOM_SEED)
    if len(train_res_frames) > 60000:
        train_res_frames = train_res_frames[rng.choice(len(train_res_frames), 60000, replace=False)]
    pca = PCA(n_components=N_PCA_RESIDUAL, whiten=False, random_state=base.RANDOM_SEED).fit(train_res_frames)
    pca_std = np.sqrt(pca.explained_variance_[0]) + 1e-8  # Scale proportionally by PC1 std (preserves variance hierarchy!)
    print("Dynamic PCA explained variance (%d components, whiten=False): %.3f" % (
        N_PCA_RESIDUAL, pca.explained_variance_ratio_.sum()))

    n = len(people)
    n_face_total = 32 + N_PCA_RESIDUAL  # 50 features total
    audio = np.zeros((n, MAX_LEN, len(columns) + 1), dtype=np.float32)
    face = np.zeros((n, MAX_LEN, n_face_total), dtype=np.float32)
    valid = np.zeros((n, MAX_LEN), dtype=bool)
    face_ok = np.zeros((n, MAX_LEN), dtype=bool)

    for k, (res_flat, reg, delta_reg, detected, voice, voiced) in enumerate(loaded):
        length = len(res_flat)
        z = (voice - audio_mean) / audio_std
        person_mean = np.nanmean(z, axis=0)
        person_mean = np.where(np.isnan(person_mean), 0.0, person_mean)
        z = np.where(np.isnan(z), person_mean, z)
        audio[k, :length, :-1] = z
        audio[k, :length, -1] = voiced

        if detected.any():
            reg32_z = (np.hstack([reg[detected], delta_reg[detected]]) - reg_mean) / reg_std
            pca18_z = pca.transform(res_flat[detected]) / pca_std
            face[k, :length][detected] = np.hstack([reg32_z, pca18_z])

        valid[k, :length] = True
        face_ok[k, :length] = detected

    np.savez_compressed(
        OUT / "sequences.npz",
        ids=people[ID].to_numpy().astype(str), split=people["split"].to_numpy().astype(str),
        targets=people[base.TARGET_COLUMNS].to_numpy(dtype=np.float32),
        audio=audio, face=face, valid=valid, face_ok=face_ok,
    )
    print("Saved", OUT / "sequences.npz", "| audio", audio.shape, "| face", face.shape)


if __name__ == "__main__":
    main() 