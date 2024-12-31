"""Evaluation functions"""

from pystoi.stoi import stoi
import torch
from pesq import pesq
import numpy as np
import tqdm
from joblib import Parallel, delayed


def STOI(signal_pred, signal_target, f_sampling, extended=False):
    """Compute the Short Time Objective Intelligibility between a denoised signal and a target signal given a sampling frequency
    La STOI mesure l'intelligibilité, pas la qualité globale du son. Pour capturer la qualité du son on utilise le PESQ
    Ici on fixe extended=False comme dans les articles originaux car on n'a pas de transformation non linéaire comme une compression ou une perte de paquet.

    Args:
        signal_pred (1D array):  float tensor of the denoised signal
        signal_target (1D array): float tensor of the target signal
        fs (int):  sampling frequency
        extended (bool): parameter of STOI

    Returns:
        score (tensor):  float scalar tensor
    """
    score = stoi(
        torch.tensor(signal_target),
        torch.tensor(signal_pred),
        f_sampling,
        extended=extended,
    )
    return score


def PESQ(signal_pred, signal_target, f_sampling, band="nb", class_=False):
    """Perceptual Evaluation of Speech Quality

    Args:
        signal_pred (1D array): Signal prédit
        signal_target (1D array): Signal de référence
        f_sampling (int): Fréquence d'échantillonnage
        band (str, optional): Bande passante, "nb" pour Narrowband, "wb" pour Wideband. Défaut à "nb".
        class_ (bool, optional): Si True, classe le score PESQ.

    Returns:
        float: Le score PESQ (qualité perçue du signal)
        str (si class_ == True): La classification basée sur PESQ.
    """

    # Calcul du score PESQ
    pesq_score = pesq(f_sampling, signal_target, signal_pred, band)

    if class_:
        # Classification selon les scores du cours
        if pesq_score < 1:
            classification = "Impossible de comprendre"
        elif 1 <= pesq_score < 2:
            classification = "Impossible de comprendre"
        elif 2 <= pesq_score < 2.4:
            classification = "Effort considérable pour comprendre"
        elif 2.4 <= pesq_score < 2.8:
            classification = "Effort modéré pour comprendre"
        elif 2.8 <= pesq_score < 3.3:
            classification = "Attention nécessaire et effort léger"
        elif 3.3 <= pesq_score < 3.8:
            classification = "Attention nécessaire"
        elif 3.8 <= pesq_score <= 4.5:
            classification = "Aucun effort requis"
        else:
            classification = "Score PESQ hors de portée"

        return pesq_score, classification

    return pesq_score


def compute_snr(signal_target, signal_pred):
    noise_power = np.sum((signal_target - signal_pred) ** 2)
    signal_power = np.sum(signal_target**2)

    if noise_power != 0:
        snr = 10 * np.log10(signal_power / noise_power)
    else:
        snr = np.inf

    return snr


def eval_signal(signal_pred, signal_target, f_sampling):
    """Calcul MSE, RMSE, SNR, PESQ, STOI entre deux signaux

    Args:
        signal_target (array-like): Signal cible
        signal_pred (array-like): Signal prédit ou débruité

    Returns:
        tuple: MSE, RMSE et SNR
    """
    # Conversion si besoin
    signal_target = np.array(signal_target)
    signal_pred = np.array(signal_pred)

    # MSE
    mse = np.mean((signal_target - signal_pred) ** 2)

    # RMSE
    rmse = np.sqrt(mse)

    # SNR
    snr = compute_snr(signal_target, signal_pred)

    # STOI
    stoi_result = STOI(signal_pred, signal_target, f_sampling, extended=False)

    # PESQ
    pesq_result = PESQ(signal_pred, signal_target, f_sampling, band="nb", class_=False)

    return mse, rmse, snr, stoi_result, pesq_result


def eval_all_signals(list_signals_pred, list_signals_target, f_sampling, n_jobs=-1):
    """Iterations parallèles sur tous les signaux avec joblib"""

    # Utilisation de joblib pour paralléliser l'évaluation
    results = Parallel(n_jobs=n_jobs)(
        delayed(eval_signal)(signal_pred, list_signals_target[i], f_sampling)
        for i, signal_pred in tqdm.tqdm(
            enumerate(list_signals_pred), total=len(list_signals_pred)
        )
    )

    # Séparer les résultats dans les listes correspondantes
    MSE, RMSE, SNR, STOI_results, PESQ_results = zip(*results)

    return list(MSE), list(RMSE), list(SNR), list(STOI_results), list(PESQ_results)


# def eval_all_signals(list_signals_pred, list_signals_target, f_sampling):
#     " Iterations sur tous les signaux "
#     MSE = []
#     RMSE = []
#     SNR = []
#     STOI_results = []
#     PESQ_results = []
#     for i, signal_pred in tqdm.tqdm(enumerate(list_signals_pred), total=len(list_signals_pred)):
#         signal_target = list_signals_target[i]
#         mse, rmse, snr, stoi_result, pesq_result = eval_signal(signal_pred, signal_target, f_sampling)
#         MSE.append(mse)
#         RMSE.append(rmse)
#         SNR.append(snr)
#         STOI_results.append(stoi_result)
#         PESQ_results.append(pesq_result)
#     return MSE, RMSE, SNR, STOI_results, PESQ_results
