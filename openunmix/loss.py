from openunmix import transforms
import torch
import auraloss

eps = 1.e-10


def _custom_mse_loss(pred_magnitude, target_magnitude):
    loss = 0
    for i in range(len(target_magnitude)):
        loss += torch.mean((pred_magnitude[i] - target_magnitude[i])**2)
    return loss/len(target_magnitude)


# cross-umx multi-target losses
# from https://github.com/JeffreyCA/spleeterweb-xumx/blob/ddcc75e97ce8b374964347e216020e1faa6bc009/xumx/loss.py
class LossCriterion:
    def __init__(self, encoder, mix_coef):
        self.nsgt, self.insgt, self.cnorm = encoder
        self.mcoef = mix_coef
        self.sdr_loss_criterion = auraloss.time.SISDRLoss()

    def __call__(
            self,
            pred_magnitude_1,
            pred_magnitude_2,
            pred_magnitude_3,
            pred_magnitude_4,
            target_magnitude_1,
            target_magnitude_2,
            target_magnitude_3,
            target_magnitude_4,
            target_waveform_1,
            target_waveform_2,
            target_waveform_3,
            target_waveform_4,
            mix_complex,
            samples
        ):
        sdr_loss = 0

        with torch.no_grad():
            # SDR losses with time-domain waveforms
            pred_waveform_1 = self.insgt(
                transforms.phasemix_sep(mix_complex, pred_magnitude_1),
                samples
            )
            pred_waveform_2 = self.insgt(
                transforms.phasemix_sep(mix_complex, pred_magnitude_2),
                samples
            )
            pred_waveform_3 = self.insgt(
                transforms.phasemix_sep(mix_complex, pred_magnitude_3),
                samples
            )
            pred_waveform_4 = self.insgt(
                transforms.phasemix_sep(mix_complex, pred_magnitude_4),
                samples
            )

            # 4C1 Combination Losses
            sdr_loss_1 = self.sdr_loss_criterion(pred_waveform_1, target_waveform_1)
            sdr_loss_2 = self.sdr_loss_criterion(pred_waveform_2, target_waveform_2)
            sdr_loss_3 = self.sdr_loss_criterion(pred_waveform_3, target_waveform_3)
            sdr_loss_4 = self.sdr_loss_criterion(pred_waveform_4, target_waveform_4)

            # 4C2 Combination Losses
            sdr_loss_5 = self.sdr_loss_criterion(pred_waveform_1+pred_waveform_2, target_waveform_1+target_waveform_2)
            sdr_loss_6 = self.sdr_loss_criterion(pred_waveform_1+pred_waveform_3, target_waveform_1+target_waveform_3)
            sdr_loss_7 = self.sdr_loss_criterion(pred_waveform_1+pred_waveform_4, target_waveform_1+target_waveform_4)
            sdr_loss_8 = self.sdr_loss_criterion(pred_waveform_2+pred_waveform_3, target_waveform_2+target_waveform_3)
            sdr_loss_9 = self.sdr_loss_criterion(pred_waveform_2+pred_waveform_4, target_waveform_2+target_waveform_4)
            sdr_loss_10 = self.sdr_loss_criterion(pred_waveform_3+pred_waveform_4, target_waveform_3+target_waveform_4)

            # 4C3 Combination Losses
            sdr_loss_11 = self.sdr_loss_criterion(pred_waveform_1+pred_waveform_2+pred_waveform_3, target_waveform_1+target_waveform_2+target_waveform_3)
            sdr_loss_12 = self.sdr_loss_criterion(pred_waveform_1+pred_waveform_2+pred_waveform_4, target_waveform_1+target_waveform_2+target_waveform_4)
            sdr_loss_13 = self.sdr_loss_criterion(pred_waveform_1+pred_waveform_3+pred_waveform_4, target_waveform_1+target_waveform_3+target_waveform_4)
            sdr_loss_14 = self.sdr_loss_criterion(pred_waveform_2+pred_waveform_3+pred_waveform_4, target_waveform_2+target_waveform_3+target_waveform_4)

            # All 14 Combination Losses (4C1 + 4C2 + 4C3)
            sdr_loss = (sdr_loss_1 + sdr_loss_2 + sdr_loss_3 + sdr_loss_4 + sdr_loss_5 + sdr_loss_6 + sdr_loss_7 + sdr_loss_8 + sdr_loss_9 + sdr_loss_10 + sdr_loss_11 + sdr_loss_12 + sdr_loss_13 + sdr_loss_14)/14.0

        # MSE losses with frequency-domain magnitude slicq transform

        # 4C1 Combination Losses
        mse_loss_1 = _custom_mse_loss(pred_magnitude_1, target_magnitude_1)
        mse_loss_2 = _custom_mse_loss(pred_magnitude_2, target_magnitude_2)
        mse_loss_3 = _custom_mse_loss(pred_magnitude_3, target_magnitude_3)
        mse_loss_4 = _custom_mse_loss(pred_magnitude_4, target_magnitude_4)

        # 4C2 Combination Losses
        mse_loss_5 = _custom_mse_loss(pred_magnitude_1+pred_magnitude_2, target_magnitude_1+target_magnitude_2)
        mse_loss_6 = _custom_mse_loss(pred_magnitude_1+pred_magnitude_3, target_magnitude_1+target_magnitude_3)
        mse_loss_7 = _custom_mse_loss(pred_magnitude_1+pred_magnitude_4, target_magnitude_1+target_magnitude_4)
        mse_loss_8 = _custom_mse_loss(pred_magnitude_2+pred_magnitude_3, target_magnitude_2+target_magnitude_3)
        mse_loss_9 = _custom_mse_loss(pred_magnitude_2+pred_magnitude_4, target_magnitude_2+target_magnitude_4)
        mse_loss_10 = _custom_mse_loss(pred_magnitude_3+pred_magnitude_4, target_magnitude_3+target_magnitude_4)

        # 4C3 Combination Losses
        mse_loss_11 = _custom_mse_loss(pred_magnitude_1+pred_magnitude_2+pred_magnitude_3, target_magnitude_1+target_magnitude_2+target_magnitude_3)
        mse_loss_12 = _custom_mse_loss(pred_magnitude_1+pred_magnitude_2+pred_magnitude_4, target_magnitude_1+target_magnitude_2+target_magnitude_4)
        mse_loss_13 = _custom_mse_loss(pred_magnitude_1+pred_magnitude_3+pred_magnitude_4, target_magnitude_1+target_magnitude_3+target_magnitude_4)
        mse_loss_14 = _custom_mse_loss(pred_magnitude_2+pred_magnitude_3+pred_magnitude_4, target_magnitude_2+target_magnitude_3+target_magnitude_4)

        # All 14 Combination Losses (4C1 + 4C2 + 4C3)
        mse_loss = (mse_loss_1 + mse_loss_2 + mse_loss_3 + mse_loss_4 + mse_loss_5 + mse_loss_6 + mse_loss_7 + mse_loss_8 + mse_loss_9 + mse_loss_10 + mse_loss_11 + mse_loss_12 + mse_loss_13 + mse_loss_14)/14.0

        final_loss = self.mcoef*sdr_loss + mse_loss

        #print(f'sdr_loss: {sdr_loss}, mse_loss: {mse_loss}, sdr_mix: {self.mcoef*sdr_loss}, final_loss: {final_loss}')
        return final_loss
