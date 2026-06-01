"""A module implementing utilities for sequence losses."""

# pylint: disable=no-member, invalid-name, import-error

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__author__ = "Chanwoo Kim(chanwcom@gmail.com)"
# Third-party imports
import torch

# Custom imports
import ctc_loss_lib

class CtcLoss(torch.autograd.Function):
    """A class for calculating the CTC loss."""

    @staticmethod
    def forward(ctx,
                labels,
                target_lens,
                logits,
                logits_len):
        """Calculates the Sequential Hypothesis Classifier (CTC) loss.

        Args:
            ctx: Contexts for this CtcLoss operation.
            labels: A tensor containing batch of ground-truth label sequences.
                Note that this label sequence should already include blank labels.
                The shape is given by (batch_size, max_target_len).
            target_lens: The lengths of labels that has the shape of
                (batch_size).
            logits: The predicted "logit value". The shape is given by
                (batch_size, max_logit_seq_len, num_classes).
            logits_len: The len of logits that has the shape of (batch_size).

        Note that zero values are assumed to be masked-values.

        Returns:
            A tuple containing (loss, grad)
        """
        # Checks whether the shape of labels is (B, L).
        assert labels.dim() == 2

        # Checks whether the shape of logits is (B, T, C)
        assert logits.dim() == 3

        # Checks the consistency of the batch size.
        assert labels.shape[0] == logits.shape[0]

        inputs = {}
        inputs["SEQ_DATA"] = labels
        inputs["SEQ_LEN"] = target_lens

        inputs = ctc_loss_lib.to_blank_augmented_labels(inputs, 0, False, False)

        labels = inputs["SEQ_DATA"]
        target_lens = inputs["SEQ_LEN"]

        clamped_labels = torch.clamp(labels, min=0)
        # Converting the sequences.
        # Note that the following is only for HuggingFace case.
        # In case of HuggingFace, the boundary blanks should be added and non
        # -blank token indices should NOT be updated.
        log_target_probs = ctc_loss_lib.calculate_log_label_prob(
            clamped_labels, torch.softmax(logits, dim=-1))

        trans_table = ctc_loss_lib.label_trans_allowance_table(
            labels, target_lens)

        # Alpha and beta should be calculated.
        log_alpha, log_beta, log_seq_prob = ctc_loss_lib.calculate_alpha_beta(
            trans_table, log_target_probs, target_lens, logits_len)

        # "gamma" is the posterior probability of the alignment variable $q_t$.
        #
        # The "alignment variable" $q_t$ is a random variable representing
        # the distribution  of the label sequcne index $l$ at time $t$.
        #
        # gamma is defined by:
        #   p(\mathbf{q_t} = l | \mathbbm{x}, \mathbbm{y}).
        #
        # gamma can be expressed in terms of \alpha and \beta as follows:
        #   gamma_{t, l} = sum_{l \in {l | q_t = l}} \alpha_{t, l} \beta{t, l}
        #                / sum_{l=0^L-1} \alpha_{t, l} \beta{t, l}.
        #
        # log_gamma is defined as follows:
        #   log p(q_t = l| x, y) where t is the temporal index, and l is the
        # blank-augmented label sequence index.
        # The shape of log_gamma is (batch_size, max_logits_len, max_target_len).
        log_gamma = log_alpha + log_beta
        log_gamma = log_gamma - torch.logsumexp(log_gamma, axis=2, keepdim=True)

        # Calculates an estimated time-aligned ground-truth sequence.
        #
        # log_ground_truth_prob is \tilde{\mathbbm{y}_t}.
        #
        # Scatter-add avoids constructing a separate class tensor per label.
        # --- (Optimized Alignment Block) ---
        # 1. Convert log_gamma to the probability domain (B, T, L).
        gamma = torch.exp(log_gamma)

        # 2. Initialize a tensor to store results (B, T, C).
        # Memory overhead is negligible since C is small (32~128).
        ground_truth_prob = torch.zeros_like(logits)

        # 3. Expand labels from (B, L) to (B, T, L) as a memory-efficient view.
        # clamped_labels shape is (batch_size, max_target_len).
        expanded_indices = clamped_labels.unsqueeze(1).expand(
            -1, logits.size(1), -1
        )

        # 4. Core operation: Accumulate probability values into class indices.
        # Scatter and add gamma values along dim=2 (class dimension).
        ground_truth_prob.scatter_add_(2, expanded_indices, gamma)

        # 5. Compute the final gradient.
        gradient = -(ground_truth_prob - torch.softmax(logits, dim=2))

        # Seqeunce mask
        seq_mask = ctc_loss_lib.sequence_mask(
            logits_len, maxlen=torch.max(logits_len))

        # The dimension of "gradient" is (batch_size, logit_len, num_classes)
        gradient = torch.multiply(gradient, torch.unsqueeze(seq_mask, axis=2))

        ctx.save_for_backward(gradient)

        loss = -log_seq_prob

        return loss

    @staticmethod
    def backward(ctx, grad):
        gradient, = ctx.saved_tensors

        gradient = torch.multiply(gradient, torch.reshape(grad, (-1, 1, 1)))

        return None, None, gradient, None
